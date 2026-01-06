from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re
import csv
import os
import json
import argparse
import lxml.etree as ET
from dataclasses import dataclass


@dataclass
class BoundingBox:

    x_min: int | float
    x_max: int | float
    y_min: int | float
    y_max: int | float

    # ---- Derived ----
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    @property
    def width(self) -> float | int:
        return self.x_max - self.x_min

    @property
    def height(self) -> float | int:
        return self.y_max - self.y_min

    @property
    def area(self) -> float | int:
        return self.width * self.height


@dataclass
class UIElement:
    text: Optional[str] = None
    content_description: Optional[str] = None
    class_name: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    bbox_pixels: Optional[BoundingBox] = None
    hint_text: Optional[str] = None

    # state flags
    is_checked: Optional[bool] = None
    is_checkable: Optional[bool] = None
    is_clickable: Optional[bool] = None
    is_editable: Optional[bool] = None
    is_enabled: Optional[bool] = None
    is_focused: Optional[bool] = None
    is_focusable: Optional[bool] = None
    is_long_clickable: Optional[bool] = None
    is_scrollable: Optional[bool] = None
    is_selected: Optional[bool] = None
    is_visible: Optional[bool] = None

    # identifiers
    package_name: Optional[str] = None
    resource_name: Optional[str] = None
    resource_id: Optional[str] = None

    # tree relations
    self_id: Optional[int] = None
    parent_id: Optional[int] = None


def _parse_ui_hierarchy(xml_string: str) -> dict[str, Any]:
    """Parses the UI hierarchy XML into a dictionary structure."""
    root = ET.fromstring(xml_string, ET.XMLParser(resolve_entities=False))  # 使用 defusedxml 安全解析

    def parse_node(node):
        result = node.attrib
        result['children'] = [parse_node(child) for child in node]
        return result
    return parse_node(root)


def xml_dump_to_ui_elements(xml_string: str) -> List[UIElement]:
    """uiautomator dump → 列表[UIElement]，顺带记录 parent/self id。"""
    parsed = _parse_ui_hierarchy(xml_string)
    elements: List[UIElement] = []

    def walk(node: Dict[str, Any], parent_idx: int | None):
        nonlocal elements
        bounds = node.get("bounds")
        bbox = None
        if bounds:
            x_min, y_min, x_max, y_max = map(
                int, bounds.strip("[]").replace("][", ",").split(",")
            )
            bbox = BoundingBox(x_min, x_max, y_min, y_max)

        elem = UIElement(
            text=node.get("text"),
            content_description=node.get("content-desc"),
            class_name=node.get("class"),
            bbox=bbox,
            bbox_pixels=bbox,
            is_checked=node.get("checked") == "true",
            is_checkable=node.get("checkable") == "true",
            is_clickable=node.get("clickable") == "true",
            is_enabled=node.get("enabled") == "true",
            is_focused=node.get("focused") == "true",
            is_focusable=node.get("focusable") == "true",
            is_long_clickable=node.get("long-clickable") == "true",
            is_scrollable=node.get("scrollable") == "true",
            is_selected=node.get("selected") == "true",
            package_name=node.get("package"),
            resource_id=node.get("resource-id"),
            is_visible=True,
            self_id=len(elements),
            parent_id=parent_idx,
        )
        cur_idx = elem.self_id  # type: ignore[arg-type]
        elements.append(elem)

        for child in node.get("children", []):
            walk(child, cur_idx)

    walk(parsed, None)
    return elements


def _regex_match(pattern: str, target: Optional[str]) -> bool:
    if target is None:
        return False
    return bool(re.search(pattern, target.lower()))


def compare_single(rule: Dict[str, Any], elem: UIElement) -> bool:
    """按 rule 的字段对单元素做正则匹配。"""
    if "text" in rule and not _regex_match(rule["text"].lower(), elem.text):
        return False
    if "resource_id" in rule and not _regex_match(rule["resource_id"].lower(), elem.resource_id):
        return False
    if "content_description" in rule and not _regex_match(
        rule["content_description"].lower(), elem.content_description):
        return False
    if "class_name" in rule and not _regex_match(rule["class_name"].lower(), elem.class_name):
        return False
    # flag‑type fields
    for flag in ("is_checkable", "is_checked", "is_selected"):
        if flag in rule:
            if str(getattr(elem, flag)).lower() != str(rule[flag]).lower():
                return False
    return True


def compare_single_position(rule: Dict[str, Any], elem: UIElement, pos: Tuple[int, int]) -> bool:
    """同时匹配属性 + 点击坐标是否落在元素 bbox 内。"""
    if not compare_single(rule, elem):
        return False
    if elem.bbox is None:
        return False
    x, y = pos
    return elem.bbox.x_min <= x <= elem.bbox.x_max and elem.bbox.y_min <= y <= elem.bbox.y_max


def check_relation(page_rule: Dict[str, Any], anchor_elem: UIElement,
                   ui_elements: List[UIElement], relation: str) -> bool:
    """验证 anchor_elem 与 page_rule 元素的亲缘关系。"""
    for other in ui_elements:
        if relation == "parent" and other.parent_id == anchor_elem.self_id:
            if compare_single(page_rule, other):
                return True
        if relation == "sibling" and other.parent_id == anchor_elem.parent_id:
            if compare_single(page_rule, other):
                return True
        if relation == "child" and other.self_id == anchor_elem.parent_id:
            if compare_single(page_rule, other):
                return True
        if relation == "self" and other.self_id == anchor_elem.self_id:
            if compare_single(page_rule, other):
                return True
    return False


def compare(ui_elements: List[UIElement], key_nodes: Dict[str, Any], action_dict: Dict[str, Any]) -> bool:
    """整体规则匹配：page_rules + action_rules。"""
    page_rules: List[Dict[str, Any]] = key_nodes.get("page", [])
    action_rules: List[Dict[str, Any]] = key_nodes.get("action", [])

    checked_page = [False] * len(page_rules)
    checked_act = [False] * len(action_rules)
    recorded_rules: List[Dict[str, Any]] = []

    # ------- page 规则 -------
    for i, rule in enumerate(page_rules):
        if checked_page[i]:
            continue
        recorded_rules.append(rule)
        for elem in ui_elements:
            if compare_single(rule, elem):
                # 若有关系要求
                if "related" in rule:
                    anchor_rule = recorded_rules[rule["related"][0]["id"]]
                    if not check_relation(anchor_rule, elem, ui_elements, rule["related"][0]["relation"]):
                        continue
                checked_page[i] = True
                break

    # ------- action 规则 -------
    for j, rule in enumerate(action_rules):
        if checked_act[j]:
            continue
        if "position_in" not in rule or "params" not in action_dict:
            continue
        recorded_rules.append(rule["position_in"])
        click_pos = tuple(action_dict["params"].get("position", ()))  # type: ignore[arg-type]
        for elem in ui_elements:
            if compare_single_position(rule["position_in"], elem, click_pos):
                if "related" in rule:
                    anchor_rule = recorded_rules[rule["related"][0]["id"]]
                    if not check_relation(anchor_rule, elem, ui_elements, rule["related"][0]["relation"]):
                        continue
                checked_act[j] = True
                break

    return all(checked_page) and all(checked_act)


def bbox_contains_point(content, bounds, point):  # noqa: ANN001, D401
    """Return True if point lies within Android bounds string."""
    # 解析 bounds
    if isinstance(bounds, str):
        x1y1, x2y2 = bounds[1:-1].split("][")
        x1, y1 = map(int, x1y1.split(","))
        x2, y2 = map(int, x2y2.split(","))
        bounds_tuple = (x1, y1, x2, y2)
    elif isinstance(bounds, list) and len(bounds) == 1 and isinstance(bounds[0], str):
        # 处理 ['[0,94][1080,248]'] 这种情况
        bounds_str = bounds[0]
        x1y1, x2y2 = bounds_str[1:-1].split("][")
        x1, y1 = map(int, x1y1.split(","))
        x2, y2 = map(int, x2y2.split(","))
        bounds_tuple = (x1, y1, x2, y2)
    else:
        bounds_tuple = tuple(map(int, bounds))  # type: ignore[arg-type]

    # 解析 point
    if isinstance(point, str):
        x, y = map(int, point.split(","))
    else:
        x, y = map(int, point)  # type: ignore[arg-type]

    x1, y1, x2, y2 = bounds_tuple
    return x1 <= x <= x2 and y1 <= y <= y2


def evaluate_action_xml(xml: str, xpath: str, action_dict: Dict[str, Any]) -> Tuple[int, set[int]]:
    """在单份 XML 与一次 action 上评估 XPath；返回 (1|0, visited_nodes)"""
    visited_nodes: set[int] = set()
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.fromstring(xml.encode('utf-8'), parser)

    if not xpath:
        return 0, visited_nodes

    # 注册自定义函数
    ns = ET.FunctionNamespace(None)
    ns["bbox_contains_point"] = bbox_contains_point

    modified_xpath = xpath
    if "$point" in xpath and "params" in action_dict and "position" in action_dict["params"]:
        x, y = action_dict["params"]["position"]
        modified_xpath = xpath.replace("$point", f"'{x},{y}'")
    elif "$point" in xpath:
        # 缺少点击坐标，视为失败
        return 0, visited_nodes

    results = tree.xpath(modified_xpath)
    print("modified_xpath", modified_xpath)
    print(results)

    node_id = hash(xpath)
    if (isinstance(results, bool) and results) or (not isinstance(results, bool) and results):
        visited_nodes.add(node_id)
        return 1, visited_nodes
    return 0, visited_nodes


def evaluate_ratio(task, step_data):
    """
    返回匹配比例（float，范围0~1）
    """
    rule_list = task.split("###")
    all_checked = []  # 收集所有 XPath 的匹配情况（True/False）

    for rule_index, rule_str in enumerate(rule_list, 1):
        rule_parts = rule_str.split("'''")
        xpath_list = rule_parts[1::2]
        checked = [False] * len(xpath_list)

        print(f"\n####### 检查规则 #{rule_index} #############")
        print("目标XPath:", xpath_list)

        history_xml = step_data["history_xml_string"]
        history_actions = step_data["history_action"]

        if len(history_xml) != len(history_actions) or not history_actions:
            print(f"XML记录数: {len(history_xml)}")
            print(f"操作记录数: {len(history_actions)}")
            return 0.0

        for i in range(len(history_xml) - 1, -1, -1):
            xml_string = history_xml[i]
            action_dict = history_actions[i]
            image_path = step_data["history_image_path"][i]

            for xpath_idx, xpath in enumerate(xpath_list):
                if checked[xpath_idx]:
                    continue
                match_flag, visited_nodes = evaluate_action_xml(xml_string, xpath, action_dict)
                if match_flag == 1:
                    checked[xpath_idx] = True

            print(f"检查步骤 #{i+1} ({image_path}): {checked}")

            if all(checked):
                return 1.0  # 虽然不提前 return，但可以跳出循环节省开销

        all_checked.extend(checked)  # 汇总每条规则的匹配情况

    # 最后计算总匹配比例
    if not all_checked:
        return 0.0
    match_ratio = sum(all_checked) / len(all_checked)
    return match_ratio


def evaluate(task, step_data):
    # 解析规则列表 (格式: "规则1###规则2###规则3")
    rule_list = task.split("###")
    # 遍历每条规则进行验证
    for rule_index, rule_str in enumerate(rule_list, 1):
        # 提取规则中的所有XPath表达式 (格式: "文本'''XPath'''文本")
        rule_parts = rule_str.split("'''")
        xpath_list = rule_parts[1::2]  # 奇数索引位置为XPath
        checked = [False] * len(xpath_list)

        print(f"\n####### Check Rule #{rule_index} #############")
        print("Target XPath:", xpath_list)

        # 验证历史交互数据完整性
        history_xml = step_data["history_xml_string"]
        history_actions = step_data["history_action"]

        if len(history_xml) != len(history_actions) or not history_actions:
            print(f"XML Nums: {len(history_xml)}")
            print(f"Action Nums: {len(history_actions)}")
            return False

        # 从最新到最旧遍历历史记录
        for i in range(len(history_xml) - 1, -1, -1):
            xml_string = history_xml[i]
            action_dict = history_actions[i]
            image_path = step_data["history_image_path"][i]

            # 从最新到最旧匹配XPath
            for xpath_idx in range(len(xpath_list) - 1, -1, -1):
                xpath = xpath_list[xpath_idx]
                if checked[xpath_idx]:
                    continue  # 已匹配的XPath跳过检查

                match_flag, visited_nodes = evaluate_action_xml(xml_string, xpath, action_dict)
                if match_flag == 1:
                    checked[xpath_idx] = True
                    break

            print(f"Check Step #{i+1} ({image_path}): {checked}")

            # 若所有XPath均匹配成功，立即返回结果
            if all(checked):
                return True

    # 所有规则遍历完成后，检查是否所有XPath均被匹配
    return all(checked)


def evaluate_by_local(task_rule, path):
    with open(path + "trajectory.json", encoding='utf-8') as f:
        data = json.load(f)
    history_image_path = data['history_image_path']
    print("history_image_path", history_image_path)
    history_action = data['history_action']
    history_xml_string = []
    for image_path in history_image_path:
        xml_path = image_path.replace("png", "xml")
        with open(xml_path, encoding='utf-8') as f:
            xml_string = f.read()
            history_xml_string.append(xml_string)
    step_data = {
    "history_xml_string": history_xml_string,
    "history_action": history_action,
     "history_image_path": history_image_path}
    flag = evaluate(task_rule, step_data)
    print("flag", flag)
    return flag


def evaluate_by_local_ratio(task_rule, path):
    with open(path + "trajectory.json", encoding='utf-8') as f:
        data = json.load(f)
    history_image_path = data['history_image_path']
    print("history_image_path", history_image_path)
    history_action = data['history_action']
    history_xml_string = []
    for image_path in history_image_path:
        xml_path = image_path.replace("png", "xml")
        with open(xml_path, encoding='utf-8') as f:
            xml_string = f.read()
            history_xml_string.append(xml_string)
    step_data = {
    "history_xml_string": history_xml_string,
    "history_action": history_action,
     "history_image_path": history_image_path}
    ratio = evaluate_ratio(task_rule, step_data)
    print("ratio", ratio)
    return ratio


def evaluate_by_local_old(task_rule, path):
    with open(path + "trajectory.json", encoding='utf-8') as f:
        data = json.load(f)
    history_image_path = data['history_image_path']
    print("history_image_path", history_image_path)
    history_action_dict = data['history_action_dict']

    history_xml_string = []
    for image_path in history_image_path:
        xml_path = image_path.replace("png", "xml")
        with open(xml_path, encoding='utf-8') as f:
            xml_string = f.read()
            history_xml_string.append(xml_string)
    step_data = {
    "history_xml_string": history_xml_string,
    "history_action": history_action_dict,
     "history_image_path": history_image_path}
    flag = evaluate(task_rule, step_data)
    print("flag", flag)
    return flag


def re_evaluate_all(model_name, file_name, reset: bool):
    import csv
    import json
    from collections import defaultdict

    def safe_int(x):
        return int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else 0

    def percentage(v, total):
        return f"{(v * 100.0 / total):.2f}%" if total else "0.00%"

    result_list = []
    gold_steps_all, gold_steps_sr = [], []
    real_steps_all, real_steps_sr = [], []
    ratio_all = []
    g_level = defaultdict(int)
    level_all = defaultdict(int)
    level_sr = defaultdict(int)
    level_steps_all = defaultdict(int)
    level_steps_sr = defaultdict(int)

    count_exists = 0
    count_notexiests = 0
    count_totaltasks = 0
    if reset:
        with open(file_name, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if len(row.get('reset_xpath', '').strip()) > 0:
                    eval_path = f"{model_name}/{row['task_identifier']}/"
                    json_file = f"{eval_path}trajectory.json"
                    if os.path.exists(json_file):
                        count_exists += 1
                    else:
                        count_notexiests += 1
                        continue
                    flag = evaluate_by_local(row["reset_xpath"], eval_path)

                    with open(json_file, encoding='utf-8') as f:
                        data = json.load(f)

                    last_action = data['history_action'][-1]["action"]
                    finish_flag = (last_action == "terminate")

                    result_list.append((row['task_identifier'], flag, finish_flag))
    else:
        with open(file_name, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if len(row.get('key_nodes', '').strip()) > 0:
                    eval_path = f"{model_name}/{row['task_identifier']}/"
                    json_file = f"{eval_path}trajectory.json"
                    count_totaltasks += 1
                    if os.path.exists(json_file):
                        count_exists += 1
                    else:
                        count_notexiests += 1
                        continue
                    flag = evaluate_by_local(row["key_nodes"], eval_path)
                    ratio = evaluate_by_local_ratio(row["key_nodes"], eval_path)
                    ratio_all.append(ratio)

                    with open(json_file, encoding='utf-8') as f:
                        data = json.load(f)

                    actions = data.get("history_action", [])
                    action_len = len(actions) + 2
                    level = row.get("level", "").lower()
                    g_level[level] += safe_int(row["golden_steps"])
                    level_all[level] += 1
                    level_steps_all[level] += action_len
                    real_steps_all.append(action_len)
                    gold_steps_all.append(row["golden_steps"])

                    last_action = actions[-1]["action"] if actions else ""
                    finish_flag = (last_action == "terminate")

                    if flag and finish_flag:
                        real_steps_sr.append(action_len)
                        gold_steps_sr.append(row["golden_steps"])
                        level_sr[level] += 1
                        level_steps_sr[level] += safe_int(row["golden_steps"])

                    result_list.append((row['task_identifier'], flag, finish_flag))
    print(
    "Total " +
    str(count_totaltasks) +
    " tasks. \033[32m" +
    str(count_exists) +
    " tasks generated.\033[0m " +
    "\033[31m" +
    str(count_notexiests) +
     " tasks not generated.\033[0m ")

    # 分类计数
    count_SR = count_overdue = count_premature = count_hard_fail = 0
    category_dict = {}

    for task_id, flag, finish_flag in result_list:
        if flag and finish_flag:
            count_SR += 1
            category_dict[task_id] = "SR"
        elif flag:
            count_overdue += 1
            category_dict[task_id] = "Overdue"
        elif finish_flag:
            count_premature += 1
            category_dict[task_id] = "Premature"
        else:
            count_hard_fail += 1
            category_dict[task_id] = "HardFail"

    # 分类输出
    print("\nTask classify result:")
    for task_id, category in category_dict.items():
        print(f"{task_id}: {category}")

    total = len(result_list)
    count_matched = count_SR + count_overdue
    count_unmatched = count_premature + count_hard_fail

    # 步数统计
    r_steps_all = sum(safe_int(x) for x in real_steps_all)
    r_steps_sr = sum(safe_int(x) for x in real_steps_sr)
    g_steps_all = sum(safe_int(x) for x in gold_steps_all)
    g_steps_sr = sum(safe_int(x) for x in gold_steps_sr)

    step_ratio_sr = r_steps_sr / g_steps_sr if g_steps_sr else 0
    step_ratio_all = r_steps_all / g_steps_all if g_steps_all else 0

    print("\nStep Analyze:")
    print(f"Success Task Average Step Ratio (SR): {step_ratio_sr:.2f}")
    print(f"All Task Average Step Ratio : {step_ratio_all:.2f}")

    for level in ['easy', 'medium', 'hard']:
        acc = level_sr[level] / level_all[level] if level_all[level] else 0
        avg_steps = level_steps_all[level] / level_all[level] if level_all[level] else 0
        print(f"{level}-SR: {acc * 100:.2f}%")
        print(f"{level}-AvgSteps: {avg_steps:.2f}")

    # XPath 匹配得分均值
    score_ratio = sum(float(x) for x in ratio_all)
    sr_ratio = score_ratio / len(ratio_all) if ratio_all else 0
    print(f"\nsub-ratio (XPath matched avg): {sr_ratio*100:.2f}%")

    print("\nEvaluation:")
    print(f"SR (matched & finished): {count_SR} ({percentage(count_SR, total)})")
    print(f"Overdue (matched & unfinished): {count_overdue} ({percentage(count_overdue, total)})")
    print(f"xpath_True (matched): {count_matched} ({percentage(count_matched, total)})")
    print(f"HardFail (unmatched & unfinished): {count_hard_fail} ({percentage(count_hard_fail, total)})")
    print(f"Premature (unmatched & finished): {count_premature} ({percentage(count_premature, total)})")
    print(f"xpath_Fail (unmatched): {count_unmatched} ({percentage(count_unmatched, total)})")


def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory-Evalution.")
    parser.add_argument('--result_name', type=str, default='test', help='轨迹文件夹名字')
    parser.add_argument('--file_name', type=str, default='top12.csv', help='数据集')
    parser.add_argument('--reset', action='store_true', help='数据集是否是reset类型')
    return parser.parse_args()


def main():
    args = parse_args()
    result_name = args.result_name
    file_name = args.file_name
    reset = args.reset
    re_evaluate_all(result_name, file_name, reset)


def test_by_task():

    file_name = "top12.csv"

    task_list = []
    with open(file_name, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row)
            if len(row['key_nodes']) > 1:
                rule_list = row["key_nodes"].split("###")
                if row['task_identifier'] == "bili_0":
                    task_list.append(row)
                    for rule_str in rule_list:
                        print(rule_str)
    for task in task_list:
        evaluate_by_local(task["key_nodes"], "result/round1/bili_0/")


# 示例调用
if __name__ == "__main__":
    # main()
    test_by_task()
