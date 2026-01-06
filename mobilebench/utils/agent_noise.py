import time
import copy
from mobilebench.utils import adb_executor
import os
import random
import shutil


class base_agent():

    def __init__(self, env, llm, noise_type="delay"
    ):

        self.llm = llm
        self.env = env

        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        self.history_action = []
        self.summary = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = 1
        self.noise_ratio = 0.2
        self.noise_type = noise_type  # ["repeat"，"unexecuted","delay","popup"]

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        pass

    def clear(self):
        self.history_image_path = []
        self.history_response = []
        self.history_xml_string = []
        self.history_action = []
        self.summary = []

    def save_home_page(self, path="screenshot/"):

        step_prefix = os.path.join(path, "step_0")

        # 保存截图
        img_path = step_prefix + ".png"
        pixels = self.env.screenshot()
        pixels.save(img_path)

        # 保存XML
        xml_path = step_prefix + ".xml"
        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)

    def save_page_by_prefix(self, page_prefix="screenshot/step_0"):
        # 保存截图
        img_path = page_prefix + ".png"
        pixels = self.env.screenshot()
        pixels.save(img_path)

        # 保存XML
        xml_path = page_prefix + ".xml"
        xml_string = self.env.dump_hierarchy()
        with open(xml_path, 'w', encoding="utf-8") as f:
            f.write(xml_string)

    def copy_page1_page2(self, source_path, target_path):
        shutil.copy(source_path, target_path)
        source_path = source_path.replace("png", "xml")
        target_path = target_path.replace("png", "xml")
        shutil.copy(source_path, target_path)

    def copy_noise_page(self, predefined_path="noise/bili/", keyword="delay", page_prefix="screenshot/step_0"):
        files = []
        for item in os.listdir(predefined_path):
            if os.path.isfile(os.path.join(predefined_path, item)) and item.startswith(
                keyword) and item.endswith("png"):
                files.append(item)
        print("################copy_noise_page###############")
        print(files)
        random_string = random.choice(files)
        source_path = predefined_path + random_string
        target_path = page_prefix + ".png"
        self.copy_page1_page2(source_path, target_path)

        return random_string

    def check_files_with_prefix(self, path, prefix):
        """
        检查指定路径下是否有以prefix开头的文件
        返回布尔值
        """
        if not os.path.exists(path):
            return False

        for filename in os.listdir(path):
            if filename.startswith(prefix) and os.path.isfile(os.path.join(path, filename)):
                return filename
        return False

    def check_close_popup(self, filename, action_output):
        print("filename", filename)

        # Convert .xml to .png if needed
        if filename.endswith(".xml"):
            filename = filename.replace(".xml", ".png")

        # temp=input("close_popup")
        box = filename.split(".png")[-2]
        print("box", box)
        boxlist = box.split("_")
        x1 = int(boxlist[-4])
        y1 = int(boxlist[-3])
        x2 = int(boxlist[-2])
        y2 = int(boxlist[-1])
        print("action_output", action_output)
        if "position" not in action_output["params"]:
            return False
        point = action_output["params"]["position"]
        print("point", point)
        x = point[0]
        y = point[1]
        print("x,y", x, y)
        if x >= x1 and x <= x2 and y >= y1 and y <= y2:
            return True
        return False

    def step(self, goal: str, path="screenshot/"):

        step_data = {
          'history_xml_string': self.history_xml_string,
          "history_image_path": self.history_image_path,
          "history_response": self.history_response,
          "history_action": self.history_action,
          "summary": self.summary,
        }

        step_index = "step_" + str(len(self.history_image_path) + 1)
        step_prefix = os.path.join(path, step_index)

        # 保存截图
        img_path = step_prefix + ".png"
        # 保存XML
        xml_path = step_prefix + ".xml"
        if not os.path.exists(img_path):
            pixels = self.env.screenshot()
            pixels.save(img_path)

            xml_string = self.env.dump_hierarchy()
            with open(xml_path, 'w', encoding="utf-8") as f:
                f.write(xml_string)
        else:
            with open(xml_path, encoding='utf-8') as f:
                xml_string = f.read()
        # pixels_array=np.asarray(pixels)

        history = copy.deepcopy(step_data)

        response, action_output = self.llm.predict_mm(
            goal, img_path, history
        )

        self.history_xml_string.append(xml_string)
        self.history_image_path.append(img_path)
        self.history_response.append(response)
        self.history_action.append(action_output)

        if action_output['action'] == 'terminate':
            self.summary.append('Agent thinks the request has been completed.')
            step_data = {
              'history_xml_string': self.history_xml_string,
              "history_image_path": self.history_image_path,
              "history_response": self.history_response,
              "history_action": self.history_action,
              "summary": self.summary,
            }
            return (True, step_data)

        print("##########model_response#################\n")
        print(response + "\n")
        print("##########execute_action#################\n")

        # 这里也先取消try except
        # try:
        print(action_output["action"])
        print(action_output["params"])
        print(action_output["normalized_params"])

        random_score = random.random()
        if random_score < self.noise_ratio:
            print("here")
            print(self.noise_type)
            # temp=input("pop up here0")
            if self.noise_type == "repeat":
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(5.0)
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(2.0)
                step_index = "step_" + str(len(self.history_image_path) + 1)
                step_prefix = os.path.join(path, step_index)
                self.save_page_by_prefix(page_prefix=step_prefix)
                self.save_page_by_prefix(page_prefix=step_prefix + "_repeat")
            elif self.noise_type == "unexecuted":
                step_index = "step_" + str(len(self.history_image_path) + 1)
                step_prefix = os.path.join(path, step_index)
                self.save_page_by_prefix(page_prefix=step_prefix)
                self.save_page_by_prefix(page_prefix=step_prefix + "_unexecuted")
            elif self.noise_type == "delay":
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(5.0)
                step_index = "step_" + str(len(self.history_image_path) + 1)
                step_prefix = os.path.join(path, step_index)
                app_name = step_prefix.split(os.path.sep)[-2].split("_")[0]
                delay_file = self.copy_noise_page(predefined_path=os.path.join(
                    "noise", app_name, ""), keyword="delay", page_prefix=step_prefix)
                self.save_page_by_prefix(page_prefix=step_prefix + "_delay")
            elif self.noise_type == "popup":
                step_index_last = "step_" + str(len(self.history_image_path))
                if self.check_files_with_prefix(path, step_index + "_" + "popup") == False:
                    adb_executor.execute_adb_action(action_output, self.env)
                    time.sleep(5.0)
                    step_index = "step_" + str(len(self.history_image_path) + 1)
                    step_prefix = os.path.join(path, step_index)
                    app_name = step_prefix.split(os.path.sep)[-2].split("_")[0]
                    popup_file = self.copy_noise_page(predefined_path=os.path.join(
                        "noise", app_name, ""), keyword="popup", page_prefix=step_prefix)
                    self.save_page_by_prefix(page_prefix=step_prefix + "_" + popup_file.split(".")[0])
                    # temp=input("pop up here")
                else:
                    filename = self.check_files_with_prefix(path, step_index + "_" + "popup")
                    if filename.endswith(".xml"):
                        filename = filename.replace(".xml", ".png")
                    flag = self.check_close_popup(filename, action_output)
                    # temp=input("pop up here3")
                    if flag == True:
                        source_path = os.path.join(path, filename)
                        step_index1 = "step_" + str(len(self.history_image_path) + 1)
                        target_path = os.path.join(path, step_index1 + ".png")
                        self.copy_page1_page2(source_path, target_path)
                        # target_path=source_path.replace(step_index,step_index1)
                        # self.copy_page1_page2(source_path, target_path)
                    else:
                        source_path = os.path.join(path, step_index + ".png")
                        step_index1 = "step_" + str(len(self.history_image_path) + 1)
                        target_path = os.path.join(path, step_index1 + ".png")
                        self.copy_page1_page2(source_path, target_path)
                        source_path = os.path.join(path, filename)
                        target_path = source_path.replace(step_index, step_index1)
                        self.copy_page1_page2(source_path, target_path)

                    # temp=input("pop up here4")
            else:
                assert 1 == 2
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(2.0)
        else:
            if self.noise_type == "popup":
                step_index = "step_" + str(len(self.history_image_path))
                # step_prefix = os.path.join(path, step_index)
                if self.check_files_with_prefix(path, step_index + "_" + "popup") != False:
                    filename = self.check_files_with_prefix(path, step_index + "_" + "popup")
                    if filename.endswith(".xml"):
                        filename = filename.replace(".xml", ".png")
                    flag = self.check_close_popup(filename, action_output)
                    print("flag", flag)
                    # temp=input("pop up here3")
                    if flag == True:
                        source_path = os.path.join(path, filename)
                        step_index1 = "step_" + str(len(self.history_image_path) + 1)
                        target_path = os.path.join(path, step_index1 + ".png")
                        self.copy_page1_page2(source_path, target_path)
                        # target_path=source_path.replace(step_index,step_index1)
                        # self.copy_page1_page2(source_path, target_path)
                    else:
                        source_path = os.path.join(path, step_index + ".png")
                        step_index1 = "step_" + str(len(self.history_image_path) + 1)
                        target_path = os.path.join(path, step_index1 + ".png")
                        self.copy_page1_page2(source_path, target_path)
                        source_path = os.path.join(path, filename)
                        target_path = source_path.replace(step_index, step_index1)
                        self.copy_page1_page2(source_path, target_path)

                    # temp=input("pop up here2")
                else:
                    adb_executor.execute_adb_action(action_output, self.env)
                    time.sleep(2.0)
            else:
                adb_executor.execute_adb_action(action_output, self.env)
                time.sleep(2.0)
        # except Exception as e:
        #     print('Failed to execute action.')
        #     print(str(e))
        #     print("action_output", action_output)
        #     summary_error = 'Can not execute the action, make sure to select the action with the required '
        #     summary_error += 'parameters (if any) in the correct JSON format!'
        #     self.summary.append(summary_error)
        #     step_data = {
        #       'history_xml_string': self.history_xml_string,
        #       "history_image_path": self.history_image_path,
        #       "history_response": self.history_response,
        #       "history_action": self.history_action,
        #       "summary": self.summary,
        #     }
        #     return (False, step_data)

        time.sleep(self.wait_after_action_seconds)

        step_data = {
          'history_xml_string': self.history_xml_string,
          "history_image_path": self.history_image_path,
          "history_response": self.history_response,
          "history_action": self.history_action,
          "summary": self.summary,
        }
        return (False, step_data)
