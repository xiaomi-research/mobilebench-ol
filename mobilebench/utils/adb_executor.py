import re
import time
import logging


def get_main_activity(package_name, d):
    """解析 dumpsys 获取主 Activity"""
    result = d.shell(f"dumpsys package {package_name}")
    lines = result.output.splitlines()
    activity = None
    for i, line in enumerate(lines):
        if "android.intent.action.MAIN" in line:
            for j in range(i, max(i - 10, 0), -1):
                match = re.search(rf"{re.escape(package_name)}/([a-zA-Z0-9_.]+)", lines[j])
                if match:
                    activity = match.group(1)
                    break
            if activity:
                break
    if not activity:
        print("⚠️ 主 Activity 未找到，将尝试使用 monkey 启动")
        return None
    return activity

# def launch_app(package_name, d):
#     """自动查找并启动 App"""
#     print(f"🚀 正在尝试启动 {package_name} ...")
#     result = d.shell(f"am start -n {package_name}")
#     if "Error" in result.output or result.exit_code != 0:
#         print(f"❌ 启动失败：{result.stdout.strip()}")
#     else:
#         print("✅ 启动成功")
#     '''
#     activity = get_main_activity(package_name, d)
#     if activity:
#         #print("activity",activity)
#         result = d.shell(f"am start -n {package_name}/{activity}")
#         if "Error" in result.output or result.exit_code != 0:
#             print(f"❌ 启动失败：{result.stdout.strip()}")
#         else:
#             print("✅ 启动成功")
#     else:
#         monkey_result = d.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
#         if "Error" in monkey_result.output or monkey_result.exit_code != 0:
#             print(f"❌ Monkey 启动失败：{monkey_result.output.strip()}")
#         else:
#             print("✅ Monkey 启动成功")
#     '''


def launch_app(package_name, d):
    """自动查找并启动 App"""
    print(f"🚀 Starting APP {package_name} ...")
    result = d.shell(f"am start -n {package_name}")
    if "Error" in result.output or result.exit_code != 0:
        print(f"❌ APP Initializing Failed: {result.stdout.strip()}")
    else:
        print("✅ APP Initializing Succeed")


def execute_adb_action(action, env) -> None:
    try:
        if action["action"] in ['click', 'double_tap', 'long_press']:
            x = action["params"]['position'][0]
            y = action["params"]['position'][1]
            if action["action"] == 'click':
                env.click(x, y)
                # adb_utils.tap_screen(x, y, env)
            elif action["action"] == 'double_tap':
                env.double_click(x, y)
            # adb_utils.double_tap(x, y, env)
            elif action["action"] == 'long_press':
                env.long_click(x, y)
            else:
                raise ValueError(f'Invalid click action: {action}')
        elif action["action"] == 'type':
            text = action["params"]['text']
            if text:
                env.set_input_ime(True)
                env.send_keys(text, clear=True)
                env.set_input_ime(False)
                env.press('enter')
            else:
                logging.warning(
                'Input_text action indicated, but no text provided. No '
                    'action will be executed.'
                )
        elif action["action"] in {'swipe', 'scroll', 'drag'}:
            params = action.get("params", {})
            direction = params.get("direction")
            screen_height = 2400
            screen_width = 1080
            mid_x, mid_y = 0.3 * screen_width, 0.3 * screen_height

            if direction:
                start_x = params.get('position', [screen_width // 2, screen_height // 2])[0]
                start_y = params.get('position', [screen_width // 2, screen_height // 2])[1]

                if direction == 'down':
                    end_x, end_y = start_x, min(start_y - mid_y, screen_height)
                elif direction == 'up':
                    end_x, end_y = start_x, max(start_y + mid_y, 0)
                elif direction == 'left':
                    end_x, end_y = min(start_x + mid_x, screen_width), start_y
                elif direction == 'right':
                    end_x, end_y = max(start_x - mid_x, 0), start_y
                else:
                    raise ValueError(f"Unknown direction: {direction}")
            else:
                start_x = params.get('start_position', [0, 0])[0]
                start_y = params.get('start_position', [0, 0])[1]
                end_x = params.get('end_position', [0, 0])[0]
                end_y = params.get('end_position', [0, 0])[1]

            env.swipe(int(start_x), int(start_y), int(end_x), int(end_y), 500 / 1000)

        elif action["action"] == 'enter':
            env.press('enter')
        elif action["action"] == 'home':
            env.press('home')
        elif action["action"] == 'back':
            env.press('back')
        elif action["action"] == 'open':
            time.sleep(1.0)
        #     app_name = action["params"]['app_name']
        #     print("app_name",app_name)
        #     try:
        #         activity = package_utils.get_adb_activity(app_name.lower())
        #         print("#########activity######")
        #         print(activity)
        #         launch_app(activity, d)
        #     except Exception as e:
        #         print(e)
        #     try:
        #         e=env(text=app_name).click()
        #     except:
        #         raise ValueError('No app name provided')
        elif action["action"] == 'wait_time' or action["action"] == 'wait':
            time.sleep(5.0)
        else:
            print('Invalid action type')
    except Exception as e:
        print('Failed in execute_adb_action')
        print("action_type", action["action"])
        print(str(e))
