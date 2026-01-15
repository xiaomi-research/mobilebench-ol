import os
import time
import subprocess
from PIL import Image


def get_screenshot(adb_path, device_id):
    command = adb_path + f" -s {device_id} shell rm /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = adb_path + f" -s {device_id} shell screencap -p /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = adb_path + f" -s {device_id} pull /sdcard/screenshot.png ./screenshot"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    image_path = "./screenshot/screenshot.png"
    save_path = "./screenshot/screenshot.jpg"
    image = Image.open(image_path)
    image.convert("RGB").save(save_path, "JPEG")
    os.remove(image_path)


def get_screenshot_with_path(adb_path, device_id, save_path):
    command = adb_path + f" -s {device_id} shell rm /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    command = adb_path + f" -s {device_id} shell screencap -p /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)
    temp_path = save_path.replace('.jpg', '.png')
    command = adb_path + f" -s {device_id} pull /sdcard/screenshot.png {temp_path}"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    image = Image.open(temp_path)
    image.convert("RGB").save(save_path, "JPEG")
    os.remove(temp_path)


def get_screenshot_u2(device, save_path="./screenshot/screenshot.jpg"):
    device.screenshot(save_path)


def get_xml_u2(device, save_path="./screenshot/hierarchy.xml"):
    xml = device.dump_hierarchy()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(xml)


def get_xml(adb_path, device_id, save_path="./screenshot/hierarchy.xml"):
    command = adb_path + f" -s {device_id} shell uiautomator dump /sdcard/hierarchy.xml"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.3)
    command = adb_path + f" -s {device_id} pull /sdcard/hierarchy.xml {save_path}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def tap(adb_path, device_id, x, y):
    command = adb_path + f" -s {device_id} shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type(adb_path, device_id, text):
    '''
    输入文本前先全选并删除已有内容
    '''
    # 全选并删除现有内容
    # KEYCODE_MOVE_END: 移动到文本末尾
    command = adb_path + f" -s {device_id} shell input keyevent KEYCODE_MOVE_END"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.1)

    # KEYCODE_SHIFT_LEFT + KEYCODE_MOVE_HOME: 按住Shift并移动到开头，实现全选
    command = adb_path + f" -s {device_id} shell input keyevent --longpress KEYCODE_SHIFT_LEFT"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    command = adb_path + f" -s {device_id} shell input keyevent KEYCODE_MOVE_HOME"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.1)

    # KEYCODE_DEL: 删除选中内容
    command = adb_path + f" -s {device_id} shell input keyevent KEYCODE_DEL"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.1)

    # 输入新文本
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == ' ':
            command = adb_path + f" -s {device_id} shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == '_':
            command = adb_path + f" -s {device_id} shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
            command = adb_path + f" -s {device_id} shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in '-.,!?@\'°/:;()':
            command = adb_path + f" -s {device_id} shell input text \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = adb_path + f" -s {device_id} shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)


def slide(adb_path, device_id, x1, y1, x2, y2):
    command = adb_path + f" -s {device_id} shell input swipe {x1} {y1} {x2} {y2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def back(adb_path, device_id):
    command = adb_path + f" -s {device_id} shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def home(adb_path, device_id):
    command = adb_path + f" -s {device_id}   shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    subprocess.run(command, capture_output=True, text=True, shell=True)
