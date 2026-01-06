import os
import time
import subprocess

# === 配置 ===
APK_FOLDER = r"C:/Users/leonic/Downloads/longtail_apk"  # 替换成你的apk文件夹路径
WAIT_SECONDS = 5
MAX_APK_COUNT = 69
DEVICE_SERIAL = "orp7u4jrkjnrsw75"  # <<< 替换为你自己的序列号


def get_apk_files(folder):
    apk_files = [f for f in os.listdir(folder) if f.endswith('.apk')]
    apk_files.sort()
    return apk_files[:MAX_APK_COUNT]


def install_apk(apk_path):
    print(f"Installing: {apk_path}")
    result = subprocess.run(
        ["adb", "-s", DEVICE_SERIAL, "install", apk_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅ 安装成功")
    else:
        print("❌ 安装失败")
        print(result.stderr)


def main():
    apk_list = get_apk_files(APK_FOLDER)
    if not apk_list:
        print("没有找到任何APK文件。")
        return

    for idx, apk_name in enumerate(apk_list, 1):
        full_path = os.path.join(APK_FOLDER, apk_name)
        install_apk(full_path)
        print(f"[{idx}/{len(apk_list)}] 等待你在手机上确认（{WAIT_SECONDS}秒）...\n")
        time.sleep(WAIT_SECONDS)

    print("🎉 所有 APK 已处理完毕。")


if __name__ == "__main__":
    main()
