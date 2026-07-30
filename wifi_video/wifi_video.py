import os

# 在打开视频流之前清除代理环境变量
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import sys
import time  # 引入 time 模块用于计算真实帧率
from datetime import datetime

import cv2


def get_stream(video_url, scale_factor=0.8, output_dir=".", target_fps=10.0):
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"错误：无法打开视频流 {video_url}")
        sys.exit(1)

    # 1. 动态生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_dir}/recorded_stream_{timestamp}.mp4"

    print(f"成功连接，图像将等比放大 {scale_factor} 倍。按 'q' 键退出...")
    print(f"本次录像将保存至: {output_filename}")
    print(
        f"当前设定的写入帧率: {target_fps} FPS (若播放速度异常，请根据下方打印的真实帧率调整此值)"
    )

    ret, first_frame = cap.read()
    if not ret:
        print("错误：无法读取初始帧。")
        sys.exit(1)

    h, w = first_frame.shape[:2]
    target_width = int(w * scale_factor)
    target_height = int(h * scale_factor)
    frame_size = (target_width, target_height)

    # 注意：cap.get(cv2.CAP_PROP_FPS) 对网络流(MJPEG)通常返回 0 或不准确的值
    # 因此我们直接使用传入的 target_fps，或者你可以手动指定一个保守值（如 10.0）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, target_fps, frame_size)

    if not out.isOpened():
        print("错误：无法创建视频写入器。")
        sys.exit(1)

    # 写入第一帧
    out.write(cv2.resize(first_frame, frame_size, interpolation=cv2.INTER_CUBIC))

    # 用于计算真实帧率的变量
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取帧，流可能已中断。")
            break

        resized_frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_CUBIC)
        out.write(resized_frame)
        cv2.imshow("Video Stream (Scaled)", resized_frame)

        # --- 真实帧率监测逻辑 ---
        frame_count += 1
        current_time = time.time()
        # 每 2 秒计算并打印一次实际捕获帧率
        if current_time - start_time >= 2.0:
            actual_fps = frame_count / (current_time - start_time)
            print(f"-> 当前实际捕获帧率: {actual_fps:.2f} FPS")
            frame_count = 0
            start_time = current_time
        # ------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("正在保存文件并释放资源...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已成功保存为: {output_filename}")


if __name__ == "__main__":
    URL = "http://192.168.4.1:81/stream"

    # 建议：对于常见的图传模块(如ESP32-CAM)，真实帧率通常在 5 ~ 15 之间。
    # 你可以先运行一次，观察终端打印的“实际捕获帧率”，然后将下面的 10.0 修改为那个真实值。
    get_stream(URL, scale_factor=1.5, output_dir=".", target_fps=13.0)
