#!/usr/bin/env python3
"""多点标定工具 — 在横杆上放置钢球，逐点记录像素→厘米映射"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
import yaml

from camera import Camera
from processor import BallDetector, ROIManager
from tuner import Tuner
from calib import Calibration

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

CM_POSITIONS = [-12.5, -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5]

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cam_cfg = cfg["camera"]
cam = Camera(
    device_id=cam_cfg["device_id"],
    width=cam_cfg["width"],
    height=cam_cfg["height"],
    fps=cam_cfg["fps"],
)
det_cfg = cfg.get("detector", {})
detector = BallDetector(
    clahe_clip=det_cfg.get("clahe_clip", 2.0),
    adaptive_bg_alpha=det_cfg.get("adaptive_bg_alpha", 0.98),
    hough_cfg=det_cfg.get("hough"),
)
td = cfg.get("tuner_defaults", {})
bar_cm = cfg["system"]["bar_length_cm"]

recorded_px = []
current_target_idx = 0

if not cam.open():
    print("无法打开摄像头")
    sys.exit(1)

Tuner.init(defaults=td)
print("=" * 55)
print("多点标定工具")
print("=" * 55)
print("步骤:")
print("  1. 按 's' 截取静态背景（清空小球）")
print("  2. 把小球放到指定 cm 位置")
print("  3. 按 'r' 记录当前检测到的像素位置")
print("  4. 所有点完成后按 'w' 保存")
print("  5. 按 'q' 退出（不保存）")
print("-" * 55)

while True:
    ret, frame = cam.get_frame()
    if not ret:
        break

    p = Tuner.get_params()
    roi_y_min, roi_y_max = p["roi_y_min"], p["roi_y_max"]

    ROIManager.draw_roi_overlay(frame, roi_y_min, roi_y_max)
    roi_gray = ROIManager.extract_roi(frame, roi_y_min, roi_y_max)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        detector.capture_background(roi_gray)
        print(f">> 背景截取成功！")

    if key == ord("r") and detector.has_background():
        if current_target_idx >= len(CM_POSITIONS):
            print(">> 所有位置已记录完毕，按 'w' 保存")
        else:
            pos_cm, debug = detector.detect(
                roi_gray, p["diff_threshold"], p["morph_kernel_size"],
                p["projection_snr"],
                Calibration.from_linear(
                    p["pixel_left"], p["pixel_right"], bar_cm,
                ),
            )
            if pos_cm is not None:
                raw_px = debug.get("ball_x_pixel", 0)
                target_cm = CM_POSITIONS[current_target_idx]
                recorded_px.append(raw_px)
                print(f"  记录 #{current_target_idx+1}: px={raw_px} → cm={target_cm}  (实测cm={pos_cm:.2f})")
                current_target_idx += 1
            else:
                print("  未检测到小球，请检查小球位置和光照")

    if key == ord("w") and len(recorded_px) >= 3:
        out_path = os.path.join(PROJECT_ROOT, "config", "px2cm.npz")
        calib = Calibration(
            np.array(recorded_px, dtype=np.float32),
            np.array(CM_POSITIONS[:len(recorded_px)], dtype=np.float32),
        )
        calib.save(out_path)
        print(f"\n>> 标定文件已保存: {out_path}")
        print(f"   {len(recorded_px)} 个点已记录")
        print(f"   请将文件路径填入 config.yaml calibration.file")
        break

    if key == 27 or key == ord("q"):
        print("退出（未保存）")
        break

    # 显示信息
    if current_target_idx < len(CM_POSITIONS):
        target_cm = CM_POSITIONS[current_target_idx]
        cv2.putText(frame, f"STEP {current_target_idx+1}/{len(CM_POSITIONS)}: put ball at {target_cm:+.1f} cm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    if recorded_px:
        for i, (px_val, cm_val) in enumerate(zip(recorded_px, CM_POSITIONS[:len(recorded_px)])):
            y = 460 - i * 18
            cv2.circle(frame, (int(px_val), int((roi_y_min + roi_y_max) / 2)),
                       5, (0, 0, 255), -1)
            cv2.putText(frame, f"{cm_val:+.1f}cm", (int(px_val) + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("Calibration", frame)

cam.release()
cv2.destroyAllWindows()
