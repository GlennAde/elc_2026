import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

camera_cfg = cfg.get("camera", {})
tuner_cfg = cfg.get("tuner_defaults", {})
detector_cfg = cfg.get("detector", {})
system_cfg = cfg.get("system", {})

ROI_Y_MIN = tuner_cfg.get("roi_y_min", 180)
ROI_Y_MAX = tuner_cfg.get("roi_y_max", 300)

PIXEL_LEFT = tuner_cfg.get("pixel_left", 70)
PIXEL_RIGHT = tuner_cfg.get("pixel_right", 570)
PIXEL_CENTER = tuner_cfg.get("center_pixel", (PIXEL_LEFT + PIXEL_RIGHT) / 2.0)
BAR_LENGTH_CM = system_cfg.get("bar_length_cm", 25.0)
SCALE_K = BAR_LENGTH_CM / (PIXEL_RIGHT - PIXEL_LEFT)

DIFF_THRESHOLD = tuner_cfg.get("diff_threshold", 30)
MORPH_KERNEL_SIZE = tuner_cfg.get("morph_kernel_size", 7)
PROJECTION_SNR = tuner_cfg.get("projection_snr", 5.0)

CAPTURE_FPS = camera_cfg.get("fps", 120)
DT = cfg.get("kalman", {}).get("dt", 1.0 / 120.0)

CLAHE_CLIP = detector_cfg.get("clahe_clip", 2.0)

cap = cv2.VideoCapture(camera_cfg.get("device_id", 0))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

if CLAHE_CLIP > 0:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
else:
    clahe = None

bg_gray = None

last_pos = None
last_time = None
velocity = 0.0

print("=" * 50)
print("桌面钢珠检测 Demo 启动！")
print("1. 请先清空画面中的钢珠/小球。")
print("2. 按键盘上的 's' 键捕获当前桌面作为【静态背景】。")
print("3. 把钢珠放入 ROI 框内，观察识别与一维投影效果。")
print("4. 按 'ESC' 或 'q' 键退出。")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面！")
        break

    cv2.rectangle(frame, (0, ROI_Y_MIN), (frame.shape[1], ROI_Y_MAX),
                  (0, 255, 255), 2)
    cv2.putText(frame, "ROI Area", (10, ROI_Y_MIN - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    roi = frame[ROI_Y_MIN:ROI_Y_MAX, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if clahe is not None:
        gray = clahe.apply(gray)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        bg_gray = gray.copy()
        last_pos = None
        print(">> 【背景截取成功】现在可以放入钢珠测试了！ <<")

    ball_pos_cm = None

    if bg_gray is not None:
        diff = cv2.absdiff(gray, bg_gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        y_projection = np.sum(thresh_clean, axis=0, dtype=np.float32)
        max_val = np.max(y_projection)

        proj_canvas = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)

        if max_val > 500:
            total_mass = np.sum(y_projection)
            if total_mass > 0:
                indices = np.arange(len(y_projection), dtype=np.float32)
                ball_x_pixel = np.average(indices, weights=y_projection)
            else:
                ball_x_pixel = float(np.argmax(y_projection))

            ball_pos_cm = (ball_x_pixel - PIXEL_CENTER) * SCALE_K
            ball_pos_cm = max(-12.5, min(12.5, ball_pos_cm))

            roi_center_y = int((ROI_Y_MIN + ROI_Y_MAX) / 2)
            cv2.circle(frame, (int(ball_x_pixel), roi_center_y), 12,
                       (0, 255, 0), -1)
            cv2.putText(frame, f"Ball Pos: {ball_pos_cm:.2f} cm",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

            cv2.line(proj_canvas, (int(ball_x_pixel), 0),
                     (int(ball_x_pixel), 150), (0, 0, 255), 2)

        # 有限差分简易速度估计
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if ball_pos_cm is not None:
            if last_pos is not None and last_time is not None:
                dt_real = max(now - last_time, 0.001)
                velocity = (ball_pos_cm - last_pos) / dt_real
            last_pos = ball_pos_cm
            last_time = now
        else:
            last_pos = None
            velocity = 0.0

        direction = "->" if velocity > 0 else "<-" if velocity < 0 else "--"
        cv2.putText(frame, f"Velocity: {velocity:.2f} cm/s {direction}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        norm_proj = (y_projection / (np.max(y_projection) + 1e-5) * 120).astype(np.int32)
        for x in range(1, len(norm_proj)):
            cv2.line(proj_canvas,
                     (x - 1, 140 - norm_proj[x - 1]),
                     (x, 140 - norm_proj[x]),
                     (255, 255, 255), 1)

        cv2.imshow("1. Diff Image", diff)
        cv2.imshow("2. Thresh + Morphology", thresh_clean)
        cv2.imshow("3. 1D Projection Curve", proj_canvas)

    cv2.imshow("Main View (Demo)", frame)

    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
