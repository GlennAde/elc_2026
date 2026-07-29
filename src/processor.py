import cv2
import numpy as np

class ROIManager:
    """ROI 区域提取与绘制工具类"""
    @staticmethod
    def extract_roi(frame, roi_y_min, roi_y_max):
        h, _ = frame.shape[:2]
        y1 = max(0, roi_y_min)
        y2 = min(h, max(y1 + 1, roi_y_max))
        roi = frame[y1:y2, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    @staticmethod
    def draw_roi_overlay(frame, roi_y_min, roi_y_max):
        h, w = frame.shape[:2]
        y1 = max(0, roi_y_min)
        y2 = min(h, max(y1 + 1, roi_y_max))
        cv2.rectangle(frame, (0, y1), (w, y2), (0, 255, 255), 2)
        cv2.putText(frame, "ROI Area", (10, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


class Visualizer:
    """视觉绘制与调试窗口工具类"""
    @staticmethod
    def draw_ball_overlay(frame, ball_pos_cm, ball_x_pixel, roi_y_min, roi_y_max):
        roi_center_y = int((roi_y_min + roi_y_max) / 2)
        cv2.circle(frame, (ball_x_pixel, roi_center_y), 12, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Ball Pos: {ball_pos_cm:.2f} cm",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    @staticmethod
    def build_projection_canvas(y_projection, width, ball_x_pixel=None):
        canvas = np.zeros((150, width, 3), dtype=np.uint8)
        if ball_x_pixel is not None:
            cv2.line(canvas, (ball_x_pixel, 0), (ball_x_pixel, 150), (0, 0, 255), 2)

        max_p = np.max(y_projection) + 1e-5
        norm_proj = (y_projection / max_p * 120).astype(np.int32)
        for x in range(1, len(norm_proj)):
            cv2.line(
                canvas,
                (x - 1, 140 - norm_proj[x - 1]),
                (x, 140 - norm_proj[x]),
                (255, 255, 255),
                1,
            )
        return canvas

    @staticmethod
    def show_debug_windows(diff, thresh, proj_canvas):
        cv2.imshow("1. Diff Image", diff)
        cv2.imshow("2. Thresh + Morphology", thresh)
        cv2.imshow("3. 1D Projection Curve", proj_canvas)


class BallDetector:
    """钢球检测核心处理类"""
    def __init__(self):
        self.bg_gray = None

    def capture_background(self, roi_gray):
        self.bg_gray = roi_gray.copy()

    def has_background(self):
        return self.bg_gray is not None

    def detect(self, roi_gray, pixel_center, scale_k, diff_threshold, morph_kernel_size, projection_min_val):
        if self.bg_gray is None:
            return None, {}

        diff = cv2.absdiff(roi_gray, self.bg_gray)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        ks = max(1, morph_kernel_size)
        if ks % 2 == 0:
            ks += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        y_projection = np.sum(thresh_clean, axis=0)
        max_val = np.max(y_projection)

        debug = {
            "diff": diff,
            "thresh": thresh_clean,
            "projection": y_projection,
            "max_val": max_val,
        }

        ball_pos_cm = None
        if max_val > projection_min_val:
            ball_x_pixel = int(np.argmax(y_projection))
            ball_pos_cm = (ball_x_pixel - pixel_center) * scale_k
            ball_pos_cm = max(-12.5, min(12.5, ball_pos_cm))
            debug["ball_x_pixel"] = ball_x_pixel

        return ball_pos_cm, debug
