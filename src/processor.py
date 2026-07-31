import cv2
import numpy as np

from hough import HoughRefiner


class ROIManager:
    """ROI 区域提取与绘制工具类"""

    @staticmethod
    def extract_roi(frame, roi_y_min, roi_y_max, roi_x_min=None, roi_x_max=None):
        h, w = frame.shape[:2]
        y1 = max(0, roi_y_min)
        y2 = min(h, max(y1 + 1, roi_y_max))
        if roi_x_min is None:
            x1 = 0
        else:
            x1 = max(0, min(w - 1, roi_x_min))
        if roi_x_max is None:
            x2 = w
        else:
            x2 = max(x1 + 1, min(w, roi_x_max))
        if y1 >= y2 or x1 >= x2:
            return None, x1
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, x1

    @staticmethod
    def draw_roi_overlay(frame, roi_y_min, roi_y_max, center_pixel=None,
                         roi_x_min=None, roi_x_max=None):
        h, w = frame.shape[:2]
        y1 = max(0, roi_y_min)
        y2 = min(h, max(y1 + 1, roi_y_max))

        def _dash(x, color):
            for step in range(y1, y2, 8):
                cv2.line(frame, (x, step), (x, min(step + 4, y2)), color, 1)

        x1_vis = roi_x_min if roi_x_min is not None else 0
        x2_vis = roi_x_max if roi_x_max is not None else w
        cv2.rectangle(frame, (x1_vis, y1), (x2_vis, y2), (0, 255, 255), 2)
        cv2.putText(
            frame, "ROI Area", (x1_vis + 5, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
        )
        if roi_x_min is not None:
            _dash(roi_x_min, (0, 220, 0))
        if roi_x_max is not None:
            _dash(roi_x_max, (0, 220, 0))
        if center_pixel is not None:
            cx = max(0, min(w, int(center_pixel)))
            _dash(cx, (255, 0, 0))


class Visualizer:
    """视觉绘制与调试窗口工具类"""

    @staticmethod
    def draw_ball_overlay(frame, ball_x_pixel, roi_y_min, roi_y_max):
        roi_center_y = int((roi_y_min + roi_y_max) / 2)
        cv2.circle(frame, (int(ball_x_pixel), roi_center_y), 12, (0, 255, 0), -1)

    @staticmethod
    def draw_pose_overlay(frame, position_cm, velocity_cm_s):
        cv2.putText(
            frame,
            f"Pos: {position_cm:.2f} cm",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        direction = "->" if velocity_cm_s > 0 else "<-" if velocity_cm_s < 0 else "--"
        cv2.putText(
            frame,
            f"Vel: {velocity_cm_s:.2f} cm/s {direction}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    @staticmethod
    def draw_serial_status(frame, enabled):
        color = (0, 255, 0) if enabled else (0, 0, 255)
        status = "SERIAL: ON" if enabled else "SERIAL: OFF"
        cv2.putText(
            frame, status, (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
        )

    @staticmethod
    def build_projection_canvas(y_projection, width, ball_x_pixel=None):
        canvas = np.zeros((150, width, 3), dtype=np.uint8)
        if ball_x_pixel is not None:
            cv2.line(canvas, (int(ball_x_pixel), 0),
                     (int(ball_x_pixel), 150), (0, 0, 255), 2)

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
    """钢球检测 — 反相二值化 + 最高峰连通域加权质心"""

    AREA_MIN = 500
    AREA_MAX = 15000
    PEAK_WIDTH_MIN = 3
    PEAK_WIDTH_MAX = 120
    POS_JUMP_MAX_CM = 2.0
    CONFIRM_FRAMES = 2

    def __init__(self, clahe_clip=2.0, hough_cfg=None):
        if clahe_clip > 0:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        else:
            self.clahe = None

        self.hough = None
        if hough_cfg and hough_cfg.get("enabled", False):
            self.hough = HoughRefiner(
                dp=hough_cfg.get("dp", 1),
                min_dist=hough_cfg.get("min_dist", 50),
                param1=hough_cfg.get("param1", 80),
                param2=hough_cfg.get("param2", 12),
                min_radius=hough_cfg.get("min_radius", 12),
                max_radius=hough_cfg.get("max_radius", 32),
                max_deviation=hough_cfg.get("max_deviation", 25),
            )

        self._last_raw_pos = None
        self._confirm_count = 0
        self._is_tracking = False

    def _preprocess(self, roi_gray):
        if self.clahe is not None:
            return self.clahe.apply(roi_gray)
        return roi_gray

    def detect_simple(self, roi_gray, simple_threshold, morph_kernel_size,
                      projection_snr, center_pixel, scale_k, half_bar,
                      area_min=None, area_max=None, roi_x_min=0):
        roi_gray = self._preprocess(roi_gray)

        _, thresh = cv2.threshold(roi_gray, simple_threshold, 255, cv2.THRESH_BINARY_INV)

        ks_m = max(1, morph_kernel_size)
        if ks_m % 2 == 0:
            ks_m += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_m, ks_m))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        y_projection = np.sum(thresh_clean, axis=0, dtype=np.float32)
        max_val = np.max(y_projection)

        debug = {
            "diff": thresh,
            "thresh": thresh_clean,
            "projection": y_projection,
            "max_val": max_val,
            "bg_status": "simple",
        }

        ball_pos_cm = None
        raw_detected = False

        mean_proj = float(np.mean(y_projection))
        baseline = max(mean_proj, 1.0)
        snr = max_val / baseline

        eff_snr = projection_snr
        if self._is_tracking:
            eff_snr = projection_snr * 0.7

        if max_val > 10 and snr > eff_snr:
            half_max = max_val / 2.0
            above = y_projection > half_max

            peak_center = int(np.argmax(y_projection))
            left = peak_center
            while left > 0 and above[left - 1]:
                left -= 1
            right = peak_center
            while right < len(above) - 1 and above[right + 1]:
                right += 1

            peak_region = y_projection[left:right + 1]
            total_mass = np.sum(peak_region)
            if total_mass > 0:
                indices = np.arange(left, right + 1, dtype=np.float32)
                ball_x_pixel = np.average(indices, weights=peak_region)
            else:
                ball_x_pixel = float(peak_center)
            ball_x_global = ball_x_pixel + roi_x_min
            debug["ball_x_pixel"] = int(ball_x_global)
            ball_pos_cm = (ball_x_global - center_pixel) * scale_k
            ball_pos_cm = max(-half_bar, min(half_bar, ball_pos_cm))

            peak_width = right - left + 1
            if self.PEAK_WIDTH_MIN <= peak_width <= self.PEAK_WIDTH_MAX:
                raw_detected = True

        if raw_detected and self.hough is not None:
            hough_x, hough_r = self.hough.refine(roi_gray, ball_x_pixel)
            if hough_x is not None:
                ball_x_global = hough_x + roi_x_min
                debug["ball_x_pixel"] = int(ball_x_global)
                debug["hough_r"] = int(hough_r)
                ball_pos_cm = (ball_x_global - center_pixel) * scale_k
                ball_pos_cm = max(-half_bar, min(half_bar, ball_pos_cm))

        if raw_detected:
            area = cv2.countNonZero(thresh_clean)
            _min = area_min if area_min is not None else self.AREA_MIN
            _max = area_max if area_max is not None else self.AREA_MAX
            if area < _min or area > _max:
                raw_detected = False
                ball_pos_cm = None
                debug.pop("ball_x_pixel", None)

        if raw_detected:
            if self._last_raw_pos is not None and \
               abs(ball_pos_cm - self._last_raw_pos) > self.POS_JUMP_MAX_CM:
                raw_detected = False
                ball_pos_cm = None
                debug.pop("ball_x_pixel", None)

        if raw_detected:
            self._last_raw_pos = ball_pos_cm
            self._confirm_count += 1
            if self._confirm_count < self.CONFIRM_FRAMES:
                ball_pos_cm = None
        else:
            self._confirm_count = 0
            self._last_raw_pos = None
            ball_pos_cm = None

        self._is_tracking = (self._confirm_count >= self.CONFIRM_FRAMES)

        return ball_pos_cm, debug

    def detect_simple(self, roi_gray, simple_threshold, morph_kernel_size,
                      projection_snr, center_pixel, scale_k, half_bar,
                      area_min=None, area_max=None, roi_x_min=0):
        roi_gray = self._preprocess(roi_gray)

        _, thresh = cv2.threshold(roi_gray, simple_threshold, 255, cv2.THRESH_BINARY_INV)

        ks_m = max(1, morph_kernel_size)
        if ks_m % 2 == 0:
            ks_m += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_m, ks_m))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        y_projection = np.sum(thresh_clean, axis=0, dtype=np.float32)
        max_val = np.max(y_projection)

        debug = {
            "diff": thresh,
            "thresh": thresh_clean,
            "projection": y_projection,
            "max_val": max_val,
            "bg_status": "simple",
        }

        ball_pos_cm = None
        raw_detected = False

        mean_proj = float(np.mean(y_projection))
        baseline = max(mean_proj, 1.0)
        snr = max_val / baseline

        eff_snr = projection_snr
        if self._is_tracking:
            eff_snr = projection_snr * 0.7

        if max_val > 10 and snr > eff_snr:
            half_max = max_val / 2.0
            above = y_projection > half_max

            peak_center = int(np.argmax(y_projection))
            left = peak_center
            while left > 0 and above[left - 1]:
                left -= 1
            right = peak_center
            while right < len(above) - 1 and above[right + 1]:
                right += 1

            peak_region = y_projection[left:right + 1]
            total_mass = np.sum(peak_region)
            if total_mass > 0:
                indices = np.arange(left, right + 1, dtype=np.float32)
                ball_x_pixel = np.average(indices, weights=peak_region)
            else:
                ball_x_pixel = float(peak_center)
            ball_x_global = ball_x_pixel + roi_x_min
            debug["ball_x_pixel"] = int(ball_x_global)
            ball_pos_cm = (ball_x_global - center_pixel) * scale_k
            ball_pos_cm = max(-half_bar, min(half_bar, ball_pos_cm))

            peak_width = right - left + 1
            if self.PEAK_WIDTH_MIN <= peak_width <= self.PEAK_WIDTH_MAX:
                raw_detected = True

        if raw_detected and self.hough is not None:
            hough_x, hough_r = self.hough.refine(roi_gray, ball_x_pixel)
            if hough_x is not None:
                ball_x_global = hough_x + roi_x_min
                debug["ball_x_pixel"] = int(ball_x_global)
                debug["hough_r"] = int(hough_r)
                ball_pos_cm = (ball_x_global - center_pixel) * scale_k
                ball_pos_cm = max(-half_bar, min(half_bar, ball_pos_cm))

        if raw_detected:
            area = cv2.countNonZero(thresh_clean)
            _min = area_min if area_min is not None else self.AREA_MIN
            _max = area_max if area_max is not None else self.AREA_MAX
            if area < _min or area > _max:
                raw_detected = False
                ball_pos_cm = None
                debug.pop("ball_x_pixel", None)

        if raw_detected:
            if self._last_raw_pos is not None and \
               abs(ball_pos_cm - self._last_raw_pos) > self.POS_JUMP_MAX_CM:
                raw_detected = False
                ball_pos_cm = None
                debug.pop("ball_x_pixel", None)

        if raw_detected:
            self._last_raw_pos = ball_pos_cm
            self._confirm_count += 1
            if self._confirm_count < self.CONFIRM_FRAMES:
                ball_pos_cm = None
        else:
            self._confirm_count = 0
            self._last_raw_pos = None
            ball_pos_cm = None

        self._is_tracking = (self._confirm_count >= self.CONFIRM_FRAMES)

        return ball_pos_cm, debug
