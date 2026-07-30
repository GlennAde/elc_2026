import cv2
import numpy as np

from hough import HoughRefiner


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
        cv2.putText(
            frame, "ROI Area", (10, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
        )


class Visualizer:
    """视觉绘制与调试窗口工具类"""

    @staticmethod
    def draw_ball_overlay(frame, ball_pos_cm, ball_x_pixel, roi_y_min, roi_y_max,
                          velocity_cm_s=None):
        roi_center_y = int((roi_y_min + roi_y_max) / 2)
        cv2.circle(frame, (int(ball_x_pixel), roi_center_y), 12, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Raw Pos: {ball_pos_cm:.2f} cm",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    @staticmethod
    def draw_pose_overlay(frame, position_cm, velocity_cm_s):
        cv2.putText(
            frame,
            f"Filtered Pos: {position_cm:.2f} cm",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        direction = "->" if velocity_cm_s > 0 else "<-" if velocity_cm_s < 0 else "--"
        cv2.putText(
            frame,
            f"Velocity: {velocity_cm_s:.2f} cm/s {direction}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
    """钢球检测核心处理类"""

    AREA_MIN = 500
    AREA_MAX = 15000
    PEAK_WIDTH_MIN = 3
    PEAK_WIDTH_MAX = 120
    POS_JUMP_MAX_CM = 2.0
    CONFIRM_FRAMES = 2

    def __init__(self, clahe_clip=2.0, adaptive_bg_alpha=0.98, hough_cfg=None):
        self.bg_gray = None
        self.bg_initialized = False
        self.adaptive_bg_alpha = adaptive_bg_alpha
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
        self._no_raw_detect_count = 0
        self._is_tracking = False

    @property
    def should_update_background(self):
        return self._no_raw_detect_count >= 5 and self.bg_initialized

    def _preprocess(self, roi_gray):
        if self.clahe is not None:
            return self.clahe.apply(roi_gray)
        return roi_gray

    def capture_background(self, roi_gray):
        processed = self._preprocess(roi_gray)
        self.bg_gray = processed.copy()
        self.bg_initialized = True
        self._last_raw_pos = None
        self._confirm_count = 0
        self._no_raw_detect_count = 0
        self._is_tracking = False

    def has_background(self):
        return self.bg_initialized

    def update_background(self, roi_gray):
        if not self.bg_initialized:
            return
        processed = self._preprocess(roi_gray)
        alpha = self.adaptive_bg_alpha
        self.bg_gray = cv2.addWeighted(processed, 1.0 - alpha,
                                       self.bg_gray, alpha, 0)
        self._no_raw_detect_count = 0

    def detect(self, roi_gray, diff_threshold, morph_kernel_size,
               projection_snr, calib):
        if self.bg_gray is None:
            return None, {}

        roi_gray = self._preprocess(roi_gray)

        diff = cv2.absdiff(roi_gray, self.bg_gray)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        ks = max(1, morph_kernel_size)
        if ks % 2 == 0:
            ks += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        y_projection = np.sum(thresh_clean, axis=0, dtype=np.float32)
        max_val = np.max(y_projection)

        debug = {
            "diff": diff,
            "thresh": thresh_clean,
            "projection": y_projection,
            "max_val": max_val,
        }

        ball_pos_cm = None
        raw_detected = False

        # A. 峰噪比 (SNR)：max / mean(全列投影)，与 DIFF_THRESH 和光照解耦
        mean_proj = float(np.mean(y_projection))
        baseline = max(mean_proj, 1.0)
        snr = max_val / baseline

        eff_snr = projection_snr
        if self._is_tracking:
            eff_snr = projection_snr * 0.7

        if max_val > 10 and snr > eff_snr:
            total_mass = np.sum(y_projection)
            if total_mass > 0:
                indices = np.arange(len(y_projection), dtype=np.float32)
                ball_x_pixel = np.average(indices, weights=y_projection)
            else:
                ball_x_pixel = float(np.argmax(y_projection))
            debug["ball_x_pixel"] = int(ball_x_pixel)
            ball_pos_cm = calib.px_to_cm_clamped(ball_x_pixel)

            # B. 波峰半高宽校验：小球 ~25-55px / 噪声尖刺 <5px / 阴影 >120px
            half_max = max_val / 2.0
            peak_width = int(np.sum(y_projection > half_max))
            if self.PEAK_WIDTH_MIN <= peak_width <= self.PEAK_WIDTH_MAX:
                raw_detected = True

        # B2. 霍夫圆精修：修正因光照不对称造成的投影质心偏位
        if raw_detected and self.hough is not None:
            hough_x, hough_r = self.hough.refine(roi_gray, ball_x_pixel)
            if hough_x is not None:
                ball_x_pixel = hough_x
                debug["ball_x_pixel"] = int(ball_x_pixel)
                debug["hough_r"] = int(hough_r)
                ball_pos_cm = calib.px_to_cm_clamped(ball_x_pixel)

        # C. 面积校验：排除噪点团块和大面积光照伪影
        if raw_detected:
            area = cv2.countNonZero(thresh_clean)
            if area < self.AREA_MIN or area > self.AREA_MAX:
                raw_detected = False
                ball_pos_cm = None
                debug.pop("ball_x_pixel", None)

        # D. 位置连续性：物理上 8ms 内不可能跳变 >2cm
        if raw_detected:
            if self._last_raw_pos is not None and \
               abs(ball_pos_cm - self._last_raw_pos) > self.POS_JUMP_MAX_CM:
                raw_detected = False
                ball_pos_cm = None
                debug.pop("ball_x_pixel", None)

        # E. 多帧确认 + 安全背景更新计数
        if raw_detected:
            self._last_raw_pos = ball_pos_cm
            self._confirm_count += 1
            self._no_raw_detect_count = 0
            if self._confirm_count < self.CONFIRM_FRAMES:
                ball_pos_cm = None
        else:
            self._confirm_count = 0
            self._last_raw_pos = None
            self._no_raw_detect_count += 1
            ball_pos_cm = None

        self._is_tracking = (self._confirm_count >= self.CONFIRM_FRAMES)

        return ball_pos_cm, debug
