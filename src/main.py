import os
import cv2
import yaml

from camera import Camera
from processor import BallDetector, ROIManager, Visualizer
from tuner import Tuner
from kalman import BallKalmanFilter
from serial import SerialSender
from calib import Calibration

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


class Application:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        cam_cfg = self.cfg["camera"]
        self.cam = Camera(
            device_id=cam_cfg["device_id"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
            config_path=os.path.join(PROJECT_ROOT, "config", "camera_matrix.yaml"),
        )

        det_cfg = self.cfg.get("detector", {})
        self.detector = BallDetector(
            clahe_clip=det_cfg.get("clahe_clip", 2.0),
            adaptive_bg_alpha=det_cfg.get("adaptive_bg_alpha", 0.98),
            hough_cfg=det_cfg.get("hough"),
        )

        kf_cfg = self.cfg.get("kalman", {})
        self.kf = BallKalmanFilter(
            dt=kf_cfg.get("dt", 1.0 / 120),
            q_pos=kf_cfg.get("q_pos", 0.01),
            q_vel=kf_cfg.get("q_vel", 0.5),
            r_meas=kf_cfg.get("r_meas", 0.1),
        )

        self.bar_length_cm = self.cfg["system"]["bar_length_cm"]

        calib_cfg = self.cfg.get("calibration", {})
        self.calib = self._load_calibration(calib_cfg)

        self.show_debug = self.cfg.get("gui", {}).get("debug_windows", True)
        self._static_params = self._build_static_params()
        self._auto_bg_frames = 0
        self._last_frame_ts = None

        self.serial = None
        ser_cfg = self.cfg.get("serial", {})
        if ser_cfg:
            self.serial = SerialSender(
                port=ser_cfg.get("port", "/dev/ttyUSB0"),
                baudrate=ser_cfg.get("baudrate", 115200),
            )

    def _load_calibration(self, calib_cfg):
        filepath = calib_cfg.get("file")
        if filepath:
            abs_path = os.path.join(PROJECT_ROOT, filepath) if not os.path.isabs(filepath) else filepath
            if os.path.exists(abs_path):
                print(f"[Calib Info] 加载标定文件 '{abs_path}'")
                return Calibration.from_file(abs_path)

        d = self.cfg.get("tuner_defaults", {})
        pl = d.get("pixel_left", 70)
        pr = d.get("pixel_right", 570)
        bar = self.cfg["system"]["bar_length_cm"]
        print(f"[Calib Info] 标定文件不存在，使用线性模式 ({pl}→{-bar/2}cm, {pr}→{bar/2}cm)")
        return Calibration.from_linear(pl, pr, bar)

    def _build_static_params(self):
        d = self.cfg.get("tuner_defaults", {})
        return {
            "roi_y_min": d.get("roi_y_min", 180),
            "roi_y_max": d.get("roi_y_max", 300),
            "pixel_left": d.get("pixel_left", 70),
            "pixel_right": d.get("pixel_right", 570),
            "diff_threshold": d.get("diff_threshold", 30),
            "morph_kernel_size": d.get("morph_kernel_size", 7),
            "projection_snr": d.get("projection_snr", 35) / 10.0,
        }

    def run(self):
        if not self.cam.open():
            return

        if self.show_debug:
            Tuner.init(defaults=self.cfg.get("tuner_defaults"))

        if self.serial:
            self.serial.open()

        while True:
            ret, frame = self.cam.get_frame()
            if not ret:
                break

            if self.show_debug:
                p = Tuner.get_params()
            else:
                p = self._static_params

            roi_y_min, roi_y_max = p["roi_y_min"], p["roi_y_max"]

            # 实时帧间隔（替代固定 dt=1/120）
            now = cv2.getTickCount() / cv2.getTickFrequency()
            dt_real = now - self._last_frame_ts if self._last_frame_ts else 1.0 / 120.0
            dt_real = min(dt_real, 0.05)  # 上限 50ms，防止异常帧
            self._last_frame_ts = now

            if self.show_debug:
                ROIManager.draw_roi_overlay(frame, roi_y_min, roi_y_max)
            roi_gray = ROIManager.extract_roi(frame, roi_y_min, roi_y_max)

            key = cv2.waitKey(1) & 0xFF if self.show_debug else -1

            if not self.show_debug and not self.detector.has_background():
                self._auto_bg_frames += 1
                if self._auto_bg_frames == 30:
                    self.detector.capture_background(roi_gray)
                    self.kf.reset()
                    print(">> [System] 无头模式：自动截取背景。")

            if key == ord("s"):
                self.detector.capture_background(roi_gray)
                self.kf.reset()
                print(">> [System] 背景截取成功！卡尔曼滤波器已重置。")

            if key == ord("o"):
                if self.serial:
                    if self.serial.enabled:
                        self.serial.close()
                    else:
                        self.serial.open()

            filtered_pos = None
            filtered_vel = None

            if self.detector.has_background():
                ball_pos_cm, debug = self.detector.detect(
                    roi_gray,
                    p["diff_threshold"], p["morph_kernel_size"],
                    p["projection_snr"], self.calib,
                )

                if ball_pos_cm is not None:
                    filtered_pos, filtered_vel = self.kf.update(ball_pos_cm, dt=dt_real)
                else:
                    filtered_pos, filtered_vel = self.kf.update(dt=dt_real)
                    if self.detector.should_update_background:
                        self.detector.update_background(roi_gray)

                if self.show_debug and debug:
                    proj_canvas = Visualizer.build_projection_canvas(
                        debug["projection"], frame.shape[1],
                        debug.get("ball_x_pixel"),
                    )
                    Visualizer.show_debug_windows(
                        debug["diff"], debug["thresh"], proj_canvas,
                    )

                    if ball_pos_cm is not None:
                        Visualizer.draw_ball_overlay(
                            frame, ball_pos_cm, debug["ball_x_pixel"],
                            roi_y_min, roi_y_max,
                        )

                if self.show_debug and filtered_pos is not None:
                    Visualizer.draw_pose_overlay(frame, filtered_pos,
                                                 filtered_vel or 0.0)

            if self.serial and self.serial.enabled:
                if self.show_debug:
                    Visualizer.draw_serial_status(frame, True)
                if filtered_pos is not None:
                    self.serial.send(filtered_pos, filtered_vel or 0.0)
            else:
                if self.show_debug:
                    Visualizer.draw_serial_status(frame, False)

            if self.show_debug:
                cv2.imshow("Main View", frame)

            if key == 27 or key == ord("q"):
                break

    def release(self):
        self.cam.release()
        if self.serial:
            self.serial.close()
        if self.show_debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Application()
    try:
        app.run()
    finally:
        app.release()
