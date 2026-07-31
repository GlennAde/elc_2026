import os
import cv2
import yaml

from camera import Camera
from processor import BallDetector, ROIManager, Visualizer
from tuner import Tuner
from kalman import BallKalmanFilter
from serial_sender import SerialSender

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

        self.show_debug = self.cfg.get("gui", {}).get("debug_windows", True)
        self._static_params = self._build_static_params()
        self._last_frame_ts = None

        self._simple_threshold = det_cfg.get("simple_threshold", 90)

        self.serial = None
        ser_cfg = self.cfg.get("serial", {})
        if ser_cfg:
            self.serial = SerialSender(
                port=ser_cfg.get("port", "/dev/ttyCP210x"),
                baudrate=ser_cfg.get("baudrate", 115200),
            )

    def _build_static_params(self):
        d = self.cfg.get("tuner_defaults", {})
        pl = d.get("pixel_left", 70)
        pr = d.get("pixel_right", 570)
        return {
            "roi_y_min": d.get("roi_y_min", 180),
            "roi_y_max": d.get("roi_y_max", 300),
            "pixel_left": pl,
            "pixel_right": pr,
            "center_pixel": d.get("center_pixel", (pl + pr) // 2),
            "morph_kernel_size": d.get("morph_kernel_size", 7),
            "projection_snr": d.get("projection_snr", 35) / 10.0,
            "bar_length_cm": d.get("bar_length_cm", self.bar_length_cm),
            "area_min": d.get("area_min", 500),
            "area_max": d.get("area_max", 15000),
            "simple_threshold": d.get("simple_threshold", 90),
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
                Tuner.show_info(p)
            else:
                p = self._static_params

            roi_y_min, roi_y_max = p["roi_y_min"], p["roi_y_max"]

            # 实时帧间隔（替代固定 dt=1/120）
            now = cv2.getTickCount() / cv2.getTickFrequency()
            dt_real = now - self._last_frame_ts if self._last_frame_ts else 1.0 / 120.0
            dt_real = min(dt_real, 0.05)  # 上限 50ms，防止异常帧
            self._last_frame_ts = now

            roi_gray, roi_x_min = ROIManager.extract_roi(
                frame, roi_y_min, roi_y_max,
                roi_x_min=p["pixel_left"], roi_x_max=p["pixel_right"],
            )
            if roi_gray is None:
                continue

            if self.show_debug:
                ROIManager.draw_roi_overlay(frame, roi_y_min, roi_y_max,
                                            center_pixel=p.get("center_pixel"),
                                            roi_x_min=p["pixel_left"],
                                            roi_x_max=p["pixel_right"])

            key = cv2.waitKey(1) & 0xFF if self.show_debug else -1

            if key == ord("o"):
                if self.serial:
                    if self.serial.enabled:
                        self.serial.close()
                    else:
                        self.serial.open()

            filtered_pos = None
            filtered_vel = None

            ball_pos_cm, debug = self.detector.detect_simple(
                roi_gray,
                simple_threshold=p.get("simple_threshold", 90),
                morph_kernel_size=p["morph_kernel_size"],
                projection_snr=p["projection_snr"],
                center_pixel=p.get("center_pixel", (p["pixel_left"] + p["pixel_right"]) // 2),
                scale_k=self.bar_length_cm / (p["pixel_right"] - p["pixel_left"]) if p["pixel_right"] != p["pixel_left"] else 0.0,
                half_bar=self.bar_length_cm / 2.0,
                area_min=p.get("area_min"), area_max=p.get("area_max"),
                roi_x_min=roi_x_min,
            )

            if ball_pos_cm is not None:
                filtered_pos, filtered_vel = self.kf.update(ball_pos_cm, dt=dt_real)
            else:
                filtered_pos, filtered_vel = self.kf.update(dt=dt_real)

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
                        frame, debug["ball_x_pixel"],
                        roi_y_min, roi_y_max,
                    )

            if self.show_debug and filtered_pos is not None:
                Visualizer.draw_pose_overlay(frame, filtered_pos,
                                             filtered_vel or 0.0)

            if self.serial and self.serial.enabled:
                if filtered_pos is not None:
                    self.serial.send(filtered_pos, filtered_vel or 0.0)
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
