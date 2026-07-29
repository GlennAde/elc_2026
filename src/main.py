import os
import cv2
import yaml

from camera import Camera
from processor import BallDetector, ROIManager, Visualizer
from tuner import Tuner

# 获取当前 main.py 所在的真实绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 自动推导上一级目录中的 config 文件夹路径
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
DEFAULT_MATRIX_PATH = os.path.join(PROJECT_ROOT, "config", "camera_matrix.yaml")

class Application:
    def __init__(self, config_path="/home/glennade/elc_2026/config/config.yaml"):
        # 1. 读取主配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 2. 初始化相机（Camera 内部会自动去读取指定的内参 yaml）
        cam_cfg = self.cfg["camera"]
        self.cam = Camera(
            device_id=cam_cfg["device_id"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
            config_path="/home/glennade/elc_2026/config/camera_matrix.yaml",  # 统一对应 camera.py 中的参数名
        )

        self.detector = BallDetector()
        self.bar_length_cm = self.cfg["system"]["bar_length_cm"]

    def run(self):
        if not self.cam.open():
            return

        # 初始化调参窗口
        Tuner.init(defaults=self.cfg.get("tuner_defaults"))

        while True:
            ret, frame = self.cam.get_frame()
            if not ret:
                break

            p = Tuner.get_params()
            roi_y_min, roi_y_max = p["roi_y_min"], p["roi_y_max"]
            pixel_left, pixel_right = p["pixel_left"], p["pixel_right"]

            pixel_center = (pixel_left + pixel_right) / 2.0
            scale_k = self.bar_length_cm / max(1, pixel_right - pixel_left)

            # 绘制并提取 ROI
            ROIManager.draw_roi_overlay(frame, roi_y_min, roi_y_max)
            roi_gray = ROIManager.extract_roi(frame, roi_y_min, roi_y_max)

            key = cv2.waitKey(1) & 0xFF

            # 按 's' 捕获静态背景
            if key == ord("s"):
                self.detector.capture_background(roi_gray)
                print(">> [System] 背景截取成功！")

            # 钢球检测
            if self.detector.has_background():
                ball_pos_cm, debug = self.detector.detect(
                    roi_gray,
                    pixel_center,
                    scale_k,
                    p["diff_threshold"],
                    p["morph_kernel_size"],
                    p["projection_min_val"],
                )

                if debug:
                    proj_canvas = Visualizer.build_projection_canvas(
                        debug["projection"], frame.shape[1], debug.get("ball_x_pixel")
                    )
                    Visualizer.show_debug_windows(
                        debug["diff"], debug["thresh"], proj_canvas
                    )

                    if ball_pos_cm is not None:
                        Visualizer.draw_ball_overlay(
                            frame,
                            ball_pos_cm,
                            debug["ball_x_pixel"],
                            roi_y_min,
                            roi_y_max,
                        )

            cv2.imshow("Main View", frame)

            if key == 27 or key == ord("q"):
                break

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Application()
    try:
        app.run()
    finally:
        app.release()
