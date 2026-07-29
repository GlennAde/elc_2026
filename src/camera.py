import os

import cv2
import numpy as np
import yaml


class Camera:
    def __init__(
        self,
        device_id=0,
        width=640,
        height=480,
        fps=120,
        camera_matrix=None,
        dist_coeff=None,
        config_path="../config/camera_matrix.yaml",
    ):
        # 基本参数
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

        # 内参及去畸变标志位
        self.use_calibration = False
        self.camera_matrix = camera_matrix
        self.dist_coeff = dist_coeff

        # 如果没有直接传入矩阵，尝试从 yaml 文件加载
        if self.camera_matrix is None and config_path:
            self._load_camera_matrix(config_path)

    def _load_camera_matrix(self, config_path):
        """尝试读取相机内参 YAML 文件，失败则降级使用默认配置"""
        if not os.path.exists(config_path):
            print(
                f"[Camera Info] 未找到内参文件 '{config_path}'，使用默认配置（不矫正畸变）。"
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                calib_cfg = yaml.safe_load(f)

            if calib_cfg and calib_cfg.get("use_calibration", False):
                self.camera_matrix = np.array(
                    calib_cfg["camera_matrix"], dtype=np.float32
                )
                self.dist_coeff = np.array(calib_cfg["dist_coeff"], dtype=np.float32)
                self.use_calibration = True
                print(f"[Camera Info] 成功加载相机内参配置文件 '{config_path}'！")
            else:
                print(f"[Camera Info] 内参配置文件 '{config_path}' 中未开启畸变矫正。")

        except Exception as e:
            print(f"[Camera Warning] 读取内参文件失败: {e}，降级使用默认配置。")
            self.use_calibration = False

    def open(self):
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"[Camera Error] 无法打开摄像头 ID: {self.device_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 锁定曝光

        print(
            f"[Camera Info] 摄像头 ID:{self.device_id} 打开成功！({self.width}x{self.height}"
            f" @ {self.fps}fps)"
        )
        return True

    def get_frame(self):
        """获取最新一帧图像（如果开启内参矫正，则自动进行去畸变处理）"""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # 预留相机内参去畸变接口（若配置了内参且开启，则自动矫正）
        if (
            self.use_calibration
            and self.camera_matrix is not None
            and self.dist_coeff is not None
        ):
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeff)

        return True, frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Camera Info] 摄像头已释放")
