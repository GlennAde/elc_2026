import cv2


class Camera:
    def __init__(self, device_id=0, width=640, height=480, fps=120):
        """
        初始化相机参数
        :param device_id: 摄像头设备号，一般 USB 摄像头为 0 或 1
        :param width: 画面宽度
        :param height: 画面高度
        :param fps: 帧率
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        """打开摄像头并配置底层参数"""
        self.cap = cv2.VideoCapture(
            self.device_id, cv2.CAP_V4L2
        )  # Linux/Jetson Nano 上推荐 V4L2 后端

        if not self.cap.isOpened():
            print(f"[Camera Error] 无法打开摄像头设备 ID: {self.device_id}")
            return False

        # 1. 设置分辨率与帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 2. 锁定曝光（极度关键：关闭自动曝光，防止光线变动干扰背景差分）
        # 0.25 或 1 通常表示手动模式 (Manual Mode)，具体数值依系统而定
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        print(
            f"[Camera Info] 摄像头 ID:{self.device_id} 打开成功！({self.width}x{self.height} @ {self.fps}fps)"
        )
        return True

    def get_frame(self):
        """获取最新一帧图像"""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            print("[Camera Warning] 画面抓取失败！")
            return False, None

        return True, frame

    def release(self):
        """释放摄像头资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Camera Info] 摄像头资源已释放。")


# ==================== 单独测试该模块 ====================
if __name__ == "__main__":
    cam = Camera(device_id=0, width=640, height=480)

    if cam.open():
        print("按 'q' 键退出相机测试...")
        while True:
            ret, frame = cam.get_frame()
            if not ret:
                break

            cv2.imshow("Camera Module Test", frame)

            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cam.release()
        cv2.destroyAllWindows()
