import struct
import serial


class SerialSender:
    """串口发送器，将位姿数据发送给下位机 MCU

    数据包格式（小端序，共 9 字节）:
        uint8   header         1B  帧头 0xA5
        float   position_cm    4B  小球位置(cm)，-12.5 ~ 12.5
        float   velocity_cm_s  4B  小球速度(cm/s)，带正负方向
    """

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._enabled = False

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.001)
            self._enabled = True
            print(f"[Serial Info] 串口 {self.port} @ {self.baudrate} 打开成功")
            return True
        except Exception as e:
            print(f"[Serial Warning] 无法打开串口 {self.port}: {e}")
            self._enabled = False
            return False

    def send(self, position_cm, velocity_cm_s):
        if not self._enabled or self.ser is None:
            return

        packet = struct.pack(
            "<Bff",
            0xA5,
            float(position_cm),
            float(velocity_cm_s),
        )

        try:
            self.ser.write(packet)
        except Exception as e:
            print(f"[Serial Error] 发送失败: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[Serial Info] 串口已关闭")
        self._enabled = False

    @property
    def enabled(self):
        return self._enabled
