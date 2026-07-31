import cv2
import numpy as np


class Tuner:
    WINDOW_NAME = "Parameter Tuner"

    TRACKBARS = [
        ("ROI_Y_MIN", 180, 480),
        ("ROI_Y_MAX", 300, 480),
        ("PIXEL_LEFT", 70, 640),
        ("PIXEL_RIGHT", 570, 640),
        ("CENTER_PIXEL", 320, 640),
        ("BAR_LENGTH_CM", 25, 50),
        ("SIMPLE_THRESHOLD", 90, 255),
        ("MORPH_KERNEL_SIZE", 16, 17),
        ("PROJECTION_SNR", 51, 100),
        ("AREA_MIN", 500, 5000),
        ("AREA_MAX", 15000, 50000),
    ]

    @staticmethod
    def _noop(_x):
        pass

    @classmethod
    def init(cls, defaults=None):
        cv2.namedWindow(cls.WINDOW_NAME)
        for name, default, max_val in cls.TRACKBARS:
            val = int(defaults.get(name.lower(), default)) if defaults else default
            cv2.createTrackbar(name, cls.WINDOW_NAME, val, max_val, cls._noop)

    @classmethod
    def get_params(cls):
        return {
            "roi_y_min": cv2.getTrackbarPos("ROI_Y_MIN", cls.WINDOW_NAME),
            "roi_y_max": cv2.getTrackbarPos("ROI_Y_MAX", cls.WINDOW_NAME),
            "pixel_left": cv2.getTrackbarPos("PIXEL_LEFT", cls.WINDOW_NAME),
            "pixel_right": cv2.getTrackbarPos("PIXEL_RIGHT", cls.WINDOW_NAME),
            "center_pixel": cv2.getTrackbarPos("CENTER_PIXEL", cls.WINDOW_NAME),
            "bar_length_cm": cv2.getTrackbarPos("BAR_LENGTH_CM", cls.WINDOW_NAME),
            "simple_threshold": cv2.getTrackbarPos("SIMPLE_THRESHOLD", cls.WINDOW_NAME),
            "morph_kernel_size": cv2.getTrackbarPos("MORPH_KERNEL_SIZE", cls.WINDOW_NAME),
            "projection_snr": cv2.getTrackbarPos("PROJECTION_SNR", cls.WINDOW_NAME) / 10.0,
            "area_min": cv2.getTrackbarPos("AREA_MIN", cls.WINDOW_NAME),
            "area_max": cv2.getTrackbarPos("AREA_MAX", cls.WINDOW_NAME),
            "spatial_bg_kernel_size": cv2.getTrackbarPos("SPATIAL_BG_KERNEL", cls.WINDOW_NAME),
            "adaptive_block_size": cv2.getTrackbarPos("ADAPTIVE_BLOCK", cls.WINDOW_NAME),
            "adaptive_c": cv2.getTrackbarPos("ADAPTIVE_C", cls.WINDOW_NAME),
        }

    @staticmethod
    def show_info(params):
        """在独立窗口中显示所有参数当前数值"""
        cols = [
            ("ROI Top", str(params["roi_y_min"])),
            ("ROI Bottom", str(params["roi_y_max"])),
            ("Pixel Left", str(params["pixel_left"])),
            ("Pixel Right", str(params["pixel_right"])),
            ("Center Pixel", str(params.get("center_pixel", 320))),
            ("Bar Length", str(params.get("bar_length_cm", 25)) + " cm"),
            ("Simple Threshold", str(params.get("simple_threshold", 90))),
            ("Morph Kernel", str(params["morph_kernel_size"])),
            ("SNR Threshold", f"{params['projection_snr']:.1f}"),
            ("Area Min", str(params.get("area_min", 500))),
            ("Area Max", str(params.get("area_max", 15000))),
            ("Spatial BG Kernel", str(params.get("spatial_bg_kernel_size", 31))),
            ("Adaptive Block", str(params.get("adaptive_block_size", 31))),
            ("Adaptive C", str(params.get("adaptive_c", 5))),
        ]
        canvas = np.zeros((len(cols) * 34 + 2, 280, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        for i, (label, value) in enumerate(cols):
            y = 26 + i * 34
            cv2.putText(canvas, f"{label}: {value}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.imshow("Parameter Info", canvas)
