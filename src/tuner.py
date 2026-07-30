import cv2


class Tuner:
    WINDOW_NAME = "Parameter Tuner"

    TRACKBARS = [
        ("ROI_Y_MIN", 180, 480),
        ("ROI_Y_MAX", 300, 480),
        ("PIXEL_LEFT", 70, 640),
        ("PIXEL_RIGHT", 570, 640),
        ("DIFF_THRESHOLD", 30, 255),
        ("MORPH_KERNEL_SIZE", 7, 17),
        ("PROJECTION_SNR", 35, 100),
    ]

    @staticmethod
    def _noop(_x):
        pass

    @classmethod
    def init(cls, defaults=None):
        cv2.namedWindow(cls.WINDOW_NAME)
        for name, default, max_val in cls.TRACKBARS:
            val = defaults.get(name.lower(), default) if defaults else default
            cv2.createTrackbar(name, cls.WINDOW_NAME, val, max_val, cls._noop)

    @classmethod
    def get_params(cls):
        return {
            "roi_y_min": cv2.getTrackbarPos("ROI_Y_MIN", cls.WINDOW_NAME),
            "roi_y_max": cv2.getTrackbarPos("ROI_Y_MAX", cls.WINDOW_NAME),
            "pixel_left": cv2.getTrackbarPos("PIXEL_LEFT", cls.WINDOW_NAME),
            "pixel_right": cv2.getTrackbarPos("PIXEL_RIGHT", cls.WINDOW_NAME),
            "diff_threshold": cv2.getTrackbarPos("DIFF_THRESHOLD", cls.WINDOW_NAME),
            "morph_kernel_size": cv2.getTrackbarPos("MORPH_KERNEL_SIZE", cls.WINDOW_NAME),
            "projection_snr": cv2.getTrackbarPos("PROJECTION_SNR", cls.WINDOW_NAME) / 10.0,
        }
