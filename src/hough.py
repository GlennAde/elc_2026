import cv2
import numpy as np


class HoughRefiner:
    """霍夫圆精修：修正投影质心因光照不对称造成的偏位"""

    def __init__(self, dp=1, min_dist=50, param1=80, param2=12,
                 min_radius=12, max_radius=32, max_deviation=25):
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_deviation = max_deviation

    def refine(self, roi_gray, projection_x):
        circles = cv2.HoughCircles(
            roi_gray, cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return None, None

        circles = circles[0]
        best_x, best_r, best_dist = None, None, self.max_deviation
        for x, y, r in circles:
            dist = abs(x - projection_x)
            if dist < best_dist:
                best_dist = dist
                best_x, best_r = x, r

        return best_x, best_r
