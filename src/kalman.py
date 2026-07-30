import cv2
import numpy as np


class BallKalmanFilter:
    """一维位置+速度卡尔曼滤波器，基于 cv2.KalmanFilter

    状态向量: [x, vx]  — 位置(cm), 速度(cm/s)
    观测向量: [x]       — 位置测量值(cm)
    """

    def __init__(self, dt=1.0/120, q_pos=0.01, q_vel=0.5, r_meas=0.1):
        self.dt = dt

        self.kf = cv2.KalmanFilter(2, 1, 0)

        self.kf.transitionMatrix = np.array([
            [1, dt],
            [0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.array([
            [q_pos, 0],
            [0, q_vel],
        ], dtype=np.float32)

        self.kf.measurementNoiseCov = np.array([
            [r_meas],
        ], dtype=np.float32)

        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 100.0

        self.kf.statePost = np.zeros((2, 1), dtype=np.float32)

    def predict(self):
        predicted = self.kf.predict()
        return predicted[0, 0], predicted[1, 0]

    def correct(self, measurement):
        meas = np.array([[measurement]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        return corrected[0, 0], corrected[1, 0]

    def update(self, measurement=None, dt=None):
        if dt is not None:
            self.kf.transitionMatrix[0, 1] = dt
        pred_pos, pred_vel = self.predict()
        if measurement is not None:
            corr_pos, corr_vel = self.correct(measurement)
            return corr_pos, corr_vel
        return pred_pos, pred_vel

    def reset(self):
        self.kf.statePost = np.zeros((2, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 100.0
