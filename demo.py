import cv2
import numpy as np

# ==================== 1. 参数配置 ====================
# 假设你的桌面/摆杆在画面的 Y 轴范围 (按实际画面调整)
ROI_Y_MIN, ROI_Y_MAX = 180, 300

# 比例映射参数 (假设 ROI 区域内 X 轴 500 像素对应实际 25cm)
PIXEL_LEFT = 70
PIXEL_RIGHT = 570
PIXEL_CENTER = (PIXEL_LEFT + PIXEL_RIGHT) / 2.0
SCALE_K = 25.0 / (PIXEL_RIGHT - PIXEL_LEFT)  # cm / pixel

# ==================== 2. 初始化摄像头 ====================
cap = cv2.VideoCapture(0)
# 固定曝光，防止环境光线跳变
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

bg_gray = None

print("==================================================")
print("桌面钢珠检测 Demo 启动！")
print("1. 请先清空画面中的钢珠/小球。")
print("2. 按键盘上的 's' 键捕获当前桌面作为【静态背景】。")
print("3. 把钢珠放入 ROI 框内，观察识别与一维投影效果。")
print("4. 按 'ESC' 或 'q' 键退出。")
print("==================================================")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面！")
        break

    # 绘制 ROI 区域指示框 (黄色矩形框)
    cv2.rectangle(frame, (0, ROI_Y_MIN), (frame.shape[1], ROI_Y_MAX), (0, 255, 255), 2)
    cv2.putText(
        frame,
        "ROI Area",
        (10, ROI_Y_MIN - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    # 1. 裁剪 ROI 区域（只保留摆杆/滚动轨道的这一带）
    roi = frame[ROI_Y_MIN:ROI_Y_MAX, :]

    # 2. 灰度化与高斯滤波
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 按 's' 键捕获背景
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        bg_gray = gray.copy()
        print(">> 【背景截取成功】现在可以放入钢珠测试了！ <<")

    # 4. 算法核心处理（背景已抓取后触发）
    if bg_gray is not None:
        # 【步骤一：背景差分】
        diff = cv2.absdiff(gray, bg_gray)

        # 【步骤二：二值化】 (阈值设为 30，遮挡像素变化超过 30 变白)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # 【步骤三：形态学闭运算】 (填补钢珠表面的反光空洞，把两半缝合为一个整体)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 【步骤四：一维坍缩投影】 (沿 Y 轴求和，把二维矩阵压缩成一维数组)
        # axis=0 表示纵向压扁，得到长度为 画面宽度 的一维数组
        y_projection = np.sum(thresh_clean, axis=0)

        # 寻找投影波峰（即差分响应最强的位置，即钢珠所在列）
        max_val = np.max(y_projection)

        # 建立一个黑底画布来可视化显示这个一维波峰
        proj_canvas = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)

        # 阈值判断：如果一维响应值大于 500，说明有钢珠进入
        if max_val > 500:
            ball_x_pixel = int(np.argmax(y_projection))

            # 转换为厘米坐标
            ball_pos_cm = (ball_x_pixel - PIXEL_CENTER) * SCALE_K
            ball_pos_cm = max(-12.5, min(12.5, ball_pos_cm))

            # 在原图 ROI 上标记钢珠位置
            roi_center_y = int((ROI_Y_MIN + ROI_Y_MAX) / 2)
            cv2.circle(frame, (ball_x_pixel, roi_center_y), 12, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"Ball Pos: {ball_pos_cm:.2f} cm",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # 在一维投影窗口绘制红色标记线
            cv2.line(
                proj_canvas, (ball_x_pixel, 0), (ball_x_pixel, 150), (0, 0, 255), 2
            )

        # 绘制一维投影曲线 (白色波峰)
        norm_proj = (y_projection / (np.max(y_projection) + 1e-5) * 120).astype(
            np.int32
        )
        for x in range(1, len(norm_proj)):
            cv2.line(
                proj_canvas,
                (x - 1, 140 - norm_proj[x - 1]),
                (x, 140 - norm_proj[x]),
                (255, 255, 255),
                1,
            )

        # 显示辅助调试窗口
        cv2.imshow("1. Diff Image", diff)  # 差分原图
        cv2.imshow("2. Thresh + Morphology", thresh_clean)  # 形态学处理后的二值图
        cv2.imshow("3. 1D Projection Curve", proj_canvas)  # 一维波峰曲线

    # 显示主画面
    cv2.imshow("Main View (Demo)", frame)

    if key == 27 or key == ord("q"):  # ESC 或 q 退出
        break

cap.release()
cv2.destroyAllWindows()
