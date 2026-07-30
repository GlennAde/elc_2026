改动汇总
文件 改动
src/calib.py 新建 — 多点标定类，np.interp 插值 + 线性降级
src/kalman.py update() 接受 dt 参数，动态更新转移矩阵
src/processor.py detect() 签名 (pixel_center, scale_k) → (calib)
src/main.py 加载标定 + tickCount 实测帧间隔 + 传 calib 和 dt
config/config.yaml 新增 calibration.file: "config/px2cm.npz"
scripts/calibrate.py 新建 — 交互式标定，按 r 逐点录，w 保存

数据流变化
改前:  ball_x_pixel → (px - pixel_center) × 25/500 → cm  (线性，透视误差)
改后:  ball_x_pixel → np.interp(px, 标定表) → cm          (多点，消除透视)
       Kalman dt: 固定 1/120 → tickCount 实测

参数调整表
检测参数（Tuner trackbar / config）
参数 范围 当前值 效果 调法
roi_y_min/max 0~480 180/300 横杆在画面中的 Y 范围 框住整根杆，留 10px 余量
pixel_left/right 0~640 70/570 横杆两端 X 像素（线性降级用） 对准杆的实际物理两端
diff_threshold 0~255 106 背景差分灵敏度 太高→球碎片化；太低→噪点；调到球完整且背景干净
morph_kernel_size 3~17 16 形态学闭运算核大小 调大→填补反光空洞；太大→两球粘连误判
projection_snr 3.0~10.0 5.1 峰噪比阈值(高=严格) 有球时 SNr 几十，调到 > 峰值 SNR 的 1/3 即可

检测参数（config only，无 trackbar）
参数 位置 当前值 效果 调法
clahe_clip detector 2.0 局部直方图均衡强度 0 关闭；2~4 适应阴影；太高会增强噪点
adaptive_bg_alpha detector 0.98 背景更新速率 接近 1=保守(慢适应)；0.9=激进(快适应）

卡尔曼参数（config only）
参数 当前值 效果 症状 → 调法
q_pos 0.01 位置过程噪声 位置滞后 → 调大；位置抖 → 调小
q_vel 0.5 速度过程噪声 速度响应慢 → 调大；速度噪声大 → 调小
r_meas 0.1 测量噪声 滤波太激进(滞后) → 调大；滤波太弱(噪声) → 调小
卡尔曼联调口诀：先调 r_meas 定平滑度 → 再调 q_vel 定速度响应 → 最后微调 q_pos

霍夫参数（config only，enabled: false 时不生效）

参数 当前值 效果 调法
param1 80 Canny 边缘检测高阈值 降低 → 更多边缘；太高 → 圆边缘丢失
param2 12 圆心累加器阈值 主旋钮：找不到圆→调低(810)；伪圆多→调高(1520)
min_radius 12 最小圆半径(px) 按实际球像素半径设
max_radius 32 最大圆半径(px) 同上
max_deviation 25 霍夫圆与投影质心最大允许偏差(px) 光照严重不对称时适当放宽

标定参数（一次性，用 scripts/calibrate.py 做）
步骤 操作
1 python scripts/calibrate.py
2 按 s 截背景
3 屏幕提示"put ball at -12.5 cm"，放球，按 r
4 依次完成 11 个点，按 w 保存
5 config/px2cm.npz 已生成，重启主程序自动加载
