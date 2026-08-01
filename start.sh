#!/bin/bash

# ===== 机器人主程序自启动脚本 =====

cd /home/guga/elc_2026/src  # 进入代码目录

while true; do
    python3 main.py          # 运行主程序
    sleep 3                  # 崩溃后等 3 秒重启
done