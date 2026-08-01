# 自启动 脚本命令
chmod +x run.sh          # 给执行权限
nohup ./run.sh &         # 后台运行，关闭终端也不停

pkill -f main.py         # 杀掉主程序
pkill -f run.sh          # 如果连守护脚本也想停

# 1. 创建服务文件（注意名字统一为 elc）
sudo nano /etc/systemd/system/elc.service

#  内容
[Unit]
Description=ELC Main Program
After=network.target

[Service]
Type=simple
User=guga
WorkingDirectory=/home/guga/elc_2026/src
ExecStart=/bin/bash /home/guga/elc_2026/start.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

# 查看服务是否开机自启
sudo systemctl is-enabled elc.service
# 应该输出：enabled

# 查看服务是否在运行
sudo systemctl is-active elc.service
# 应该输出：active

# 查看实时日志（确认 main.py 跑起来了）
sudo journalctl -u elc.service -f

# tips
确保你的 start.sh 有执行权限：

bash
chmod +x /home/guga/elc_2026/start.sh