#!/bin/bash

# SN-VLA データセット収集のクイックスタート
# このスクリプトは、データ収集に必要な依存関係をインストールし、
# 設定をガイドします。

set -e

echo "========================================"
echo "SN-VLA Dataset Recording Quick Start"
echo "========================================"
echo ""

# 1. 依存関係の確認
echo "[Step 1/4] Checking dependencies..."
python -c "import cv2" 2>/dev/null || {
    echo "OpenCV not found. Installing..."
    pip install opencv-python
}
python -c "import numpy" 2>/dev/null || {
    echo "NumPy not found. Installing..."
    pip install numpy
}
echo "✓ Dependencies OK"
echo ""

# 2. ロボットポートの検出
echo "[Step 2/4] Detecting robot ports..."
echo "Available serial ports:"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  (No USB serial devices found)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    ls /dev/tty.usb* 2>/dev/null || echo "  (No USB serial devices found)"
fi
echo ""
echo "Please edit examples/snvla/record.py and set:"
echo "  FOLLOWER_PORT = '/dev/tty.your_follower_port'"
echo "  LEADER_PORT = '/dev/tty.your_leader_port'"
echo ""

# 3. 設定の確認
echo "[Step 3/4] Configuration checklist:"
echo "Please ensure the following are configured in record.py:"
echo "  ☐ HF_REPO_ID (your Hugging Face repository)"
echo "  ☐ TASK_DESCRIPTION (task description)"
echo "  ☐ NARRATION_LIST (narration texts for your task)"
echo "  ☐ FOLLOWER_PORT and LEADER_PORT"
echo ""

# 4. 実行方法の表示
echo "[Step 4/4] Ready to record!"
echo ""
echo "To start recording:"
echo "  python examples/snvla/record.py"
echo ""
echo "During recording:"
echo "  [Enter]       - Insert next narration / End episode"
echo "  [i-hold 1s]   - Interrupt and delete episode"
echo "  [q]           - Stop recording"
echo ""
echo "To verify dataset after recording:"
echo "  python examples/snvla/varify_dataset.py <your-repo-id>"
echo ""
echo "========================================"
echo "Happy recording! 🤖📹"
echo "========================================"
