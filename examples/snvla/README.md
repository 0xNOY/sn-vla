# SN-VLA (Self-Narrating Vision-Language-Action) データ収集ツール

このディレクトリには、SN-VLAモデルの学習に必要なデータセットを作成するためのツールが含まれています。

## 概要

SN-VLAは、ロボットの行動と自然言語の実況を統合して学習するVision-Language-Actionモデルです。
このデータ収集ツールを使用すると、テレオペレーション中に実況文を挿入しながらデータセットを作成できます。

## ファイル構成

- **`record.py`**: データセット収集スクリプト（メインツール）
- **`varify_dataset.py`**: データセット検証スクリプト
- **`README.md`**: このファイル

## データセット形式

SN-VLAは以下のデータを必要とします:

1. **observation**: ロボットの観測（カメラ画像、状態など）
2. **action**: ロボットのアクション
3. **complementary_data**: 追加データ
   - `task`: タスクの説明（例: "Pick up the object and place it in the target position"）
   - `current_narration`: 現在のフレームの実況文（空文字列の場合はアクション生成モード）
   - `previous_narrations`: それまでの実況文のリスト

## 使用方法

### 1. 設定のカスタマイズ

`record.py` の冒頭にある設定パラメータを編集します:

```python
# ======================== 設定パラメータ ========================
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick up the object and place it in the target position"
HF_REPO_ID = "your-username/snvla_so101_pickplace"
DATASET_ROOT = Path("data")

# ロボット・テレオペレータの設定
FOLLOWER_PORT = "/dev/tty.usbmodem58760431541"
LEADER_PORT = "/dev/tty.usbmodem58760431551"

# 実況リスト（エピソードごとに使用）
NARRATION_LIST = [
    "Approaching the object with the gripper",
    "Grasping the object firmly",
    "Lifting the object upward",
    "Moving toward the target position",
    "Lowering the object carefully",
    "Releasing the object at the target",
]
```

### 2. データ収集の実行

```bash
python examples/snvla/record.py
```

### 3. キー操作

データ収集中に以下のキー操作が可能です:

| キー                | 動作                                                           |
| ------------------- | -------------------------------------------------------------- |
| **Enter**           | 実況リストの先頭要素を挿入（リストが空の場合はエピソード終了） |
| **i** (1秒間長押し) | エピソードを中断・削除                                         |
| **q**               | 記録を停止                                                     |

### 4. 画面表示

データ収集中、OpenCVウィンドウに以下が表示されます:

- **カメラ画像**: ロボットの視野（複数カメラは水平連結）
- **ステータス情報**:
  - エピソード経過時間 / 残り時間
  - タスク名
  - 残りの実況数
  - 次の実況文
  - 過去の実況文（最新3件）
  - キー操作ヘルプ

### 5. データセットの検証

データ収集後、以下のコマンドでデータセットを検証できます:

```bash
python examples/snvla/varify_dataset.py your-username/snvla_so101_pickplace
```

検証スクリプトは以下を確認します:
- データセットのメタデータ
- フィールドの存在確認
- 実況文の統計情報
- エピソードごとの情報

## データ収集のワークフロー

1. **準備**
   - ロボットとテレオペレータを接続
   - 環境をセットアップ
   - カメラの位置を調整

2. **エピソード開始**
   - スクリプトが自動的にエピソードを開始
   - OpenCVウィンドウにカメラ画像とステータスが表示される

3. **テレオペレーション + 実況挿入**
   - テレオペレータでロボットを操作
   - タスクの進行に合わせて **Enter** キーで実況を挿入
   - 実況はリストの順番通りに挿入される

4. **エピソード終了**
   - 実況リストが空の状態で **Enter** を押すとエピソード終了
   - または、指定時間（`EPISODE_TIME_SEC`）経過で自動終了

5. **エピソード保存**
   - スクリプトが自動的にエピソードを保存
   - 次のエピソードまで待機（`RESET_TIME_SEC`）

6. **データセットのアップロード**
   - すべてのエピソード完了後、Hugging Face Hubに自動アップロード

## 実況のタイミング

### 推奨される実況挿入のタイミング

- **動作の開始時**: 新しい行動フェーズが始まる瞬間
- **重要な変化**: オブジェクトの把持、リリースなど
- **サブゴール達成**: タスクの一部が完了した時点

### 例: Pick and Place タスク

```python
NARRATION_LIST = [
    "Approaching the object with the gripper",      # ← 物体に近づき始めた時
    "Grasping the object firmly",                   # ← グリッパーを閉じた時
    "Lifting the object upward",                    # ← 持ち上げ始めた時
    "Moving toward the target position",            # ← 目標位置への移動開始
    "Lowering the object carefully",                # ← 下ろし始めた時
    "Releasing the object at the target",           # ← グリッパーを開いた時
]
```

## トラブルシューティング

### データセットが作成されない

- ロボットのポートが正しいか確認
- `FOLLOWER_PORT` と `LEADER_PORT` を確認
- ロボットが正しく接続されているか確認

### カメラ画像が表示されない

- カメラが正しく接続されているか確認
- OpenCVがインストールされているか確認: `pip install opencv-python`

### 実況が保存されない

- `COMPLEMENTARY_DATA` フィールドがデータセットに含まれているか確認
- 検証スクリプトでデータセットを確認

### 'i' キーでエピソードが中断できない

- 1秒間キーを長押ししているか確認
- OpenCVウィンドウがアクティブか確認

## カスタマイズ

### 他のロボットを使用する場合

```python
# 例: SO100を使用する場合
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

robot_config = SO100FollowerConfig(port=FOLLOWER_PORT, id="so100_follower")
teleop_config = SO100LeaderConfig(port=LEADER_PORT, id="so100_leader")
```

### 画面レイアウトをカスタマイズする場合

`create_display_layout()` 関数を編集してください:

```python
def create_display_layout(camera_images: dict[str, np.ndarray], status_text: list[str]) -> np.ndarray:
    # カスタマイズ可能
    # - カメラ配置（グリッド、縦連結など）
    # - フォントサイズ、色
    # - 追加情報の表示
    ...
```

## ライセンス

Apache License 2.0

## 関連リンク

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [SN-VLA Policy Configuration](../../src/lerobot/policies/snvla/)
- [Dataset Documentation](../../docs/source/lerobot-dataset-v3.mdx)
