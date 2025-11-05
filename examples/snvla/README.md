# SN-VLA Recording Example

このディレクトリには、**SN-VLA (Self-Narrating Vision-Language-Action)** ポリシーのためのデータセット収集スクリプトが含まれています。

## 概要

`record.py` は、LeRobotの標準的なデータ収集スクリプト (`lerobot_record.py`) を拡張し、実況(narration)機能を追加したものです。

### 主な機能

- **OpenCVによるリアルタイム表示**: ロボットのカメラ視野をリアルタイムで表示
- **複数カメラの合成表示**: 複数のカメラがある場合、見やすいグリッドレイアウトで表示
- **ステータスオーバーレイ**: 記録時間、過去・次の実況文などを画面上に表示
- **実況文の挿入**: 予め定義された実況リストから `Enter` キーで実況を挿入
- **エピソード終了**: 実況リストが空の状態で `Enter` を押すとエピソード終了
- **エピソード中断**: `i` キーを1秒間長押しでエピソードを中断・削除

## 必要なデータ形式

SN-VLAポリシーは、以下の追加データを必要とします：

1. **`complementary_data.current_narration`**: 現在のフレームで挿入された実況文（文字列）
2. **`complementary_data.previous_narrations`**: そのフレームまでの実況履歴（文字列のリストをシリアライズしたもの）

これらは、通常のLeRobotデータセット（観測、アクション、タスク記述など）に加えて記録されます。

## 使用方法

### 基本的な使用例

```bash
python examples/snvla/record.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyUSB1 \
    --dataset.repo_id=username/snvla_cube_pickup \
    --dataset.single_task="Pick up the cube and place it in the box" \
    --dataset.num_episodes=10 \
    --dataset.fps=30
```

### パラメータ説明

#### ロボット設定 (`--robot.*`)
- `type`: ロボットタイプ（例: `so100_follower`, `koch_follower`）
- `port`: ロボットの接続ポート
- `cameras`: カメラ設定（辞書形式）

#### テレオペレータ設定 (`--teleop.*`)
- `type`: テレオペレータタイプ（例: `so100_leader`）
- `port`: テレオペレータの接続ポート

#### データセット設定 (`--dataset.*`)
- `repo_id`: データセットID（例: `username/dataset_name`）
- `single_task`: タスク記述（文字列）
- `num_episodes`: 収集するエピソード数
- `fps`: フレームレート
- `episode_time_s`: 1エピソードの最大記録時間（秒）
- `reset_time_s`: エピソード間のリセット時間（秒）
- `video`: ビデオエンコーディングを有効化（デフォルト: `true`）
- `push_to_hub`: Hugging Face Hubにアップロード（デフォルト: `false`）

#### その他の設定
- `--display_cameras`: カメラ表示を有効化（デフォルト: `true`）
- `--play_sounds`: 音声合成を有効化（デフォルト: `true`）
- `--resume`: 既存データセットに追加記録（デフォルト: `false`）

## 実況リストのカスタマイズ

`record.py` 内の `narration_list` を編集して、タスクに応じた実況文を定義できます：

```python
# 例: キューブをつかんで箱に入れるタスク
narration_list = [
    "Approaching the target object",
    "Grasping the object with gripper",
    "Lifting the object upward",
    "Moving toward the destination",
    "Placing the object down",
    "Releasing the gripper",
    "Returning to rest position",
]
```

将来的には、設定ファイルや外部ファイルから読み込めるように拡張予定です。

## 操作方法

### 記録中のキーボード操作

- **`Enter`**: 実況リストの先頭要素を現在のフレームに挿入
  - 実況リストが空の場合、エピソードを終了
- **`i` (1秒長押し)**: 現在のエピソードを中断し、データを削除

### 画面表示

OpenCVウィンドウには以下が表示されます：

1. **カメラビュー**: ロボットの視野（複数カメラがある場合は合成）
2. **ステータスオーバーレイ**:
   - エピソード番号、フレーム番号、経過時間
   - 前回挿入した実況文
   - 次に挿入する実況文（または `[End Episode]`）
   - 操作方法のヒント

## データセット構造

記録されたデータセットは、LeRobot標準形式に加えて以下を含みます：

```
dataset/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet  # 観測、アクション、実況データ
├── videos/  # (video=true の場合)
│   └── observation.images.{camera_name}/
├── meta_data/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.json
│   └── episodes.jsonl
└── README.md
```

各フレームには以下が含まれます：

- `observation.*`: ロボットの観測データ（カメラ画像、状態など）
- `action`: ロボットのアクション
- `task`: タスク記述
- `complementary_data.current_narration`: 現在のフレームの実況文
- `complementary_data.previous_narrations`: 実況履歴

## トラブルシューティング

### カメラが表示されない

- ロボットの設定で `cameras` が正しく定義されているか確認
- `display_cameras=true` が設定されているか確認
- OpenCVがインストールされているか確認: `pip install opencv-python`

### 実況が記録されない

- `Enter` キーを押したタイミングで実況が挿入されます
- 実況リストが空の場合、エピソードが終了します
- ログ出力を確認: `logging.info` でフレーム毎に実況挿入が記録されます

### エピソードが中断できない

- `i` キーを **1秒間連続で** 押し続ける必要があります
- 短く押しただけでは中断されません

## 関連ファイル

- `src/lerobot/policies/snvla/`: SNVLAポリシーの実装
- `src/lerobot/scripts/lerobot_record.py`: LeRobotの標準記録スクリプト
- `src/lerobot/utils/constants.py`: 定数定義（`CURRENT_NARRATION`, `PREVIOUS_NARRATIONS`など）

## ライセンス

Apache License 2.0
