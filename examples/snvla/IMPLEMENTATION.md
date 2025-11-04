# SN-VLA データ収集スクリプト実装の概要

## 実装完了項目 ✅

### 1. メインスクリプト (`record.py`)

**主要機能:**
- ✅ OpenCV `imshow` でリアルタイムビジュアライゼーション
- ✅ 複数カメラ画像の見やすい合成表示（水平連結、自動リサイズ）
- ✅ ステータス情報のオーバーレイ表示
  - エピソード経過時間と残り時間
  - タスク説明
  - 過去・次の実況文
  - キー操作ガイド
- ✅ 実況リストからの自動挿入
- ✅ キー入力ハンドリング
  - **Enter**: 実況挿入 / エピソード終了
  - **i (1秒長押し)**: エピソード中断・削除
  - **q**: 記録停止

**データセット形式対応:**
- ✅ `complementary_data.task`: タスク説明
- ✅ `complementary_data.current_narration`: 現在フレームの実況文
- ✅ `complementary_data.previous_narrations`: 過去の実況文リスト
- ✅ 標準的なLeRobotデータセット形式との互換性

**クラス設計:**
```
SNVLARecorder
├── reset_episode()          # エピソードの初期化
├── handle_key_input()       # キー入力処理
├── get_status_text()        # ステータステキスト生成
└── record_episode()         # メインループ
```

### 2. 検証スクリプト (`varify_dataset.py`)

**主要機能:**
- ✅ データセットメタデータの表示
- ✅ フィールド存在確認
- ✅ 実況文統計（件数、サンプル表示）
- ✅ エピソード情報の表示
- ✅ 画像・アクション・ステートの次元確認

### 3. ドキュメント

- ✅ **README.md**: 詳細な使用方法ガイド
- ✅ **setup.sh**: クイックスタートスクリプト
- ✅ **IMPLEMENTATION.md**: この実装概要（本ファイル）

---

## アーキテクチャの詳細

### データフロー

```
Teleop → Robot → Observation
                     ↓
              [Processor Pipeline]
                     ↓
         ┌───────────┴───────────┐
         │                       │
    [Action Data]        [Complementary Data]
         │                       │
         │                  - task
         │                  - current_narration
         │                  - previous_narrations
         │                       │
         └───────────┬───────────┘
                     ↓
            [LeRobot Dataset]
                     ↓
              [Hugging Face Hub]
```

### ビジュアライゼーション設計

```
┌─────────────────────────────────────────────┐
│  Camera Images (Horizontal Concatenation)   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │Camera 1 │ │Camera 2 │ │Camera 3 │       │
│  └─────────┘ └─────────┘ └─────────┘       │
├─────────────────────────────────────────────┤
│  Status Information                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  • Episode Time: 15.3s / 60.0s (44.7s left) │
│  • Task: Pick and place object              │
│  • Remaining Narrations: 4                  │
│  • Next: "Lifting the object upward"        │
│  • Previous: Approaching | Grasping         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  [Enter] Insert  [i-hold] Interrupt  [q] Quit│
└─────────────────────────────────────────────┘
```

### 実況挿入のロジック

```python
# エピソード開始時
narration_queue = deque(NARRATION_LIST)
previous_narrations = []
current_narration_for_frame = ""

# Enter押下時
if narration_queue:
    new_narration = narration_queue.popleft()
    previous_narrations.append(new_narration)
    current_narration_for_frame = new_narration
else:
    # 実況リストが空 → エピソード終了
    end_episode()

# 各フレーム保存時
complementary_data = {
    "task": TASK_DESCRIPTION,
    "current_narration": current_narration_for_frame,  # 実況挿入されたフレームのみ値あり
    "previous_narrations": previous_narrations.copy(),
}

# 次フレームのためにクリア
current_narration_for_frame = ""
```

---

## SNVLAデータセット形式の要件

### 必須フィールド

#### 1. Observation
- **画像**: `observation.images.{camera_name}` (RGB, shape: H×W×3)
- **状態**: `observation.state` (ロボット状態ベクトル)

#### 2. Action
- **アクション**: `action` (ロボットアクションベクトル)

#### 3. Complementary Data
- **task** (str): タスクの説明
- **current_narration** (str): 現在フレームの実況文（空文字列 = アクション生成モード）
- **previous_narrations** (list[str]): それまでの実況文リスト

### データセット構造例

```python
frame = {
    "observation.images.top": np.ndarray[480, 640, 3],
    "observation.state": np.ndarray[7],  # joint positions
    "action": np.ndarray[7],  # joint velocities
    "complementary_data": {
        "task": "Pick up the object and place it in the target position",
        "current_narration": "Grasping the object firmly",  # このフレームのみ
        "previous_narrations": ["Approaching the object with the gripper"],
    }
}
```

---

## 使用例

### 基本的な使い方

```bash
# 1. セットアップ
./examples/snvla/setup.sh

# 2. record.pyの設定を編集
# - HF_REPO_ID
# - TASK_DESCRIPTION
# - NARRATION_LIST
# - ロボットポート

# 3. データ収集
python examples/snvla/record.py

# 4. データセット検証
python examples/snvla/varify_dataset.py your-username/your-dataset
```

### カスタマイズ例

#### 実況リストのカスタマイズ
```python
NARRATION_LIST = [
    "Moving to home position",
    "Scanning for target object",
    "Approaching the target",
    "Executing grasp",
    "Confirming grasp success",
    "Transporting to goal location",
    "Placing object carefully",
    "Retracting arm",
]
```

#### 他のロボットへの対応
```python
# Koch robotの場合
from lerobot.robots.koch_follower import KochFollower, KochFollowerConfig
from lerobot.teleoperators.koch_leader import KochLeader, KochLeaderConfig

robot_config = KochFollowerConfig(port=FOLLOWER_PORT, id="koch_follower")
teleop_config = KochLeaderConfig(port=LEADER_PORT, id="koch_leader")
```

---

## トラブルシューティングガイド

### よくある問題と解決方法

#### 問題1: カメラ画像が表示されない
**原因**: OpenCVが正しくインストールされていない
**解決**:
```bash
pip install opencv-python
```

#### 問題2: 実況が保存されない
**原因**: `COMPLEMENTARY_DATA`フィールドがデータセットに追加されていない
**解決**: `verify_dataset.py`で確認し、`record.py`の`complementary_data`ロジックを確認

#### 問題3: フレームレートが不安定
**原因**: 画像処理が重い、またはディスクI/Oが遅い
**解決**:
- カメラ解像度を下げる
- `image_writer_threads`を調整
- SSDを使用

#### 問題4: 'i'キーで中断できない
**原因**: 1秒間押し続けていない
**解決**: `self.interrupt_threshold`の値を調整（デフォルト: 1.0秒）

---

## 実装の技術的特徴

### 1. リアルタイムビジュアライゼーション
- OpenCV `imshow`による低レイテンシ表示
- 複数カメラの自動レイアウト調整
- ステータス情報のオーバーレイ

### 2. 柔軟なキー入力ハンドリング
- 非ブロッキング入力処理
- 長押し検出（中断機能）
- 入力状態の適切な管理

### 3. データ整合性の保証
- フレームごとの実況状態管理
- エピソード中断時のバッファクリア
- 実況リストと過去実況の同期

### 4. 拡張性
- 他のロボットへの容易な適応
- カスタム実況リストのサポート
- レイアウトのカスタマイズ可能

---

## 今後の拡張可能性

### 考えられる機能追加
- [ ] リアルタイム実況入力（音声認識）
- [ ] 実況のスキップ機能
- [ ] 実況の編集・差し替え
- [ ] 複数タスクのサポート
- [ ] データ拡張（画像変換など）
- [ ] ビデオのリアルタイムエンコーディング

### 他のロボットへの対応
- [x] SO101
- [ ] SO100
- [ ] Koch
- [ ] LeKiwi
- [ ] Hope Jr.
- [ ] Reachy 2

---

## 参考資料

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [SN-VLA Configuration](../../src/lerobot/policies/snvla/configuration_snvla.py)
- [SN-VLA Processor](../../src/lerobot/policies/snvla/processor_snvla.py)
- [Dataset Format v3](../../docs/source/lerobot-dataset-v3.mdx)

---

## ライセンス

Apache License 2.0

---

**実装完了日**: 2025-01-04
**バージョン**: 1.0.0
**作成者**: GitHub Copilot
