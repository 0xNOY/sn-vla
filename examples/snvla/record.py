#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SN-VLA用のデータセット記録スクリプト

このスクリプトは、SN-VLA (Self-Narrating Vision-Language-Action) モデルの学習に
必要なデータセットを収集します。テレオペレーション中に実況文を挿入できます。

使用例:
```shell
python examples/snvla/record.py
```

キー操作:
- Enter: 実況リストの先頭をpopして挿入（リストが空の場合はエピソード終了）
- i (1秒間長押し): エピソードを中断・削除
- r: エピソード再録画
- q: 記録停止
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import (
    ACTION,
    COMPLEMENTARY_DATA,
    CURRENT_NARRATION,
    OBS_STR,
    PREVIOUS_NARRATIONS,
)

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
# この順番で実況が挿入されます。空になった状態でEnterを押すとエピソード終了
NARRATION_LIST = [
    "Approaching the object with the gripper",
    "Grasping the object firmly",
    "Lifting the object upward",
    "Moving toward the target position",
    "Lowering the object carefully",
    "Releasing the object at the target",
]

# ======================== ユーティリティ関数 ========================


def create_display_layout(camera_images: dict[str, np.ndarray], status_text: list[str]) -> np.ndarray:
    """
    カメラ画像とステータステキストを見やすく合成した画像を作成

    Args:
        camera_images: カメラ名をキー、画像(numpy array)を値とする辞書
        status_text: 表示するステータスメッセージのリスト

    Returns:
        合成された表示用画像
    """
    if not camera_images:
        # カメラがない場合は黒画面にテキストのみ
        display_img = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        # カメラ画像を水平に連結
        images = list(camera_images.values())
        # すべての画像を同じ高さにリサイズ
        target_height = 360
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            new_w = int(w * target_height / h)
            resized = cv2.resize(img, (new_w, target_height))
            resized_images.append(resized)

        # 水平連結
        display_img = resized_images[0] if len(resized_images) == 1 else np.hstack(resized_images)

    # ステータステキスト用のスペースを下部に追加
    text_height = 30 * len(status_text) + 20
    h, w = display_img.shape[:2]
    text_area = np.zeros((text_height, w, 3), dtype=np.uint8)

    # テキストを描画
    y_offset = 25
    for line in status_text:
        cv2.putText(
            text_area,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 30

    # 画像とテキストエリアを結合
    final_img = np.vstack([display_img, text_area])
    return final_img


class SNVLARecorder:
    """SN-VLA用データセット記録クラス"""

    def __init__(
        self,
        robot: SO101Follower,
        teleop: SO101Leader,
        dataset: LeRobotDataset,
        narration_list: list[str],
        task_description: str,
        fps: int,
        control_time_s: float,
    ):
        self.robot = robot
        self.teleop = teleop
        self.dataset = dataset
        self.task_description = task_description
        self.fps = fps
        self.control_time_s = control_time_s

        # 実況リスト（エピソードごとにリセット）
        self.narration_queue = deque(narration_list.copy())
        self.previous_narrations = []

        # キー押下状態の追跡
        self.interrupt_key_pressed_time = None
        self.interrupt_threshold = 1.0  # 'i'を1秒間押し続けると中断

        # フレームごとの実況文を保存
        self.current_narration_for_frame = ""

        # エピソード制御フラグ
        self.should_interrupt = False
        self.should_end_episode = False

        # プロセッサ
        (
            self.teleop_action_processor,
            self.robot_action_processor,
            self.robot_observation_processor,
        ) = make_default_processors()

    def reset_episode(self, narration_list: list[str]):
        """エピソード開始時のリセット"""
        self.narration_queue = deque(narration_list.copy())
        self.previous_narrations = []
        self.current_narration_for_frame = ""
        self.interrupt_key_pressed_time = None
        self.should_interrupt = False
        self.should_end_episode = False

    def handle_key_input(self, key: int) -> bool:
        """
        キー入力の処理

        Returns:
            Trueの場合、録画を継続。Falseの場合、ループを終了
        """
        if key == ord("q"):
            # 記録停止
            return False

        elif key == ord("\r") or key == ord("\n"):  # Enter
            # 実況の挿入またはエピソード終了
            if self.narration_queue:
                # 実況を挿入
                new_narration = self.narration_queue.popleft()
                self.previous_narrations.append(new_narration)
                self.current_narration_for_frame = new_narration
                print(f"[NARRATION] Inserted: {new_narration}")
            else:
                # 実況リストが空→エピソード終了
                print("[INFO] Narration list empty, ending episode...")
                self.should_end_episode = True
                return False

        elif key == ord("i"):
            # 中断キーの長押し検出
            if self.interrupt_key_pressed_time is None:
                self.interrupt_key_pressed_time = time.perf_counter()
            elif time.perf_counter() - self.interrupt_key_pressed_time >= self.interrupt_threshold:
                print("[INTERRUPT] Episode interrupted and will be deleted.")
                self.should_interrupt = True
                return False

        else:
            # 他のキーが押された場合、中断タイマーをリセット
            self.interrupt_key_pressed_time = None

        return True

    def get_status_text(self, timestamp: float) -> list[str]:
        """ステータステキストを生成"""
        remaining_time = max(0, self.control_time_s - timestamp)
        status_lines = [
            f"Episode Time: {timestamp:.1f}s / {self.control_time_s:.1f}s (Remaining: {remaining_time:.1f}s)",
            f"Task: {self.task_description}",
            f"Remaining Narrations: {len(self.narration_queue)}",
        ]

        if self.narration_queue:
            status_lines.append(f"Next Narration: {self.narration_queue[0]}")
        else:
            status_lines.append("Next Narration: [Press Enter to END EPISODE]")

        if self.previous_narrations:
            status_lines.append(f"Previous: {' | '.join(self.previous_narrations[-3:])}")

        status_lines.append("")
        status_lines.append("[Enter] Insert Narration / End Episode  [i-hold] Interrupt  [q] Quit")

        return status_lines

    def record_episode(self) -> bool:
        """
        1エピソードを記録

        Returns:
            True: エピソード正常終了, False: 中断または停止
        """
        timestamp = 0
        start_time = time.perf_counter()
        dt = 1.0 / self.fps

        print(f"\n{'=' * 60}")
        print(f"Recording Episode (Task: {self.task_description})")
        print(f"{'=' * 60}")

        while timestamp < self.control_time_s:
            loop_start = time.perf_counter()

            # ロボットの観測を取得
            obs = self.robot.get_observation()
            obs_processed = self.robot_observation_processor(obs)

            # テレオペレータからアクションを取得
            action = self.teleop.get_action()
            action_processed = self.teleop_action_processor((action, obs))

            # ロボットにアクションを送信
            robot_action = self.robot_action_processor((action_processed, obs))
            self.robot.send_action(robot_action)

            # データセットに保存するための追加データ
            complementary_data = {
                "task": self.task_description,
                CURRENT_NARRATION: self.current_narration_for_frame,
                PREVIOUS_NARRATIONS: self.previous_narrations.copy(),
            }

            # フレームをバッファに追加
            frame_data = {
                "observation": obs_processed,
                "action": action_processed,
                COMPLEMENTARY_DATA: complementary_data,
            }
            self.dataset.add_frame(frame_data)

            # 次のフレームのために実況をクリア（挿入されたフレームのみに記録）
            self.current_narration_for_frame = ""

            # ビジュアライゼーション
            camera_images = {}
            for cam_key, cam_data in obs_processed.items():
                if "image" in cam_key or "camera" in cam_key.lower():
                    # RGB画像をBGRに変換
                    img = cam_data
                    if isinstance(img, np.ndarray) and len(img.shape) == 3:
                        camera_images[cam_key] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            status_text = self.get_status_text(timestamp)
            display_img = create_display_layout(camera_images, status_text)
            cv2.imshow("SN-VLA Recording", display_img)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # キーが押された
                should_continue = self.handle_key_input(key)
                if not should_continue:
                    break

            # フレームレート制御
            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            timestamp = time.perf_counter() - start_time

        cv2.destroyAllWindows()

        # 中断チェック: Trueの場合は成功、Falseの場合は中断
        return not self.should_interrupt


def main():
    """メイン実行関数"""
    print("\n" + "=" * 60)
    print("SN-VLA Dataset Recording Script")
    print("=" * 60 + "\n")

    # ロボット・テレオペレータの設定
    robot_config = SO101FollowerConfig(port=FOLLOWER_PORT, id="so101_follower")
    teleop_config = SO101LeaderConfig(port=LEADER_PORT, id="so101_leader")

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    # データセット特徴量の設定
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # データセットの作成
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
        root=DATASET_ROOT,
    )

    # ロボットとテレオペレータに接続
    print("Connecting to robot and teleoperator...")
    robot.connect()
    teleop.connect()

    if not robot.is_connected or not teleop.is_connected:
        raise RuntimeError("Failed to connect to robot or teleoperator")

    print("Connected successfully!\n")

    # レコーダーの初期化
    recorder = SNVLARecorder(
        robot=robot,
        teleop=teleop,
        dataset=dataset,
        narration_list=NARRATION_LIST,
        task_description=TASK_DESCRIPTION,
        fps=FPS,
        control_time_s=EPISODE_TIME_SEC,
    )

    recorded_episodes = 0
    stop_recording = False

    try:
        while recorded_episodes < NUM_EPISODES and not stop_recording:
            print(f"\n{'=' * 60}")
            print(f"Starting Episode {recorded_episodes + 1} / {NUM_EPISODES}")
            print(f"{'=' * 60}")

            # エピソードをリセット
            recorder.reset_episode(NARRATION_LIST)

            # エピソードを記録
            success = recorder.record_episode()

            if recorder.should_interrupt:
                # 中断された場合、バッファをクリア
                print("[INFO] Episode interrupted, clearing buffer...")
                dataset.clear_episode_buffer()
                continue

            if not success:
                # ユーザーが停止した場合
                print("[INFO] Recording stopped by user")
                stop_recording = True
                break

            # エピソードを保存
            print("[INFO] Saving episode...")
            dataset.save_episode()
            recorded_episodes += 1
            print(f"[SUCCESS] Episode {recorded_episodes} saved!")

            # リセット期間（次のエピソードの準備）
            if recorded_episodes < NUM_EPISODES:
                print(f"\n[INFO] Reset the environment ({RESET_TIME_SEC} seconds)...")
                time.sleep(RESET_TIME_SEC)

    finally:
        # クリーンアップ
        print("\n[INFO] Cleaning up...")
        robot.disconnect()
        teleop.disconnect()
        cv2.destroyAllWindows()

        # データセットをファイナライズ
        print("[INFO] Finalizing dataset...")
        dataset.finalize()

        # Hugging Face Hubにアップロード
        print(f"[INFO] Pushing dataset to {HF_REPO_ID}...")
        dataset.push_to_hub()

        print("\n" + "=" * 60)
        print(f"Recording Complete! Total episodes: {recorded_episodes}")
        print(f"Dataset: {HF_REPO_ID}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
