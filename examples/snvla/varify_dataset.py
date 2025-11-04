#!/usr/bin/env python

"""SNVLA データセットの検証スクリプト

このスクリプトは、SNVLAデータセットが正しく作成されているかを検証します。
必要なフィールド（task, current_narration, previous_narrations）の存在を確認し、
データセットの統計情報を表示します。

使用例:
```shell
python examples/snvla/varify_dataset.py your-username/snvla_so101_pickplace
```
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import COMPLEMENTARY_DATA, CURRENT_NARRATION, PREVIOUS_NARRATIONS


def verify_snvla_dataset(repo_id: str):
    """データセットの特徴量を確認"""
    print(f"\n{'=' * 70}")
    print(f"Verifying SN-VLA Dataset: {repo_id}")
    print(f"{'=' * 70}\n")

    dataset = LeRobotDataset(repo_id)

    print("Dataset Metadata:")
    print(f"  Total episodes: {dataset.meta.total_episodes}")
    print(f"  Total frames: {dataset.meta.total_frames}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Robot type: {dataset.meta.robot_type}")
    print("\nDataset Features:")
    for feature_name in sorted(dataset.meta.features.keys()):
        feature = dataset.meta.features[feature_name]
        print(f"  - {feature_name}: {feature}")

    # 最初のフレームを確認
    print(f"\n{'=' * 70}")
    print("First Frame Analysis")
    print(f"{'=' * 70}\n")

    frame = dataset[0]

    # complementary_dataの確認
    if COMPLEMENTARY_DATA in frame:
        comp_data = frame[COMPLEMENTARY_DATA]
        print(f"Task: {comp_data.get('task', 'N/A')}")
        print(f"Current Narration: '{comp_data.get(CURRENT_NARRATION, 'N/A')}'")
        prev_narr = comp_data.get(PREVIOUS_NARRATIONS, [])
        print(f"Previous Narrations: {prev_narr}")
    else:
        print("WARNING: complementary_data not found in frame!")

    # アクションとステートの次元確認
    if "action" in frame:
        print(f"\nAction shape: {frame['action'].shape}")
    if "observation.state" in frame:
        print(f"State shape: {frame['observation.state'].shape}")

    # 画像の確認
    image_keys = [k for k in frame if "image" in k.lower() or "camera" in k.lower()]
    if image_keys:
        print(f"\nImage observations: {len(image_keys)}")
        for img_key in image_keys:
            print(f"  - {img_key}: {frame[img_key].shape}")

    # ナレーションの統計
    print(f"\n{'=' * 70}")
    print("Narration Statistics")
    print(f"{'=' * 70}\n")

    all_narrations = []
    narration_frames = 0
    action_frames = 0

    for idx in range(len(dataset)):
        frame = dataset[idx]
        if COMPLEMENTARY_DATA in frame:
            comp_data = frame[COMPLEMENTARY_DATA]
            curr_narr = comp_data.get(CURRENT_NARRATION, "")
            if curr_narr:
                all_narrations.append(curr_narr)
                narration_frames += 1
            else:
                action_frames += 1

    print(f"Total narrations: {len(all_narrations)}")
    print(f"Frames with narration: {narration_frames}")
    print(f"Frames with action only: {action_frames}")

    if all_narrations:
        print("\nSample narrations (first 5):")
        for i, narr in enumerate(all_narrations[:5], 1):
            print(f"  {i}. {narr}")

    # エピソードごとの統計
    print(f"\n{'=' * 70}")
    print("Episode Statistics")
    print(f"{'=' * 70}\n")

    episode_info = dataset.meta.episode_data_index
    print(f"Episodes: {len(episode_info)}")
    for ep_idx, (start, end) in enumerate(episode_info.items()):
        frame_count = end - start
        print(f"  Episode {ep_idx}: {frame_count} frames (index {start} to {end - 1})")

    print(f"\n{'=' * 70}")
    print("Verification Complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python examples/snvla/varify_dataset.py <repo_id>")
        print("Example: python examples/snvla/varify_dataset.py your-username/snvla_so101_pickplace")
        sys.exit(1)

    repo_id = sys.argv[1]
    verify_snvla_dataset(repo_id)
