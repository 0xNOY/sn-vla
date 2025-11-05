import argparse

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def visualize_episode_with_narrations(dataset, episode_idx=0, camera_key=None):
    """エピソードをナレーション付きで動画として可視化"""
    if camera_key is None:
        camera_key = dataset.meta.camera_keys[0]

    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]

    for idx in range(from_idx, to_idx):
        frame = dataset[idx]

        # 画像を取得（CHW -> HWC変換）
        image = frame[camera_key].numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ナレーション情報をオーバーレイ
        if frame["current_narration"]:
            cv2.putText(
                image,
                f"Current: {frame['current_narration']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if frame["previous_narrations"]:
            y_pos = 60
            for prev in frame["previous_narrations"].split("\n"):
                if prev.strip():
                    cv2.putText(
                        image,
                        f"Previous: {prev}",
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                    y_pos += 25

        cv2.imshow("Episode with Narrations", image)
        if cv2.waitKey(int(1000 / dataset.fps)) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Visualize episodes with narrations")
        parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., username/dataset_name)")
        parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to visualize")
        args = parser.parse_args()

        dataset = LeRobotDataset(args.dataset_name)
        visualize_episode_with_narrations(dataset, episode_idx=args.episode_idx)
