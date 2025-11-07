#!/usr/bin/env python3
"""
実況データ拡張ツール

既存データセット内の実況フレームを前後のフレームに伝播させることで、
実況データの頻度を増やし、クラス不均衡問題を緩和します。
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import CURRENT_NARRATION, PREVIOUS_NARRATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def find_narration_frames(dataset: LeRobotDataset, episode_idx: int) -> list[dict[str, Any]]:
    """
    エピソード内で実況が存在するフレームを検索

    Args:
        dataset: LeRobotDataset インスタンス
        episode_idx: エピソードインデックス

    Returns:
        実況フレーム情報のリスト [{"frame_idx": int, "narration": str, "previous": str}, ...]
    """
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]

    narration_frames = []

    for idx in range(from_idx, to_idx):
        frame = dataset.hf_dataset[idx]
        current_narration = frame.get(CURRENT_NARRATION, "")

        # Tensorの場合は文字列に変換
        if hasattr(current_narration, "item"):
            try:
                current_narration = str(current_narration.item())
            except Exception:
                current_narration = ""

        # 文字列に変換
        if not isinstance(current_narration, str):
            current_narration = str(current_narration) if current_narration else ""

        narration_clean = current_narration.strip()

        if narration_clean:
            previous_narrations = frame.get(PREVIOUS_NARRATIONS, "")
            if hasattr(previous_narrations, "item"):
                try:
                    previous_narrations = str(previous_narrations.item())
                except Exception:
                    previous_narrations = ""

            if not isinstance(previous_narrations, str):
                previous_narrations = str(previous_narrations) if previous_narrations else ""

            narration_frames.append(
                {
                    "frame_idx": idx,
                    "relative_idx": idx - from_idx,
                    "narration": narration_clean,
                    "previous": previous_narrations.strip(),
                }
            )

    return narration_frames


def augment_episode(
    dataset: LeRobotDataset,
    episode_idx: int,
    window_size: int = 10,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    エピソードの実況データを拡張

    Args:
        dataset: LeRobotDataset インスタンス
        episode_idx: エピソードインデックス
        window_size: 実況を伝播させるフレーム数（前後それぞれ）
        dry_run: True の場合、実際の変更は行わず統計のみ計算

    Returns:
        拡張統計情報
    """
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]
    total_frames = to_idx - from_idx

    # 実況フレームを検索
    narration_frames = find_narration_frames(dataset, episode_idx)
    original_count = len(narration_frames)

    if original_count == 0:
        logging.warning(f"Episode {episode_idx}: No narrations found.")
        return {
            "episode_idx": episode_idx,
            "total_frames": total_frames,
            "original_narrations": 0,
            "augmented_narrations": 0,
            "augmentation_ratio": 0.0,
        }

    # 拡張対象のフレームを計算
    augmented_frames = set()
    for narr_info in narration_frames:
        center_idx = narr_info["relative_idx"]
        # 前後 window_size フレームを対象に
        start = max(0, center_idx - window_size)
        end = min(total_frames, center_idx + window_size + 1)
        for i in range(start, end):
            augmented_frames.add(from_idx + i)

    augmented_count = len(augmented_frames)

    logging.info(f"Episode {episode_idx}:")
    logging.info(f"  Total frames: {total_frames}")
    logging.info(f"  Original narrations: {original_count} ({original_count / total_frames * 100:.2f}%)")
    logging.info(f"  Augmented narrations: {augmented_count} ({augmented_count / total_frames * 100:.2f}%)")
    logging.info(f"  Amplification: {augmented_count / original_count:.1f}x")

    if not dry_run:
        # データの拡張を実行
        _apply_augmentation(dataset, episode_idx, narration_frames, window_size, augmented_frames)

    return {
        "episode_idx": episode_idx,
        "total_frames": total_frames,
        "original_narrations": original_count,
        "augmented_narrations": augmented_count,
        "augmentation_ratio": augmented_count / total_frames,
    }


def _apply_augmentation(
    dataset: LeRobotDataset,
    episode_idx: int,
    narration_frames: list[dict],
    window_size: int,
    augmented_frames: set[int],
) -> None:
    """
    実際にデータを拡張（Parquetファイルを更新）

    Args:
        dataset: LeRobotDataset インスタンス
        episode_idx: エピソードインデックス
        narration_frames: 実況フレーム情報
        window_size: 伝播ウィンドウサイズ
        augmented_frames: 拡張対象のフレームインデックスセット
    """
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]

    # エピソードのデータファイルを特定
    chunk_idx = dataset.meta.episodes[episode_idx]["data/chunk_index"]
    file_idx = dataset.meta.episodes[episode_idx]["data/file_index"]
    data_path = dataset.root / dataset.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)

    logging.info(f"  Updating parquet file: {data_path}")

    # Parquetファイルを読み込み
    table = pq.read_table(data_path)
    df = table.to_pandas()

    # 各フレームに対して最も近い実況を適用
    for idx in range(from_idx, to_idx):
        if idx not in augmented_frames:
            continue

        # このフレームに最も近い実況フレームを見つける
        closest_narr = None
        min_distance = float("inf")

        for narr_info in narration_frames:
            distance = abs(idx - narr_info["frame_idx"])
            if distance <= window_size and distance < min_distance:
                min_distance = distance
                closest_narr = narr_info

        if closest_narr:
            # DataFrameを更新
            mask = df["index"] == idx
            df.loc[mask, CURRENT_NARRATION] = closest_narr["narration"]
            df.loc[mask, PREVIOUS_NARRATIONS] = closest_narr["previous"]

    # Parquetファイルに書き戻し
    # 元のファイルをバックアップ
    backup_path = data_path.with_suffix(".parquet.bak")
    shutil.copy2(data_path, backup_path)

    try:
        # 新しいデータを書き込み
        df.to_parquet(data_path, engine="pyarrow", compression="snappy", index=False)
        logging.info(f"  Successfully updated {data_path}")
        # バックアップを削除
        backup_path.unlink()
    except Exception as e:
        logging.error(f"  Failed to update {data_path}: {e}")
        # エラーが発生した場合はバックアップから復元
        shutil.copy2(backup_path, data_path)
        backup_path.unlink()
        raise


def augment_dataset(
    input_repo: str,
    output_path: Path | None = None,
    window_size: int = 10,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    データセット全体の実況データを拡張

    Args:
        input_repo: 入力データセットのHuggingFace repo ID
        output_path: 出力先ローカルパス（指定時はコピーを作成）
        window_size: 実況を伝播させるフレーム数
        dry_run: True の場合、実際の変更は行わず統計のみ計算

    Returns:
        全体の統計情報
    """
    logging.info(f"Loading dataset: {input_repo}")
    dataset = LeRobotDataset(input_repo)

    # 出力先の準備
    if output_path:
        output_path = Path(output_path)
        if output_path.exists():
            logging.error(f"Output path already exists: {output_path}")
            raise FileExistsError(f"Output path already exists: {output_path}")

        if not dry_run:
            logging.info(f"Copying dataset to: {output_path}")
            shutil.copytree(dataset.root, output_path)
            # コピー先を使用
            dataset = LeRobotDataset(input_repo, root=output_path)

    # 全エピソードを処理
    num_episodes = dataset.meta.total_episodes
    episode_stats = []
    total_original = 0
    total_augmented = 0
    total_frames_count = 0

    logging.info(f"\n{'=' * 80}")
    logging.info(f"Augmenting {num_episodes} episodes with window_size={window_size}")
    logging.info(f"{'=' * 80}\n")

    for ep_idx in range(num_episodes):
        ep_stat = augment_episode(dataset, ep_idx, window_size, dry_run)
        episode_stats.append(ep_stat)
        total_original += ep_stat["original_narrations"]
        total_augmented += ep_stat["augmented_narrations"]
        total_frames_count += ep_stat["total_frames"]

    # 統計サマリー
    stats = {
        "num_episodes": num_episodes,
        "total_frames": total_frames_count,
        "original_narration_count": total_original,
        "augmented_narration_count": total_augmented,
        "original_ratio": total_original / total_frames_count if total_frames_count > 0 else 0,
        "augmented_ratio": total_augmented / total_frames_count if total_frames_count > 0 else 0,
        "amplification_factor": total_augmented / total_original if total_original > 0 else 0,
        "episode_stats": episode_stats,
    }

    logging.info(f"\n{'=' * 80}")
    logging.info("Augmentation Summary")
    logging.info(f"{'=' * 80}")
    logging.info(f"Total frames: {total_frames_count}")
    logging.info(f"Original narrations: {total_original} ({total_original / total_frames_count * 100:.2f}%)")
    logging.info(
        f"Augmented narrations: {total_augmented} ({total_augmented / total_frames_count * 100:.2f}%)"
    )
    logging.info(f"Amplification factor: {stats['amplification_factor']:.1f}x")
    logging.info(f"{'=' * 80}\n")

    if dry_run:
        logging.info("DRY RUN: No changes were made to the dataset.")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="実況データ拡張ツール - 実況フレームを前後に伝播させてデータを拡張"
    )
    parser.add_argument(
        "--input-repo",
        type=str,
        required=True,
        help="入力データセットのHuggingFace repo ID (例: username/dataset-name)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="出力先ローカルパス（指定しない場合は入力データセットを直接更新）",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default=None,
        help="出力先HuggingFace repo ID（--push-to-hub使用時に必要）",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="実況を伝播させるフレーム数（前後それぞれ、デフォルト: 10）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="変更を実行せず統計のみ表示",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="HuggingFace Hubにプッシュ",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="outputs/augmentation_stats.json",
        help="統計情報の保存先（デフォルト: outputs/augmentation_stats.json）",
    )

    args = parser.parse_args()

    # 引数の検証
    if args.push_to_hub and not args.output_repo:
        parser.error("--push-to-hub requires --output-repo")

    # データセットの拡張
    output_path = Path(args.output_path) if args.output_path else None
    stats = augment_dataset(
        input_repo=args.input_repo,
        output_path=output_path,
        window_size=args.window_size,
        dry_run=args.dry_run,
    )

    # 統計情報を保存
    stats_path = Path(args.stats_output)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Statistics saved to: {stats_path}")

    # HuggingFace Hubにプッシュ
    if args.push_to_hub and not args.dry_run:
        if output_path is None:
            logging.error("Cannot push to hub without --output-path")
            return

        logging.info(f"\n{'=' * 80}")
        logging.info(f"Pushing to HuggingFace Hub: {args.output_repo}")
        logging.info(f"{'=' * 80}")

        api = HfApi()
        api.create_repo(repo_id=args.output_repo, repo_type="dataset", exist_ok=True)

        # データセット全体をアップロード
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=args.output_repo,
            repo_type="dataset",
        )

        # バージョンタグを作成（info.jsonから_version_を読み取る）
        try:
            info_path = output_path / "meta" / "info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                    version = info.get("codebase_version", "v2.0")

                logging.info(f"Creating version tag: {version}")
                api.create_tag(
                    repo_id=args.output_repo,
                    tag=version,
                    repo_type="dataset",
                    tag_message=f"Dataset version {version} - augmented with window_size={args.window_size}",
                )
                logging.info(f"Version tag '{version}' created successfully")
            else:
                logging.warning(f"info.json not found at {info_path}, skipping tag creation")
        except Exception as e:
            logging.warning(f"Failed to create version tag: {e}")

        logging.info(f"Successfully pushed to: https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
