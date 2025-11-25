#!/usr/bin/env python
"""
Generate partial scoop episodes from 5-scoop episodes.

This script creates new episodes with 1/2/3/4 scoops from existing 5-scoop episodes
by cutting the original episodes at the appropriate narration points.

Usage:
    # Create a new augmented dataset
    uv run python examples/snvla/generate_partial_scoop_episodes.py \
        --source-repo 0xNOY/so101-with-narration \
        --output-repo 0xNOY/so101-with-narration-augmented \
        [--dry-run]

    # Add episodes to an existing dataset (same source and output repo)
    uv run python examples/snvla/generate_partial_scoop_episodes.py \
        --source-repo 0xNOY/so101-with-narration \
        --output-repo 0xNOY/so101-with-narration \
        [--push-to-hub] [--dry-run]
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    flatten_dict,
    load_episodes,
    write_info,
    write_stats,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

# Constants
SOURCE_TASK = "Put 5 scoops of soybeans into the black bowl."
TASK_TEMPLATE = "Put {n} scoops of soybeans into the black bowl."
PUT_INTO_BOWL_NARRATION = " (done)\nPut into the black bowl."
MAIN_TASK_DONE_NARRATION = " (done)\nMain task done.\n"
SCOOP_NARRATION = " (done)\nScoop the soybeans."

# Metadata keys for tracking augmented episodes
META_SOURCE_EPISODE_INDEX = "augmentation/source_episode_index"
META_SCOOP_COUNT = "augmentation/scoop_count"


def find_5_scoop_episodes(dataset: LeRobotDataset) -> list[int]:
    """Find all episode indices with the 5-scoop task."""
    episodes = []
    for ep_idx in range(dataset.num_episodes):
        tasks = dataset.meta.episodes[ep_idx].get("tasks", [])
        if tasks and SOURCE_TASK in tasks:
            episodes.append(ep_idx)
    return episodes


def find_existing_augmented_episodes(dataset: LeRobotDataset) -> set[tuple[int, int]]:
    """
    Find existing augmented episodes by checking metadata.

    Returns:
        Set of (source_episode_index, scoop_count) tuples for existing augmented episodes.
    """
    existing = set()
    for ep_idx in range(dataset.num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        source_ep = ep_meta.get(META_SOURCE_EPISODE_INDEX)
        scoop_count = ep_meta.get(META_SCOOP_COUNT)
        if source_ep is not None and scoop_count is not None:
            existing.add((int(source_ep), int(scoop_count)))
    return existing


def filter_generation_plan(
    generation_plan: list[dict],
    existing_augmented: set[tuple[int, int]],
) -> list[dict]:
    """
    Filter out already generated episodes from the generation plan.

    Args:
        generation_plan: List of planned episodes to generate
        existing_augmented: Set of (source_episode_index, scoop_count) tuples

    Returns:
        Filtered generation plan with only new episodes
    """
    filtered = []
    skipped = 0
    for plan in generation_plan:
        key = (plan["src_episode"], plan["scoop_count"])
        if key not in existing_augmented:
            filtered.append(plan)
        else:
            skipped += 1

    if skipped > 0:
        logger.info(f"Skipping {skipped} episodes that already exist in the dataset")

    return filtered


def find_cut_points(dataset: LeRobotDataset, episode_idx: int) -> dict[int, int]:
    """
    Find frame indices where each scoop is completed.

    The narration structure is:
    - Pick up the spoon. -> (empty) -> Scoop the soybeans. -> (empty) -> Put into the black bowl. (1st done)
    - (empty) -> Scoop the soybeans. -> (empty) -> Put into the black bowl. (2nd done)
    - ...
    - Main task done.

    For N scoops, we cut at the (N+1)th "Scoop the soybeans." narration.
    The last frame's "Scoop" narration will be replaced with "Main task done."

    Returns:
        dict mapping scoop_count (1-4) to the frame index where to cut (exclusive).
    """
    ep_meta = dataset.meta.episodes[episode_idx]
    from_idx = ep_meta["dataset_from_index"]
    to_idx = ep_meta["dataset_to_index"]

    batch = dataset.hf_dataset[from_idx:to_idx]
    narrations = batch["current_narration"]

    cut_points = {}
    scoop_count = 0
    prev_narration = None

    for i, narration in enumerate(narrations):
        if narration != prev_narration and narration == SCOOP_NARRATION:
            scoop_count += 1
            # The N-th "Scoop" narration marks the cut point for (N-1) scoops
            # e.g., 2nd "Scoop" = cut point for 1 scoop
            if scoop_count >= 2 and scoop_count <= 5:
                # Include this frame in the cut (exclusive means cut at i+1)
                cut_points[scoop_count - 1] = i + 1
        prev_narration = narration

    return cut_points


def generate_partial_episode(
    src_dataset: LeRobotDataset,
    src_episode_idx: int,
    scoop_count: int,
    cut_frame: int,
) -> pd.DataFrame:
    """
    Generate a partial episode by cutting at the specified frame.

    Args:
        src_dataset: Source dataset
        src_episode_idx: Source episode index
        scoop_count: Number of scoops (1-4)
        cut_frame: Frame index to cut at (exclusive, relative to episode start)

    Returns:
        DataFrame with episode data
    """
    ep_meta = src_dataset.meta.episodes[src_episode_idx]
    from_idx = ep_meta["dataset_from_index"]

    # Get the data for this partial episode
    batch = src_dataset.hf_dataset[from_idx : from_idx + cut_frame]

    # Convert to DataFrame
    df = pd.DataFrame(batch)

    # Update narrations: replace the LAST "Scoop the soybeans." with "Main task done"
    narrations = df["current_narration"].tolist()

    # Find the last occurrence of SCOOP_NARRATION and replace it
    last_scoop_idx = None
    for i in range(len(narrations) - 1, -1, -1):
        if narrations[i] == SCOOP_NARRATION:
            last_scoop_idx = i
            break

    if last_scoop_idx is not None:
        narrations[last_scoop_idx] = MAIN_TASK_DONE_NARRATION

    df["current_narration"] = narrations

    return df


def create_augmented_dataset(
    src_dataset: LeRobotDataset,
    output_repo_id: str,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> LeRobotDataset | None:
    """
    Create an augmented dataset with partial scoop episodes.

    Args:
        src_dataset: Source dataset with 5-scoop episodes
        output_repo_id: Repository ID for the output dataset
        output_dir: Output directory (defaults to HF cache)
        dry_run: If True, only print what would be done without creating the dataset

    Returns:
        The new dataset, or None if dry_run
    """
    if output_dir is None:
        output_dir = HF_LEROBOT_HOME / output_repo_id

    # Find 5-scoop episodes
    five_scoop_episodes = find_5_scoop_episodes(src_dataset)
    logger.info(f"Found {len(five_scoop_episodes)} episodes with 5-scoop task")

    if not five_scoop_episodes:
        raise ValueError("No 5-scoop episodes found in source dataset")

    # Plan the generation
    generation_plan = []
    for ep_idx in five_scoop_episodes:
        cut_points = find_cut_points(src_dataset, ep_idx)
        for scoop_count in [1, 2, 3, 4]:
            if scoop_count in cut_points:
                generation_plan.append(
                    {
                        "src_episode": ep_idx,
                        "scoop_count": scoop_count,
                        "cut_frame": cut_points[scoop_count],
                        "task": TASK_TEMPLATE.format(n=scoop_count),
                    }
                )

    logger.info(f"Will generate {len(generation_plan)} new episodes:")
    for scoop_count in [1, 2, 3, 4]:
        count = sum(1 for p in generation_plan if p["scoop_count"] == scoop_count)
        logger.info(f"  - {count} episodes for {scoop_count} scoop(s)")

    if dry_run:
        print("\n=== Dry Run - Generation Plan ===")
        print(f"Source dataset: {src_dataset.repo_id}")
        print(f"Output dataset: {output_repo_id}")
        print(f"Output directory: {output_dir}")
        print(
            f"\nWill generate {len(generation_plan)} episodes from {len(five_scoop_episodes)} source episodes:"
        )

        for plan in generation_plan[:10]:  # Show first 10
            print(
                f"  Episode {plan['src_episode']} -> {plan['scoop_count']} scoop(s) (cut at frame {plan['cut_frame']})"
            )
        if len(generation_plan) > 10:
            print(f"  ... and {len(generation_plan) - 10} more")

        return None

    # Check if output directory exists
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists, removing...")
        shutil.rmtree(output_dir)

    # Create new dataset metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=src_dataset.meta.fps,
        features=src_dataset.meta.features,
        robot_type=src_dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(src_dataset.meta.video_keys) > 0,
    )

    # Save tasks
    all_tasks = [TASK_TEMPLATE.format(n=n) for n in [1, 2, 3, 4]]
    new_meta.save_episode_tasks(all_tasks)

    # Process each episode
    all_episode_stats = []
    global_frame_idx = 0
    chunk_idx, file_idx = 0, 0

    for new_ep_idx, plan in enumerate(tqdm(generation_plan, desc="Generating episodes")):
        src_ep_idx = plan["src_episode"]
        scoop_count = plan["scoop_count"]
        cut_frame = plan["cut_frame"]
        task = plan["task"]

        # Generate the partial episode data
        df = generate_partial_episode(src_dataset, src_ep_idx, scoop_count, cut_frame)

        ep_length = len(df)

        # Update indices
        df["index"] = range(global_frame_idx, global_frame_idx + ep_length)
        df["episode_index"] = new_ep_idx
        df["frame_index"] = range(ep_length)
        df["timestamp"] = [i / src_dataset.meta.fps for i in range(ep_length)]
        df["task_index"] = new_meta.get_task_index(task)

        # Save data to parquet
        data_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, data_path, new_meta)

        # Compute episode stats
        ep_stats = _compute_episode_stats_from_df(df, new_meta.features)
        all_episode_stats.append(ep_stats)

        # Save episode metadata
        episode_dict = {
            "episode_index": new_ep_idx,
            "tasks": [task],
            "length": ep_length,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_frame_idx,
            "dataset_to_index": global_frame_idx + ep_length,
            # Augmentation metadata for duplicate detection
            META_SOURCE_EPISODE_INDEX: src_ep_idx,
            META_SCOOP_COUNT: scoop_count,
        }

        # Add video metadata (copy from source, but truncated)
        src_ep_meta = src_dataset.meta.episodes[src_ep_idx]
        for video_key in src_dataset.meta.video_keys:
            # Calculate new video duration based on cut
            src_from_ts = src_ep_meta[f"videos/{video_key}/from_timestamp"]
            src_length = src_ep_meta["length"]
            src_to_ts = src_ep_meta[f"videos/{video_key}/to_timestamp"]
            src_duration = src_to_ts - src_from_ts

            # New duration proportional to cut point
            new_duration = (cut_frame / src_length) * src_duration

            episode_dict[f"videos/{video_key}/chunk_index"] = src_ep_meta[f"videos/{video_key}/chunk_index"]
            episode_dict[f"videos/{video_key}/file_index"] = src_ep_meta[f"videos/{video_key}/file_index"]
            episode_dict[f"videos/{video_key}/from_timestamp"] = src_from_ts
            episode_dict[f"videos/{video_key}/to_timestamp"] = src_from_ts + new_duration

        episode_dict.update(flatten_dict({"stats": ep_stats}))
        new_meta._save_episode_metadata(episode_dict)

        global_frame_idx += ep_length
        file_idx += 1

    # Close metadata writer
    new_meta._close_writer()

    # Update info
    new_meta.info.update(
        {
            "total_episodes": len(generation_plan),
            "total_frames": global_frame_idx,
            "total_tasks": len(all_tasks),
            "splits": {"train": f"0:{len(generation_plan)}"},
        }
    )
    write_info(new_meta.info, new_meta.root)

    # Aggregate and save stats
    aggregated_stats = aggregate_stats(all_episode_stats)
    write_stats(aggregated_stats, new_meta.root)

    # Copy video files (we reference the same video files but with different timestamps)
    logger.info("Copying video files...")
    _copy_source_videos(src_dataset, new_meta, generation_plan)

    # Load and return the new dataset
    new_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
    )

    logger.info(f"Created augmented dataset with {new_dataset.num_episodes} episodes")
    return new_dataset


def _write_parquet(df: pd.DataFrame, path: Path, meta: LeRobotDatasetMetadata) -> None:
    """Write DataFrame to parquet with proper HF dataset format."""
    import datasets
    import pyarrow.parquet as pq

    from lerobot.datasets.utils import embed_images, get_hf_features_from_features

    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")

    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)

    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _compute_episode_stats_from_df(df: pd.DataFrame, features: dict) -> dict:
    """Compute statistics for an episode from a DataFrame."""
    stats = {}

    for feature_name, feature_info in features.items():
        dtype = feature_info["dtype"]

        # Skip non-numeric features
        if dtype in ["string", "video", "image"]:
            continue

        if feature_name not in df.columns:
            continue

        values = df[feature_name].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array([np.array(v) for v in values])
        else:
            values = np.array(values)

        if values.dtype in [np.float32, np.float64, np.int32, np.int64]:
            # Ensure stats are numpy arrays (not scalars) for aggregate_stats compatibility
            min_val = np.atleast_1d(np.min(values, axis=0).astype(np.float64))
            max_val = np.atleast_1d(np.max(values, axis=0).astype(np.float64))
            mean_val = np.atleast_1d(np.mean(values, axis=0).astype(np.float64))
            std_val = np.atleast_1d(np.std(values, axis=0).astype(np.float64))

            stats[feature_name] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "count": np.array([len(values)]),
            }

    return stats


def _copy_source_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    generation_plan: list[dict],
) -> None:
    """Copy video files from source dataset that are referenced by the new episodes."""
    # Find unique video files needed
    video_files_needed = set()

    for plan in generation_plan:
        src_ep_idx = plan["src_episode"]
        src_ep_meta = src_dataset.meta.episodes[src_ep_idx]

        for video_key in src_dataset.meta.video_keys:
            chunk_idx = src_ep_meta[f"videos/{video_key}/chunk_index"]
            file_idx = src_ep_meta[f"videos/{video_key}/file_index"]
            video_files_needed.add((video_key, chunk_idx, file_idx))

    # Copy each unique video file
    for video_key, chunk_idx, file_idx in tqdm(video_files_needed, desc="Copying videos"):
        src_path = src_dataset.root / src_dataset.meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        dst_path = dst_meta.root / dst_meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if not dst_path.exists():
            shutil.copy(src_path, dst_path)


def create_backup(dataset_root: Path) -> Path:
    """
    Create a backup of the dataset directory.

    Args:
        dataset_root: Path to the dataset root directory

    Returns:
        Path to the backup directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = dataset_root.parent / f"{dataset_root.name}_backup_{timestamp}"

    logger.info(f"Creating backup at {backup_path}")
    shutil.copytree(dataset_root, backup_path)
    logger.info("Backup created successfully")

    return backup_path


def ensure_augmentation_columns_exist(dataset: LeRobotDataset) -> None:
    """
    Ensure that augmentation columns exist in all episode metadata files.
    If not, add them with default values (-1).
    """
    # Check if columns already exist in the loaded episodes
    # We check the first episode if available
    if dataset.num_episodes > 0:
        first_ep = dataset.meta.episodes[0]
        if META_SOURCE_EPISODE_INDEX in first_ep and META_SCOOP_COUNT in first_ep:
            return

    logger.info("Updating existing episode metadata to include augmentation columns...")

    episodes_dir = dataset.root / "meta/episodes"
    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))

    for file_path in parquet_files:
        df = pd.read_parquet(file_path)

        changed = False
        if META_SOURCE_EPISODE_INDEX not in df.columns:
            df[META_SOURCE_EPISODE_INDEX] = -1
            changed = True

        if META_SCOOP_COUNT not in df.columns:
            df[META_SCOOP_COUNT] = -1
            changed = True

        if changed:
            logger.info(f"Updating {file_path}")
            # We need to ensure the types are correct (int64)
            df[META_SOURCE_EPISODE_INDEX] = df[META_SOURCE_EPISODE_INDEX].astype("int64")
            df[META_SCOOP_COUNT] = df[META_SCOOP_COUNT].astype("int64")

            df.to_parquet(file_path)

    # Reload episodes metadata to reflect changes
    dataset.meta.episodes = load_episodes(dataset.root)


def add_episodes_to_existing_dataset(
    dataset: LeRobotDataset,
    full_dataset: LeRobotDataset,
    generation_plan: list[dict],
    dry_run: bool = False,
) -> LeRobotDataset | None:
    """
    Add new partial scoop episodes to an existing dataset.

    Args:
        dataset: Dataset with loaded data for source episodes (may be a subset)
        full_dataset: Full dataset with all episode metadata
        generation_plan: List of episodes to generate
        dry_run: If True, only print what would be done

    Returns:
        Updated dataset, or None if dry_run
    """
    if not generation_plan:
        logger.info("No new episodes to add")
        return full_dataset

    if dry_run:
        print("\n=== Dry Run - Episodes to Add ===")
        print(f"Dataset: {full_dataset.repo_id}")
        print(f"Current episodes: {full_dataset.num_episodes}")
        print(f"Episodes to add: {len(generation_plan)}")
        print(f"New total: {full_dataset.num_episodes + len(generation_plan)}")
        print("\nNew episodes:")
        for plan in generation_plan[:10]:
            print(
                f"  Source episode {plan['src_episode']} -> {plan['scoop_count']} scoop(s) (cut at frame {plan['cut_frame']})"
            )
        if len(generation_plan) > 10:
            print(f"  ... and {len(generation_plan) - 10} more")
        return None

    # Ensure augmentation columns exist in existing metadata
    ensure_augmentation_columns_exist(full_dataset)

    # Save new tasks if needed
    all_tasks = [TASK_TEMPLATE.format(n=n) for n in [1, 2, 3, 4]]
    full_dataset.meta.save_episode_tasks(all_tasks)

    # Get starting indices from full dataset
    start_ep_idx = full_dataset.num_episodes
    global_frame_idx = full_dataset.num_frames

    # Determine chunk and file indices for new data
    if start_ep_idx > 0:
        last_ep = full_dataset.meta.episodes[start_ep_idx - 1]
        chunk_idx = last_ep.get("data/chunk_index", 0)
        file_idx = last_ep.get("data/file_index", 0) + 1
    else:
        chunk_idx, file_idx = 0, 0

    all_episode_stats = []

    for i, plan in enumerate(tqdm(generation_plan, desc="Adding episodes")):
        new_ep_idx = start_ep_idx + i
        src_ep_idx = plan["src_episode"]
        scoop_count = plan["scoop_count"]
        cut_frame = plan["cut_frame"]
        task = plan["task"]

        # Generate the partial episode data (use dataset which has the actual data loaded)
        df = generate_partial_episode(dataset, src_ep_idx, scoop_count, cut_frame)

        ep_length = len(df)

        # Update indices
        df["index"] = range(global_frame_idx, global_frame_idx + ep_length)
        df["episode_index"] = new_ep_idx
        df["frame_index"] = range(ep_length)
        df["timestamp"] = [j / full_dataset.meta.fps for j in range(ep_length)]
        df["task_index"] = full_dataset.meta.get_task_index(task)

        # Save data to parquet (use full_dataset root for correct location)
        data_path = full_dataset.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, data_path, full_dataset.meta)

        # Compute episode stats
        ep_stats = _compute_episode_stats_from_df(df, full_dataset.meta.features)
        all_episode_stats.append(ep_stats)

        # Save episode metadata
        episode_dict = {
            "episode_index": new_ep_idx,
            "tasks": [task],
            "length": ep_length,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_frame_idx,
            "dataset_to_index": global_frame_idx + ep_length,
            # Augmentation metadata for duplicate detection
            META_SOURCE_EPISODE_INDEX: src_ep_idx,
            META_SCOOP_COUNT: scoop_count,
        }

        # Add video metadata (copy from source, but truncated)
        # Use dataset.meta which has the source episode metadata loaded
        src_ep_meta = dataset.meta.episodes[src_ep_idx]
        for video_key in full_dataset.meta.video_keys:
            # Calculate new video duration based on cut
            src_from_ts = src_ep_meta[f"videos/{video_key}/from_timestamp"]
            src_length = src_ep_meta["length"]
            src_to_ts = src_ep_meta[f"videos/{video_key}/to_timestamp"]
            src_duration = src_to_ts - src_from_ts

            # New duration proportional to cut point
            new_duration = (cut_frame / src_length) * src_duration

            episode_dict[f"videos/{video_key}/chunk_index"] = src_ep_meta[f"videos/{video_key}/chunk_index"]
            episode_dict[f"videos/{video_key}/file_index"] = src_ep_meta[f"videos/{video_key}/file_index"]
            episode_dict[f"videos/{video_key}/from_timestamp"] = src_from_ts
            episode_dict[f"videos/{video_key}/to_timestamp"] = src_from_ts + new_duration

        episode_dict.update(flatten_dict({"stats": ep_stats}))
        full_dataset.meta._save_episode_metadata(episode_dict)

        global_frame_idx += ep_length
        file_idx += 1

    # Close metadata writer
    full_dataset.meta._close_writer()

    # Update info
    new_total_episodes = start_ep_idx + len(generation_plan)
    full_dataset.meta.info.update(
        {
            "total_episodes": new_total_episodes,
            "total_frames": global_frame_idx,
            "total_tasks": len(full_dataset.meta.tasks),
            "splits": {"train": f"0:{new_total_episodes}"},
        }
    )
    write_info(full_dataset.meta.info, full_dataset.meta.root)

    # Aggregate and save stats
    if full_dataset.meta.stats is not None:
        all_episode_stats.insert(0, full_dataset.meta.stats)
    aggregated_stats = aggregate_stats(all_episode_stats)
    write_stats(aggregated_stats, full_dataset.meta.root)

    logger.info(f"Added {len(generation_plan)} episodes to dataset")
    logger.info(f"New total: {new_total_episodes} episodes, {global_frame_idx} frames")

    # Reload dataset to reflect changes
    updated_dataset = LeRobotDataset(
        repo_id=full_dataset.repo_id,
        root=full_dataset.root,
    )

    return updated_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate partial scoop episodes from 5-scoop episodes")
    parser.add_argument(
        "--source-repo",
        type=str,
        default="0xNOY/so101-with-narration",
        help="Source dataset repository ID",
    )
    parser.add_argument(
        "--source-revision",
        type=str,
        default="main",
        help="Source dataset revision",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default="0xNOY/so101-with-narration-augmented",
        help="Output dataset repository ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to HF cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done without creating the dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to HuggingFace Hub after creation/update",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the HuggingFace repository private (default: public)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup when modifying existing dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Loading source dataset: {args.source_repo}")
    # Load full dataset metadata first
    src_dataset_meta = LeRobotDataset(args.source_repo, revision=args.source_revision)

    logger.info(
        f"Source dataset has {src_dataset_meta.num_episodes} episodes, {src_dataset_meta.num_frames} frames"
    )

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Check if we're adding to an existing dataset (same source and output repo)
    is_in_place_update = args.source_repo == args.output_repo

    if is_in_place_update:
        logger.info("Source and output repos are the same - will add episodes to existing dataset")

        # Find 5-scoop episodes from full metadata
        five_scoop_episodes = find_5_scoop_episodes(src_dataset_meta)
        logger.info(f"Found {len(five_scoop_episodes)} episodes with 5-scoop task")

        if not five_scoop_episodes:
            logger.error("No 5-scoop episodes found in source dataset")
            return

        # Check for existing augmented episodes from full metadata
        existing_augmented = find_existing_augmented_episodes(src_dataset_meta)
        if existing_augmented:
            logger.info(f"Found {len(existing_augmented)} existing augmented episodes")

        # Ensure data for 5-scoop episodes is downloaded
        # Reload dataset with specific episodes to ensure data is available
        logger.info("Downloading data for 5-scoop episodes...")
        src_dataset = LeRobotDataset(
            args.source_repo,
            revision=args.source_revision,
            episodes=five_scoop_episodes,
        )

        # Create generation plan
        generation_plan = []
        for ep_idx in five_scoop_episodes:
            cut_points = find_cut_points(src_dataset, ep_idx)
            for scoop_count in [1, 2, 3, 4]:
                if scoop_count in cut_points:
                    generation_plan.append(
                        {
                            "src_episode": ep_idx,
                            "scoop_count": scoop_count,
                            "cut_frame": cut_points[scoop_count],
                            "task": TASK_TEMPLATE.format(n=scoop_count),
                        }
                    )

        # Filter out already generated episodes
        generation_plan = filter_generation_plan(generation_plan, existing_augmented)

        if not generation_plan:
            logger.info("All partial scoop episodes already exist. Nothing to do.")
            return

        logger.info(f"Will add {len(generation_plan)} new episodes:")
        for scoop_count in [1, 2, 3, 4]:
            count = sum(1 for p in generation_plan if p["scoop_count"] == scoop_count)
            if count > 0:
                logger.info(f"  - {count} episodes for {scoop_count} scoop(s)")

        # Create backup before modifying (unless skipped)
        backup_path = None
        if not args.dry_run and not args.no_backup:
            backup_path = create_backup(src_dataset_meta.root)

        # Add episodes to existing dataset
        # Use src_dataset_meta for metadata access, src_dataset for data access
        result_dataset = add_episodes_to_existing_dataset(
            dataset=src_dataset,
            full_dataset=src_dataset_meta,
            generation_plan=generation_plan,
            dry_run=args.dry_run,
        )

        if result_dataset is not None:
            logger.info("\n=== Updated Dataset Summary ===")
            logger.info(f"Repository: {result_dataset.repo_id}")
            logger.info(f"Total Episodes: {result_dataset.num_episodes}")
            logger.info(f"Total Frames: {result_dataset.num_frames}")
            logger.info(f"Location: {result_dataset.root}")

            if backup_path:
                logger.info(f"Backup saved at: {backup_path}")

            # Push to HuggingFace Hub if requested
            if args.push_to_hub:
                logger.info("Pushing dataset to HuggingFace Hub...")
                result_dataset.push_to_hub(
                    private=args.private,
                    tags=["augmented", "partial-scoop"],
                )
                logger.info(f"Dataset pushed to: https://huggingface.co/datasets/{result_dataset.repo_id}")

    else:
        # Create a new augmented dataset
        new_dataset = create_augmented_dataset(
            src_dataset=src_dataset_meta,
            output_repo_id=args.output_repo,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )

        if new_dataset is not None:
            logger.info("\n=== Created Dataset Summary ===")
            logger.info(f"Repository: {new_dataset.repo_id}")
            logger.info(f"Total Episodes: {new_dataset.num_episodes}")
            logger.info(f"Total Frames: {new_dataset.num_frames}")
            logger.info(f"Location: {new_dataset.root}")

            # Push to HuggingFace Hub if requested
            if args.push_to_hub:
                logger.info("Pushing dataset to HuggingFace Hub...")
                new_dataset.push_to_hub(
                    private=args.private,
                    tags=["augmented", "partial-scoop"],
                )
                logger.info(f"Dataset pushed to: https://huggingface.co/datasets/{new_dataset.repo_id}")


if __name__ == "__main__":
    main()
