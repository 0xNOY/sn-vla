import argparse
from collections import Counter
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Analyze LeRobot dataset statistics")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., 0xNOY/so101-with-narration)")
    parser.add_argument(
        "revision", type=str, nargs="?", default="main", help="Dataset revision (default: main)"
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name} (revision: {args.revision})...")
    dataset = LeRobotDataset(args.dataset_name, revision=args.revision)
    print(f"\n=== Dataset Overview ===")
    print(f"Total Frames: {dataset.num_frames}")
    print(f"Total Episodes: {dataset.num_episodes}")
    print(f"FPS: {dataset.fps}")
    print(f"Local Path: {dataset.root}")

    # Analyze Tasks
    print("\nAnalyzing Tasks...")
    tasks = []
    # Try to get tasks from metadata first
    if "task" in dataset.meta.episodes:
        tasks = dataset.meta.episodes["task"]
    elif "instruction" in dataset.meta.episodes:
        tasks = dataset.meta.episodes["instruction"]
    else:
        # Fallback: check first frame of each episode
        print("Task info not found in metadata, checking first frame of each episode...")
        for ep_idx in tqdm(range(dataset.num_episodes)):
            from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
            found_task = False

            # We only need one frame, but dataset[i] loads images which is slow.
            # If we can access hf_dataset, it's faster.
            if hasattr(dataset, "hf_dataset"):
                # Try to fetch only the task column if possible, or just the item
                # HF dataset access is usually row-based but might be lazy for images if they are paths
                try:
                    item = dataset.hf_dataset[from_idx]
                    if "task" in item:
                        tasks.append(item["task"])
                        found_task = True
                    elif "instruction" in item:
                        tasks.append(item["instruction"])
                        found_task = True
                    elif "language_instruction" in item:
                        tasks.append(item["language_instruction"])
                        found_task = True
                except Exception:
                    pass

            if not found_task:
                # Fallback to standard access
                frame = dataset[from_idx]
                if "task" in frame:
                    tasks.append(frame["task"])
                elif "instruction" in frame:
                    tasks.append(frame["instruction"])
                elif "language_instruction" in frame:
                    tasks.append(frame["language_instruction"])
                else:
                    tasks.append("unknown")

    task_counts = Counter(tasks)
    print(f"\n=== Task Distribution ({len(task_counts)} tasks) ===")
    for task, count in task_counts.most_common():
        print(f"- {task}: {count} episodes")

    # Analyze Narrations
    print("\nAnalyzing Narrations...")
    narration_key = "current_narration"

    # Check if narration key exists in features
    if narration_key not in dataset.features:
        print(f"Feature '{narration_key}' not found in dataset.")
        return

    all_narrations = []
    narrations_per_episode = []

    # To speed up, we can try to use hf_dataset if available to avoid image decoding
    use_hf_direct = hasattr(dataset, "hf_dataset")

    if use_hf_direct:
        print("Using direct HF dataset access for speed (skipping image decoding)...")

    for ep_idx in tqdm(range(dataset.num_episodes), desc="Scanning episodes"):
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

        ep_narrations = []

        if use_hf_direct:
            # Fetch the slice for this episode
            # Note: dataset.hf_dataset might be an IterableDataset or Dataset
            # If it's a Dataset (map-style), slicing works.
            try:
                # Accessing a range in HF dataset returns a dict of lists
                batch = dataset.hf_dataset[from_idx:to_idx]
                if narration_key in batch:
                    narrations = batch[narration_key]
                    # Filter empty
                    ep_narrations = [n for n in narrations if n and isinstance(n, str) and n.strip()]
                else:
                    # Fallback if key not in batch (unlikely if in features)
                    pass
            except Exception as e:
                # Fallback to slow loop
                use_hf_direct = False

        if not use_hf_direct:
            for i in range(from_idx, to_idx):
                frame = dataset[i]
                n = frame.get(narration_key)
                if n and isinstance(n, str) and n.strip():
                    ep_narrations.append(n)

        all_narrations.extend(ep_narrations)
        narrations_per_episode.append(len(ep_narrations))

    narration_counts = Counter(all_narrations)

    print(f"\n=== Narration Statistics ===")
    print(f"Total Narration Events: {len(all_narrations)}")
    if dataset.num_frames > 0:
        print(f"Narration Coverage: {len(all_narrations) / dataset.num_frames * 100:.2f}% of frames")
    print(f"Unique Narrations: {len(narration_counts)}")
    print(
        f"Avg Narrations per Episode: {np.mean(narrations_per_episode):.2f} (std: {np.std(narrations_per_episode):.2f})"
    )

    print("\nTop 20 Most Frequent Narrations:")
    for narration, count in narration_counts.most_common(20):
        print(f"{count:5d}: {repr(narration)}")


if __name__ == "__main__":
    main()
