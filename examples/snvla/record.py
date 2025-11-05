#!/usr/bin/env python

"""
Records a dataset for SN-VLA (Self-Narrating Vision-Language-Action) policy.

This script extends the standard LeRobot recording script with narration support.
It displays robot camera views with status information and allows inserting
narrations at specific frames using keyboard input.

Features:
- Display robot camera views in real-time using OpenCV
- Compose multiple camera views into a single display window
- Show recording status (time, narrations, etc.) overlaid on the view
- Insert narrations by pressing Enter (uses predefined narration list)
- End episode when narration list is empty and Enter is pressed
- Abort and delete episode by holding 'i' for 1 second

Example usage:

```shell
python examples/snvla/record.py
```

Or customize robot, teleop, dataset settings as needed.
"""

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.constants import (
    ACTION,
    COMPLEMENTARY_DATA,
    CURRENT_NARRATION,
    OBS_STR,
    PREVIOUS_NARRATIONS,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say


@dataclass
class SNVLADatasetRecordConfig:
    """Configuration for SN-VLA dataset recording."""

    # Dataset identifier (e.g. 'username/snvla_dataset')
    repo_id: str
    # Task description
    single_task: str
    # Root directory for dataset storage
    root: str | Path | None = None
    # Frames per second
    fps: int = 30
    # Episode recording time (seconds)
    episode_time_s: int | float = 60
    # Reset time between episodes (seconds)
    reset_time_s: int | float = 30
    # Number of episodes to record
    num_episodes: int = 10
    # Encode videos
    video: bool = True
    # Upload to Hugging Face Hub
    push_to_hub: bool = False
    # Private repository
    private: bool = False
    # Hub tags
    tags: list[str] | None = None
    # Image writer settings
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    # Video encoding batch size
    video_encoding_batch_size: int = 1


@dataclass
class SNVLARecordConfig:
    """Main configuration for SN-VLA recording."""

    robot: RobotConfig
    dataset: SNVLADatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    # Display cameras with OpenCV
    display_cameras: bool = True
    # Play sounds for events
    play_sounds: bool = True
    # Resume recording
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError("Teleoperator is required for SN-VLA recording")


def compose_camera_views(images: dict[str, np.ndarray]) -> np.ndarray:
    """
    Compose multiple camera views into a single image for display.

    Args:
        images: Dictionary mapping camera names to images (H, W, C) in BGR format

    Returns:
        Composed image suitable for display
    """
    if not images:
        # Return black image if no cameras
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if len(images) == 1:
        return list(images.values())[0]

    # Convert all images to same height for composition
    target_height = 480
    resized_images = []

    for img in images.values():
        h, w = img.shape[:2]
        aspect = w / h
        new_width = int(target_height * aspect)
        resized = cv2.resize(img, (new_width, target_height))
        resized_images.append(resized)

    # Arrange images in a grid
    if len(resized_images) <= 3:
        # Horizontal arrangement for 2-3 cameras
        composed = np.hstack(resized_images)
    else:
        # Grid arrangement for 4+ cameras
        rows = []
        row_size = 2
        for i in range(0, len(resized_images), row_size):
            row = resized_images[i : i + row_size]
            # Pad last row if needed
            if len(row) < row_size:
                last_width = row[-1].shape[1]
                padding = np.zeros((target_height, last_width, 3), dtype=np.uint8)
                row.append(padding)
            rows.append(np.hstack(row))
        composed = np.vstack(rows)

    return composed


def add_status_overlay(
    image: np.ndarray,
    elapsed_time: float,
    previous_narrations: list[str],
    next_narration: str | None,
    episode_idx: int,
    frame_idx: int,
) -> np.ndarray:
    """
    Add status information overlay to the image.

    Args:
        image: Input image (BGR format)
        elapsed_time: Elapsed recording time in seconds
        previous_narrations: List of previously inserted narrations
        next_narration: Next narration to be inserted (if available)
        episode_idx: Current episode index
        frame_idx: Current frame index

    Returns:
        Image with status overlay
    """
    # Create overlay
    overlay = image.copy()
    h, w = overlay.shape[:2]

    # Semi-transparent background for text
    bg_height = 180
    cv2.rectangle(overlay, (0, 0), (w, bg_height), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 30
    y_offset = 25

    # Status text
    texts = [
        f"Episode: {episode_idx} | Frame: {frame_idx} | Time: {elapsed_time:.1f}s",
        f"Previous: {previous_narrations[-1] if previous_narrations else 'None'}",
        f"Next: {next_narration if next_narration else '[End Episode]'}",
        "",
        "Controls: [Enter] Insert narration | [i hold 1s] Abort episode",
    ]

    for i, text in enumerate(texts):
        y = y_offset + i * line_height
        color = (0, 255, 0) if i < 3 else (150, 150, 150)
        cv2.putText(image, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def snvla_record_loop(
    robot: Robot,
    teleop: Teleoperator,
    dataset: LeRobotDataset,
    narration_list: list[str],
    episode_idx: int,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    control_time_s: int | float,
    single_task: str,
    display_cameras: bool = True,
) -> tuple[bool, bool]:
    """
    Recording loop for SN-VLA with narration support.

    Args:
        robot: Robot instance
        teleop: Teleoperator instance
        dataset: LeRobotDataset instance
        narration_list: List of narrations to insert (will be consumed)
        episode_idx: Current episode index
        fps: Target frames per second
        teleop_action_processor: Processor for teleop actions
        robot_action_processor: Processor for robot actions
        robot_observation_processor: Processor for robot observations
        control_time_s: Maximum recording time
        single_task: Task description
        display_cameras: Whether to display camera views

    Returns:
        Tuple of (should_stop_recording, should_abort_episode)
    """
    timestamp = 0
    frame_idx = 0
    start_episode_t = time.perf_counter()

    # Narration tracking
    previous_narrations = []
    narration_queue = deque(narration_list)

    # Keyboard state tracking
    i_key_press_start = None
    abort_hold_time = 1.0  # seconds to hold 'i' to abort

    # OpenCV window
    if display_cameras:
        cv2.namedWindow("SN-VLA Recording", cv2.WINDOW_NORMAL)

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        # Build observation frame for dataset
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop
        act = teleop.get_action()
        act_processed_teleop = teleop_action_processor((act, obs))
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        robot.send_action(robot_action_to_send)

        # Build action frame
        action_frame = build_dataset_frame(dataset.features, act_processed_teleop, prefix=ACTION)

        # Handle keyboard input for narrations
        current_narration = ""
        if display_cameras:
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key
                if len(narration_queue) > 0:
                    current_narration = narration_queue.popleft()
                    previous_narrations.append(current_narration)
                    logging.info(f"[Frame {frame_idx}] Narration inserted: {current_narration}")
                else:
                    # End episode when no more narrations
                    logging.info("No more narrations. Ending episode.")
                    cv2.destroyAllWindows()
                    return False, False

            elif key == ord("i"):
                if i_key_press_start is None:
                    i_key_press_start = time.perf_counter()
                elif time.perf_counter() - i_key_press_start >= abort_hold_time:
                    logging.warning("Aborting episode!")
                    cv2.destroyAllWindows()
                    return False, True
            else:
                i_key_press_start = None

        # Prepare complete frame with narration data
        frame = {
            **observation_frame,
            **action_frame,
            "task": single_task,
            f"{COMPLEMENTARY_DATA}.{CURRENT_NARRATION}": current_narration,
            f"{COMPLEMENTARY_DATA}.{PREVIOUS_NARRATIONS}": "\n".join(previous_narrations),
        }

        # Add frame to dataset
        dataset.add_frame(frame)

        # Display camera views with status
        if display_cameras and hasattr(robot, "cameras") and len(robot.cameras) > 0:
            # Get camera images
            camera_images = {}
            for cam_name in robot.cameras:
                # Get image from observation
                obs_key = f"observation.images.{cam_name}"
                if obs_key in obs_processed:
                    img = obs_processed[obs_key]
                    # Convert from tensor/numpy to BGR
                    if hasattr(img, "numpy"):
                        img = img.numpy()
                    img = np.asarray(img, dtype=np.uint8)
                    # Convert RGB to BGR for OpenCV
                    if img.shape[-1] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    camera_images[cam_name] = img

            # Compose views
            composed = compose_camera_views(camera_images)

            # Add status overlay
            next_narration = narration_queue[0] if len(narration_queue) > 0 else None
            composed = add_status_overlay(
                composed,
                elapsed_time=timestamp,
                previous_narrations=previous_narrations,
                next_narration=next_narration,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
            )

            # Display
            cv2.imshow("SN-VLA Recording", composed)

        # Wait for next frame
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t
        frame_idx += 1

    if display_cameras:
        cv2.destroyAllWindows()

    return False, False


@parser.wrap()
def snvla_record(cfg: SNVLARecordConfig) -> LeRobotDataset:
    """
    Main function for recording SN-VLA datasets.
    """
    init_logging()
    logging.info("Starting SN-VLA recording session")
    logging.info(asdict(cfg))

    # Create robot and teleoperator
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Create processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Prepare dataset features
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    # Add complementary data features for narrations
    dataset_features[f"{COMPLEMENTARY_DATA}.{CURRENT_NARRATION}"] = {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    }
    dataset_features[f"{COMPLEMENTARY_DATA}.{PREVIOUS_NARRATIONS}"] = {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    }

    # Create or load dataset
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
    else:
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # Connect hardware
    robot.connect()
    if teleop is not None:
        teleop.connect()

    # Define narration list (customize per task)
    # TODO: Load from config or external file
    narration_list = [
        "Approaching the target object",
        "Grasping the object with gripper",
        "Lifting the object upward",
        "Moving toward the destination",
        "Placing the object down",
        "Releasing the gripper",
        "Returning to rest position",
    ]

    logging.info(f"Narration list ({len(narration_list)} items): {narration_list}")

    # Recording loop
    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes:
            log_say(f"Recording episode {recorded_episodes}", cfg.play_sounds)

            # Make a copy of narration list for this episode
            episode_narrations = narration_list.copy()

            # Record episode
            should_stop, should_abort = snvla_record_loop(
                robot=robot,
                teleop=teleop,
                dataset=dataset,
                narration_list=episode_narrations,
                episode_idx=recorded_episodes,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_cameras=cfg.display_cameras,
            )

            if should_abort:
                log_say("Episode aborted, deleting data", cfg.play_sounds)
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1

            if should_stop or recorded_episodes >= cfg.dataset.num_episodes:
                break

            # Reset phase
            if recorded_episodes < cfg.dataset.num_episodes:
                log_say("Reset the environment for next episode", cfg.play_sounds)
                time.sleep(cfg.dataset.reset_time_s)

    log_say("Recording complete", cfg.play_sounds)

    # Disconnect hardware
    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    # Push to hub if requested
    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    """Entry point for the script."""
    snvla_record()


if __name__ == "__main__":
    main()
