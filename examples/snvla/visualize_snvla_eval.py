import argparse
import contextlib
import logging
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from bokeh.layouts import column, layout, row
from bokeh.models import Button, ColumnDataSource, Div, Slider, Span
from bokeh.plotting import curdoc, figure

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class VisualizerConfig:
    narration_width: int = 300
    narration_height: int = 500
    narration_font_size: str = "18px"
    image_height: int = 400
    bon_plot_height: int = 200
    bon_line_width: int = 1
    bon_line_color: str = "green"
    time_line_width: int = 1
    time_line_color: str = "red"
    play_button_width: int = 60
    animation_interval_ms: int = 1000 // 30
    timestamp_font_size: str = "16px"
    timestamp_height: int = 30


CONFIG = VisualizerConfig()


NARRATION_DIV_TEMPLATE = """
<div style="width: 100%; height: 100%; display: flex; flex-direction: column; gap: 10px; font-family: sans-serif;">
    <!-- Previous narrations (Context) -->
    <div style="flex-grow: 1; overflow-y: auto; padding: 10px; border: 1px solid #eee; border-radius: 5px; background-color: #fcfcfc;">
        <span style="color: #666; font-size: small;">Previous Commentary:</span><br>
        <span style="color: #444;">{previous_narrations}</span>
    </div>
    
    <!-- Current narration (Robot Chat Bubble) -->
    <div style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px;">
        <div style="font-size: 24px;">🤖</div>
        <div style="background-color: #e9e9eb; padding: 10px; border-radius: 15px; border-top-left-radius: 0; max-width: 80%;">
             <div style="font-weight: bold; font-size: {font_size}; color: #000;">{current_narration}</div>
        </div>
    </div>
</div>
"""

INSTRUCTION_DIV_TEMPLATE = """
<div style="width: 100%; padding: 10px; display: flex; flex-direction: row; justify-content: flex-end; align-items: flex-start; gap: 10px; font-family: sans-serif;">
    <div style="background-color: #007aff; color: white; padding: 10px; border-radius: 15px; border-top-right-radius: 0; max-width: 80%;">
        <div style="font-size: small; opacity: 0.8; margin-bottom: 2px;">User</div>
        <div style="font-size: {font_size};">{task_instruction}</div>
    </div>
    <div style="font-size: 24px;">👤</div>
</div>
"""


TIMESTAMP_DIV_TEMPLATE = """
<div style="font-size: {font_size}; font-weight: bold; text-align: center; width: 100%; height: 100%;">
    Time: {time:.2f} s
</div>
"""


def load_data(dataset, episode_index):
    """Load data for a specific episode."""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # Pre-load all data for the episode to avoid latency during interaction
    # Note: For very large episodes, this might need optimization (lazy loading)
    data = {
        "index": np.arange(to_idx - from_idx),
        "prob_bon": [],
        "current_narration": [],
        "previous_narrations": [],
        "images": {},  # key: list of rgba arrays
        "timestamp": [],
        "task_instruction": "",
    }

    # Get task instruction from the first frame of the episode
    first_frame = dataset[from_idx]
    if "task" in first_frame:
        data["task_instruction"] = first_frame["task"]
    elif "language_instruction" in first_frame:
        data["task_instruction"] = first_frame["language_instruction"]
    else:
        # Fallback: check metadata if available
        data["task_instruction"] = "Execute the task."

    camera_keys = dataset.meta.camera_keys
    for key in camera_keys:
        data["images"][key] = []

    prev_narrations = ""
    for idx in range(from_idx, to_idx):
        frame = dataset[idx]

        # Metrics
        data["prob_bon"].append(frame.get("prob_bon", 0.0).item() if "prob_bon" in frame else 0.0)
        data["current_narration"].append(frame.get("current_narration", "").replace("\n", "<br>"))
        data["timestamp"].append(frame.get("timestamp", 0.0).item())

        data["previous_narrations"].append(prev_narrations)
        prev_narrations += data["current_narration"][-1]

        # Images
        for key in camera_keys:
            if key in frame:
                # CHW float32 -> HWC uint8 RGBA for Bokeh
                img_tensor = frame[key]
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                # Add Alpha channel
                h, w, c = img_np.shape
                if c == 3:
                    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                    img_rgba = np.concatenate([img_np, alpha], axis=2)
                else:
                    img_rgba = img_np

                # Flip vertically because Bokeh origin is bottom-left
                img_rgba = np.ascontiguousarray(np.flipud(img_rgba))

                # Convert to 32-bit integer array for Bokeh
                # (M, N) array of RGBA values packed into 32-bit integers
                view = img_rgba.view(dtype=np.uint32).reshape((h, w))
                data["images"][key].append(view)

    return data, camera_keys


def create_visualization(doc):
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--root", type=str, default=None)

    # Parse only known args to avoid issues if bokeh injects others (though --args should isolate)
    args, _ = parser.parse_known_args(sys.argv[1:])

    repo_id = args.repo_id
    episode_index = args.episode_index
    root = args.root
    if root == "None":
        root = None
    elif root is not None:
        root = Path(root)

    logging.info(f"Loading dataset: {repo_id}, Episode: {episode_index}")

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        doc.add_root(Div(text=f"Error loading dataset: {e}"))
        return

    data, camera_keys = load_data(dataset, episode_index)
    num_frames = len(data["index"])

    # --- Data Sources ---
    # Add image sources
    image_sources = {}
    for key in camera_keys:
        image_sources[key] = ColumnDataSource(data={"image": [data["images"][key][0]]})

    # Source for the full timeline (BON graph)
    timeline_source = ColumnDataSource(data={"index": data["index"], "prob_bon": data["prob_bon"]})

    # --- Components ---

    # 1. Instruction Box (User)
    instruction_div = Div(
        text=INSTRUCTION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            task_instruction=data["task_instruction"],
        ),
        width=CONFIG.narration_width,
        # Adjust height as needed, or let it be auto if supported (Div supports height)
        height=100,
    )

    # 2. Narration Box (Robot)
    # We combine previous and current narration into one chat-like interface
    narration_div = Div(
        text=NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=data["previous_narrations"][0],
            current_narration=data["current_narration"][0],
        ),
        width=CONFIG.narration_width,
        height=CONFIG.narration_height - 100,  # Substract instruction height
    )

    # 2. Camera Views
    image_plots = []
    total_width = 0
    for key in camera_keys:
        # Get dimensions from the first frame
        h, w = data["images"][key][0].shape
        width = CONFIG.image_height * w // h
        total_width += width
        p = figure(
            title=f"Camera: {key}",
            x_range=(0, w),
            y_range=(0, h),
            width=width,
            height=CONFIG.image_height,
            tools="",
        )
        p.image_rgba(image="image", x=0, y=0, dw=w, dh=h, source=image_sources[key])
        p.axis.visible = False
        p.grid.visible = False
        image_plots.append(p)

    # 3. BON Probability Plot
    bon_plot = figure(
        title="BON Probability",
        x_axis_label="Time Step",
        y_axis_label="Probability",
        width=total_width,
        height=CONFIG.bon_plot_height,
        x_range=(0, num_frames),
    )
    bon_plot.line(
        "index",
        "prob_bon",
        source=timeline_source,
        line_width=CONFIG.bon_line_width,
        color=CONFIG.bon_line_color,
    )

    # Vertical line for current time
    time_line = Span(
        location=0, dimension="height", line_color=CONFIG.time_line_color, line_width=CONFIG.time_line_width
    )
    bon_plot.add_layout(time_line)

    # 4. Slider
    slider = Slider(
        start=0,
        end=num_frames - 1,
        value=0,
        step=1,
        title="Time Step",
        width=total_width - CONFIG.play_button_width,
    )

    # 5. Play/Export Controls
    play_button = Button(label="Play", width=CONFIG.play_button_width, button_type="success")

    # 6. Timestamp Display
    timestamp_div = Div(
        text=TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][0],
        ),
        width=total_width - CONFIG.play_button_width,
        height=CONFIG.timestamp_height,
    )

    # --- Callbacks ---
    def update(attr, old, new):
        idx = int(new)

        # Update text
        prev = data["previous_narrations"][idx]
        curr = data["current_narration"][idx]
        narration_div.text = NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=prev,
            current_narration=curr,
        )

        # Update images
        for key in camera_keys:
            image_sources[key].data = {"image": [data["images"][key][idx]]}

        # Update time line
        time_line.location = idx

        # Update timestamp
        timestamp_div.text = TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][idx],
        )

    slider.on_change("value", update)

    def animate_update():
        frame = slider.value + 1
        if frame >= num_frames:
            frame = 0
        slider.value = frame

    callback_id = None

    def toggle_play():
        nonlocal callback_id
        if play_button.label == "Play":
            play_button.label = "Pause"
            callback_id = doc.add_periodic_callback(animate_update, CONFIG.animation_interval_ms)
        else:
            play_button.label = "Play"
            if callback_id:
                with contextlib.suppress(ValueError):
                    doc.remove_periodic_callback(callback_id)

    play_button.on_click(toggle_play)

    # --- Layout ---
    controls = row(play_button, slider)

    main_layout = layout(
        [
            row(
                column(
                    row(image_plots),
                    bon_plot,
                    timestamp_div,
                    controls,
                ),
                column(
                    instruction_div,
                    narration_div,
                ),
            ),
        ],
    )

    doc.add_root(main_layout)
    doc.title = "SNVLA Evaluation Visualizer"


# To run this script:
# bokeh serve src/lerobot/scripts/visualize_snvla_eval.py --args --repo-id <repo_id> --episode-index <idx>

create_visualization(curdoc())
