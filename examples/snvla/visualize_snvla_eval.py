import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, Div, Slider, Span, Range1d
from bokeh.plotting import figure, curdoc
from bokeh.io import show

from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
        "images": {} # key: list of rgba arrays
    }
    
    camera_keys = dataset.meta.camera_keys
    for key in camera_keys:
        data["images"][key] = []

    prev_narrations = ""
    for i, idx in enumerate(range(from_idx, to_idx)):
        frame = dataset[idx]
        
        # Metrics
        data["prob_bon"].append(frame.get("prob_bon", 0.0).item() if "prob_bon" in frame else 0.0)
        data["current_narration"].append(frame.get("current_narration", "").replace("\n", "<br>"))

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
    # Source for the current frame (images, text)
    current_source = ColumnDataSource(data={
        "previous_narrations": [data["previous_narrations"][0]],
        "current_narration": [data["current_narration"][0]],
    })
    
    # Add image sources
    image_sources = {}
    for key in camera_keys:
        image_sources[key] = ColumnDataSource(data={"image": [data["images"][key][0]]})

    # Source for the full timeline (BON graph)
    timeline_source = ColumnDataSource(data={
        "index": data["index"],
        "prob_bon": data["prob_bon"]
    })

    # --- Components ---

    # 1. Narration Box
    narration_div = Div(
        text=f"""
        <div style="font-size: 16px; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9; height: 100px; overflow-y: auto;">
            <span style="color: black;">{data['previous_narrations'][0]}</span>
            <span style="color: blue; font-weight: bold;">{data['current_narration'][0]}</span>
        </div>
        """,
        width=800, height=120
    )

    # 2. Camera Views
    image_plots = []
    for key in camera_keys:
        # Get dimensions from the first frame
        h, w = data["images"][key][0].shape
        p = figure(title=f"Camera: {key}", x_range=(0, w), y_range=(0, h), 
                   width=300, height=int(300 * h / w), tools="")
        p.image_rgba(image="image", x=0, y=0, dw=w, dh=h, source=image_sources[key])
        p.axis.visible = False
        p.grid.visible = False
        image_plots.append(p)

    # 3. BON Probability Plot
    bon_plot = figure(title="BON Probability", x_axis_label="Time Step", y_axis_label="Probability",
                      width=800, height=200, x_range=(0, num_frames))
    bon_plot.line("index", "prob_bon", source=timeline_source, line_width=2, color="green")
    
    # Vertical line for current time
    time_line = Span(location=0, dimension='height', line_color='red', line_width=2)
    bon_plot.add_layout(time_line)

    # 4. Slider
    slider = Slider(start=0, end=num_frames - 1, value=0, step=1, title="Time Step", width=800)

    # --- Callbacks ---
    def update(attr, old, new):
        idx = int(new)
        
        # Update text
        prev = data["previous_narrations"][idx]
        curr = data["current_narration"][idx]
        narration_div.text = f"""
        <div style="font-size: 16px; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9; height: 100px; overflow-y: auto;">
            <span style="color: black;">{prev}</span>
            <span style="color: blue; font-weight: bold;">{curr}</span>
        </div>
        """
        
        # Update images
        for key in camera_keys:
            image_sources[key].data = {"image": [data["images"][key][idx]]}
            
        # Update time line
        time_line.location = idx

    slider.on_change("value", update)

    # --- Layout ---
    l = layout([
        [row(image_plots)],
        [narration_div],
        [bon_plot],
        [slider]
    ])
    
    doc.add_root(l)
    doc.title = "SNVLA Evaluation Visualizer"

# To run this script:
# bokeh serve src/lerobot/scripts/visualize_snvla_eval.py --args --repo-id <repo_id> --episode-index <idx>

create_visualization(curdoc())
