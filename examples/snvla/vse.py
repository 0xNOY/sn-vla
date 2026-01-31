import argparse
import bisect
import contextlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    Div,
    InlineStyleSheet,
    Range1d,
    Select,
    Slider,
    Span,
)
from bokeh.plotting import curdoc, figure

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class VisualizerConfig:
    narration_font_size: str = "2em"
    image_height: int = 300
    bon_plot_height: int = 200
    bon_line_width: int = 1
    bon_line_color: str = "green"
    boa_line_width: int = 1
    boa_line_color: str = "blue"
    time_line_width: int = 1
    time_line_color: str = "red"
    animation_interval_ms: int = 1000 // 60
    timestamp_font_size: str = "1em"
    timestamp_height: int = 30


CONFIG = VisualizerConfig()


NARRATION_DIV_TEMPLATE = """
<div style="width: 100%; height: 100%; display: flex; flex-direction: column; gap: 10px; font-family: sans-serif; font-size: {font_size};">
    <div style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px;">
        <div style="font-size: 1.5em;">🤖</div>
        <div style="font-size: 1em; background-color: #e9e9eb; padding: 10px; border-radius: 15px; border-top-left-radius: 0; max-width: 90%; margin-top: 0.75em;">
            <span style="color: #444;">{previous_narrations}</span>
            <span style="font-weight: bold; color: #000;">{current_narration}</span>
        </div>
    </div>
</div>
"""

INSTRUCTION_DIV_TEMPLATE = """
<div style="width: 100%; padding: 10px; display: flex; flex-direction: row; justify-content: flex-end; align-items: flex-start; gap: 10px; font-family: sans-serif; font-size: {font_size};">
    <div style="background-color: #007aff; color: white; padding: 10px; border-radius: 15px; border-top-right-radius: 0; max-width: 90%; margin-top: 0.75em;">
        <div style="font-size: 0.8em; opacity: 0.8; margin-bottom: 2px;">Task</div>
        <div style="font-size: 1em;">{task_instruction}</div>
    </div>
    <div style="font-size: 1.5em;">👤</div>
</div>
"""


TIMESTAMP_DIV_TEMPLATE = """
<div style="font-size: {font_size}; font-weight: bold; text-align: center; width: 100%; height: 100%;">
    Time: {time:.2f} s
</div>
"""


# --- 1. 指示内容の構造化 (Target Parsing) ---


def parse_instruction_to_target(instruction_text):
    """
    自然言語の指示からターゲット（豆の種類と回数）を抽出する。
    対応パターン: "Put N scoops of [bean]...", "Put N ... and M ..."
    """
    targets = {"soybeans": 0, "red_beans": 0}

    # 小文字化して正規化
    text = instruction_text.lower()

    # 正規表現: 数字 + scoops? of + 豆の種類
    # のシナリオA, B, Cに対応
    patterns = [
        (r"(\d+)\s+scoops?\s+of\s+soybeans?", "soybeans"),
        (r"(\d+)\s+scoops?\s+of\s+red\s?beans?", "red_beans"),
    ]

    found = False
    for pattern, key in patterns:
        matches = re.findall(pattern, text)
        for count in matches:
            targets[key] += int(count)
            found = True

    # デフォルト（解析不能な場合）
    if not found:
        # 指示に明示的な数字がない場合は0のまま、あるいは手動設定を促すログを出す
        pass

    return targets


# --- 2. 評価管理クラス (Metrics Manager) ---


def load_existing_log(dataset_id, episode_index):
    """
    既存のログファイルから、指定されたデータセット・エピソードの最新の評価データを読み込む
    """
    filename = "outputs/snvla_evaluation_log.jsonl"
    path = Path(filename)
    if not path.exists():
        return None

    matching_records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if (
                        record.get("dataset_id") == dataset_id
                        and record.get("episode_index") == episode_index
                    ):
                        matching_records.append(record)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading log: {e}")
        return None

    # 最新のレコードを返す
    if matching_records:
        return matching_records[-1]
    return None


class EvaluationManager:
    def __init__(self, doc, dataset_id, episode_index, instruction_text, event_source=None):
        self.doc = doc
        self.dataset_id = dataset_id
        self.episode_index = episode_index
        self.instruction_text = instruction_text
        self.event_source = event_source

        # 目標値 (Target)
        self.targets = parse_instruction_to_target(instruction_text)

        # 実測値 (Actual) - アノテーション用
        self.actuals = {
            "soybeans": 0,
            "red_beans": 0,
            "spill": 0,  # こぼした回数
            "ghost": 0,  # 空振り回数
            "undo": 0,  # リカバリー回数
        }

        # UIコンポーネントの参照
        self.status_div = None

        # イベントログ
        self.events = []

        # 既存データのロード
        existing_record = load_existing_log(dataset_id, episode_index)
        if existing_record:
            if "actuals" in existing_record:
                # ターゲット構造が変わっている可能性もあるので、現在のキーのみ更新するなどが安全だが、
                # ここでは単純に上書きし、不足キーがあればデフォルト(0)のままにするなど考慮
                loaded_actuals = existing_record["actuals"]
                for k, v in loaded_actuals.items():
                    if k in self.actuals:
                        self.actuals[k] = v

            if "events" in existing_record:
                self.events = existing_record["events"]

            logging.info(f"Loaded existing evaluation data for episode {episode_index}")

    def increment(self, key, amount=1, frame_info=None):
        """カウントを増減させる"""
        if key in self.actuals:
            # 負の値にならないように制御（必要に応じて）
            if self.actuals[key] + amount >= 0:
                self.actuals[key] += amount

            # イベント記録
            if frame_info and amount > 0:
                event_record = {
                    "event_type": key,
                    "amount": amount,
                    "frame_index": frame_info["index"],
                }
                self.events.append(event_record)

            self.update_display()
            self.update_event_source()

    def undo_last_event(self):
        """直近のイベントを取り消す (全てのイベントタイプに対応)"""
        if not self.events:
            print("No events to undo.")
            return

        # 最後のイベントを取得して削除
        last_event = self.events.pop()
        event_type = last_event["event_type"]
        amount = last_event["amount"]

        # カウントを戻す
        if event_type in self.actuals:
            self.actuals[event_type] -= amount
            # 念のため0未満にならないように（論理的にはありえないが）
            if self.actuals[event_type] < 0:
                self.actuals[event_type] = 0

        logging.info(f"Undid event: {event_type} at frame {last_event['frame_index']}")

        self.update_display()
        self.update_event_source()

    def update_event_source(self):
        """イベントソースを更新してグラフに反映"""
        if not self.event_source:
            return

        indices = []
        colors = []
        types = []

        color_map = {
            "soybeans": "blue",
            "red_beans": "red",
            "spill": "orange",
            "ghost": "gray",
        }

        for evt in self.events:
            indices.append(evt["frame_index"])
            colors.append(color_map.get(evt["event_type"], "black"))
            types.append(evt["event_type"])

        # y位置はグラフの上部に見やすく配置 (e.g., 1.1)
        # 固定値でも良いし、イベントタイプごとに少しずらしても良い
        ys = [1.1] * len(indices)

        self.event_source.data = {
            "index": indices,
            "y": ys,
            "color": colors,
            "type": types,
        }

    def get_error(self):
        """カウント誤差 E を計算"""
        err_soy = abs(self.targets["soybeans"] - self.actuals["soybeans"])
        err_red = abs(self.targets["red_beans"] - self.actuals["red_beans"])
        return err_soy + err_red

    def update_display(self):
        """UIの表示を更新"""
        if self.status_div:
            # 達成状況の可視化
            soy_status = f"{self.actuals['soybeans']} / {self.targets['soybeans']}"
            red_status = f"{self.actuals['red_beans']} / {self.targets['red_beans']}"

            # 色付け: 完了なら緑、超過なら赤、不足なら青
            def colorize(act, tgt):
                if act == tgt:
                    return "color: green; font-weight: bold;"
                if act > tgt:
                    return "color: red; font-weight: bold;"
                return "color: blue;"

            html = f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
                <h4 style="margin-top: 0;">📊 Evaluation Metrics</h4>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <th>Class</th>
                        <th>Actual / Target</th>
                    </tr>
                    <tr>
                        <td>Soybeans</td>
                        <td style="{colorize(self.actuals["soybeans"], self.targets["soybeans"])}">{soy_status}</td>
                    </tr>
                    <tr>
                        <td>Red Beans</td>
                        <td style="{colorize(self.actuals["red_beans"], self.targets["red_beans"])}">{red_status}</td>
                    </tr>
                    <tr><td colspan="3"><hr></td></tr>
                    <tr>
                        <td>Errors</td>
                        <td>Spill: {self.actuals["spill"]} | Ghost: {self.actuals["ghost"]}</td>
                    </tr>
                     <tr>
                        <td><b>Total Error (E)</b></td>
                        <td><b>{self.get_error()}</b></td>
                    </tr>
                </table>
            </div>
            """
            self.status_div.text = html

    def export_data(self):
        """JSONL形式で保存"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset_id": self.dataset_id,
            "episode_index": self.episode_index,
            "instruction": self.instruction_text,
            "targets": self.targets,
            "actuals": self.actuals,
            "events": self.events,
            "calculated_error": self.get_error(),
            "success": self.get_error() == 0,  # 簡易判定
        }

        filename = "outputs/snvla_evaluation_log.jsonl"
        try:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved evaluation to {filename}")
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False


def load_data(dataset, episode_index):
    """Load data for a specific episode."""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # Use hf_dataset for fast scalar access
    # Slicing the dataset applies the transform (hf_transform_to_torch)
    # and returns a dict of stacked tensors/lists
    batch = dataset.hf_dataset[from_idx:to_idx]

    data = {
        "index": np.arange(to_idx - from_idx),
        "prob_bon": [],
        "prob_boa": [],
        "is_inference": [],
        "current_narration": [],
        "previous_narrations": [],
        "timestamp": [],
        "task_instruction": "",
    }

    # Helper to extract list from batch
    def get_list(key, default_val=0.0):
        if key in batch:
            val = batch[key]
            # If tensor, convert to list
            if hasattr(val, "tolist"):
                return val.tolist()
            # If list of tensors
            if isinstance(val, list) and len(val) > 0 and hasattr(val[0], "item"):
                return [v.item() for v in val]
            return val
        return [default_val] * (to_idx - from_idx)

    data["prob_bon"] = get_list("prob_bon")
    data["prob_boa"] = get_list("prob_boa")
    data["is_inference"] = [
        not (bon == 0 and boa == 0) for bon, boa in zip(data["prob_bon"], data["prob_boa"], strict=True)
    ]
    data["timestamp"] = get_list("real_timestamp", 0.0)
    if all(t == 0.0 for t in data["timestamp"]):
        data["timestamp"] = get_list("timestamp", 0.0)

    # Narrations
    raw_narrations = get_list("current_narration", "")

    prev_narrations = []
    current_accum = ""
    formatted_current = []

    for narr in raw_narrations:
        if narr is None:
            narr = ""
        # Format for display
        fmt = narr.replace("\n", "<span style='color: #aaa;'>↵</span><br>")

        prev_narrations.append(current_accum)
        formatted_current.append(fmt)
        current_accum += fmt

    data["current_narration"] = formatted_current
    data["previous_narrations"] = prev_narrations

    # Task instruction
    if "task" in batch:
        val = batch["task"]
        data["task_instruction"] = val[0] if len(val) > 0 else "Execute the task."
    elif "language_instruction" in batch:
        val = batch["language_instruction"]
        data["task_instruction"] = val[0] if len(val) > 0 else "Execute the task."
    elif "task_index" in batch:
        val = batch["task_index"]
        task_idx = val[0] if len(val) > 0 else 0
        if hasattr(task_idx, "item"):
            task_idx = task_idx.item()

        # Find task string from index
        tasks_df = dataset.meta.tasks
        # tasks_df has index=task_string, column="task_index"
        try:
            matched_tasks = tasks_df[tasks_df["task_index"] == task_idx].index
            if len(matched_tasks) > 0:
                data["task_instruction"] = matched_tasks[0]
            else:
                data["task_instruction"] = f"Unknown task index: {task_idx}"
        except Exception:
            data["task_instruction"] = "Execute the task."
    else:
        data["task_instruction"] = "Execute the task."

    # Handle tensor to string conversion if needed
    if hasattr(data["task_instruction"], "item"):
        data["task_instruction"] = data["task_instruction"].item()

    camera_keys = dataset.meta.camera_keys

    return data, camera_keys, from_idx


def process_frame(frame, camera_keys):
    images = {}
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
            images[key] = view
    return images


def create_visualization(doc):
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--root", type=str, default=None)

    # Parse only known args to avoid issues if bokeh injects others (though --args should isolate)
    args, _ = parser.parse_known_args(sys.argv[1:])

    repo_id = args.repo_id
    initial_episode_index = args.episode_index
    root = args.root
    if root == "None":
        root = None
    elif root is not None:
        root = Path(root)

    logging.info(f"Loading dataset: {repo_id}")

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        doc.add_root(Div(text=f"Error loading dataset: {e}"))
        return

    # State container
    state = {
        "data": None,
        "from_idx": 0,
        "episode_index": initial_episode_index,
        "num_frames": 0,
        "camera_keys": [],
    }

    # Initial load to get camera keys and setup plots
    data, camera_keys, from_idx = load_data(dataset, initial_episode_index)
    state.update(
        {
            "data": data,
            "from_idx": from_idx,
            "episode_index": initial_episode_index,
            "num_frames": len(data["index"]),
            "camera_keys": camera_keys,
        }
    )

    num_frames = state["num_frames"]

    # --- Data Sources ---
    # Add image sources
    # Load first frame to initialize dimensions
    first_frame = dataset[from_idx]
    first_images = process_frame(first_frame, camera_keys)

    image_sources = {}
    for key in camera_keys:
        image_sources[key] = ColumnDataSource(data={"image": [first_images[key]]})

    # Source for the full timeline (BON graph)
    # Filter only inference steps
    inference_indices = [i for i, is_inf in enumerate(data["is_inference"]) if is_inf]
    timeline_source = ColumnDataSource(
        data={
            "index": [data["index"][i] for i in inference_indices],
            "prob_bon": [data["prob_bon"][i] for i in inference_indices],
            "prob_boa": [data["prob_boa"][i] for i in inference_indices],
        }
    )

    # Source for points where BON > BOA
    bon_gt_boa_indices = [
        i for i, (bon, boa) in enumerate(zip(data["prob_bon"], data["prob_boa"], strict=True)) if bon > boa
    ]
    bon_gt_boa_source = ColumnDataSource(
        data={
            "index": bon_gt_boa_indices,
            "prob_bon": [data["prob_bon"][i] for i in bon_gt_boa_indices],
        }
    )

    # Source for events
    event_source = ColumnDataSource(data=dict(index=[], y=[], color=[], type=[]))

    # --- Evaluation Manager ---
    eval_manager = EvaluationManager(
        doc, repo_id, state["episode_index"], data["task_instruction"], event_source=event_source
    )

    # UIコンポーネント: 状態表示
    eval_status_div = Div(text="", sizing_mode="stretch_width")
    eval_manager.status_div = eval_status_div
    eval_manager.update_display()
    eval_manager.update_event_source()

    # UIコンポーネント: 操作ボタン群
    # --- Visual Feedback Div ---
    flash_div = Div(
        text="",
        sizing_mode="stretch_both",
        css_classes=["flash-overlay"],
        styles={
            "position": "fixed",
            "top": "0",
            "left": "0",
            "width": "100%",
            "height": "100%",
            "pointer-events": "none",
            "z-index": "9999",
            "box-shadow": "inset 0px 0px 0px 0px rgba(0,0,0,0)",
            # "transition": "box-shadow 0.1s ease-out", # Removed to allow instant ON in JS
        },
    )

    # --- Visual Feedback JS Helper ---
    def get_flash_js(color_rgba):
        return CustomJS(
            code=f"""
            // Helper to safely find element by class across Shadow DOMs if needed, though for root simple query suffices usually.
            // But Bokeh keeps changing DOM structure, so let's be robust.
            function findElementByClass(root, cls) {{
                const query = '.' + cls;
                let el = root.querySelector(query);
                if (el) return el;
                // Recursive check doesn't hurt
                const elements = root.querySelectorAll('*');
                for (const elem of elements) {{
                    if (elem.shadowRoot) {{
                        el = findElementByClass(elem.shadowRoot, cls);
                        if (el) return el;
                    }}
                }}
                return null;
            }}
            
            const el = findElementByClass(document, 'flash-overlay');
            if (el) {{
                // Instant ON: disable transition temporarily
                el.style.transition = 'none';
                el.style.boxShadow = "inset 0px 0px 100px 0px {color_rgba}";
                
                // Fade OUT: restore transition after a minimal delay
                // setTimeout ensures the browser repaints the 'Instant ON' state first
                requestAnimationFrame(() => {{
                    setTimeout(() => {{
                        el.style.transition = "box-shadow 0.3s";
                        el.style.boxShadow = "inset 0px 0px 0px 0px {color_rgba}";
                    }}, 1);
                }});
            }} else {{
                console.log("Flash overlay element not found");
            }}
        """,
        )

    # UIコンポーネント: 操作ボタン群
    # CSSクラスを追加してJSから参照しやすくする
    btn_soy = Button(label="Soybeans (+1)", button_type="primary", css_classes=["btn-soy"])
    btn_red = Button(label="Red Beans (+1)", button_type="danger", css_classes=["btn-red"])
    btn_spill = Button(label="Spill (+1)", button_type="warning", css_classes=["btn-spill"])
    btn_ghost = Button(label="Ghost Scoop (+1)", button_type="default", css_classes=["btn-ghost"])
    btn_undo = Button(label="Undo Last", button_type="default", css_classes=["btn-undo"])  # 誤操作修正用
    btn_save_eval = Button(label="💾 Save Metrics", button_type="success", css_classes=["btn-save"])

    # Attach Visual Feedback
    btn_soy.js_on_click(get_flash_js("rgba(0, 0, 255, 0.5)"))  # Blue
    btn_red.js_on_click(get_flash_js("rgba(255, 0, 0, 0.5)"))  # Red
    btn_save_eval.js_on_click(get_flash_js("rgba(0, 255, 0, 0.5)"))  # Green

    # コールバック定義
    def get_frame_info():
        current_idx = slider.value
        current_ts = state["data"]["timestamp"][current_idx]
        return {"index": current_idx, "timestamp": current_ts}

    def cb_soy():
        eval_manager.increment("soybeans", 1, frame_info=get_frame_info())

    def cb_red():
        eval_manager.increment("red_beans", 1, frame_info=get_frame_info())

    def cb_spill():
        eval_manager.increment("spill", 1, frame_info=get_frame_info())

    def cb_ghost():
        eval_manager.increment("ghost", 1, frame_info=get_frame_info())

    def cb_undo():
        eval_manager.undo_last_event()

    def cb_save_eval():
        if eval_manager.export_data():
            original_label = btn_save_eval.label
            btn_save_eval.label = "Saved! ✓"
            # Restore label after 2 seconds
            curdoc().add_next_tick_callback(
                lambda: time.sleep(0)
            )  # No-op just to access doc context if needed, but here we can just do nothing or use a callback
            # Note: time.sleep blocks event loop. Better to use add_timeout_callback if available, but for now simple label switch back is fine if we had it.
            # actually let's just leave it or reset it on next user action. But implementing a persistent reset is complex without more callbacks.
            # Simplified: just change label.

            # To reset label properly without blocking, we'd need another periodic callback or timeout.
            # For now, let's just leave "Saved! ✓" as explicit feedback or rely on the flash.
            pass

    btn_soy.on_click(cb_soy)
    btn_red.on_click(cb_red)
    btn_spill.on_click(cb_spill)
    btn_ghost.on_click(cb_ghost)
    btn_undo.on_click(cb_undo)
    btn_save_eval.on_click(cb_save_eval)

    # --- Components that need to be defined before Hotkey JS ---

    # 0. Episode Selector
    episode_select = Select(
        title="Episode:",
        value=str(initial_episode_index),
        options=[str(i) for i in range(dataset.num_episodes)],
        sizing_mode="stretch_width",
    )

    # ホットキー (Key Event) の実装 - JavaScriptを使用
    # ドキュメント全体でキーイベントをリッスンし、ボタンのclickをトリガー
    # CSSクラスを使用してボタンを特定する方式に変更（最も堅牢）
    hotkey_js = CustomJS(
        args=dict(episode_select=episode_select),
        code="""
        // 指定されたCSSクラスを持つボタンをクリックするヘルパー関数
        function clickCheckButton(cssClass) {
            // Shadow DOM対応: 再帰的にShadow Root内を検索
            function findElementByClass(root, cls) {
                const query = '.' + cls;
                // 通常の検索
                let el = root.querySelector(query);
                if (el) return el;

                // Shadow Rootを持つ要素を全て取得して再帰検索
                const elements = root.querySelectorAll('*');
                for (const elem of elements) {
                    if (elem.shadowRoot) {
                        el = findElementByClass(elem.shadowRoot, cls);
                        if (el) return el;
                    }
                }
                return null;
            }

            const wrapper = findElementByClass(document, cssClass);
            if (wrapper) {
                // Bokehのボタンはラッパーdivの中にある場合と、Shadow DOM内のbutton要素そのものの場合がある
                // CSSクラスはラッパーdivに付与されることが多い
                let btn = wrapper.shadowRoot ? wrapper.shadowRoot.querySelector('button') : wrapper.querySelector('button');

                // もしwrapper自体がbuttonなら
                if (!btn && wrapper.tagName === 'BUTTON') {
                    btn = wrapper;
                }

                if (btn) {
                    btn.click();
                    return true;
                } else {
                     // wrapper自体をクリックしてみる（一部のBokeh構成用）
                    wrapper.click();
                    return true;
                }
            }
            console.log("Button not found for class:", cssClass);
            return false;
        }

        // グローバルにキーイベントリスナーを追加（一度だけ）
        if (!window._hotkeyListenerAdded) {
            window._hotkeyListenerAdded = true;
            const hotkeys = {
                's': 'btn-soy',
                'r': 'btn-red',
                'x': 'btn-spill',
                'g': 'btn-ghost',
                'z': 'btn-undo',
                'w': 'btn-save',
            };
            document.addEventListener('keydown', function(e) {
                // テキスト入力中は無視
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                    return;
                }
                const key = e.key.toLowerCase();
                console.log("Key pressed: " + key);
                
                // Hotkeys mapped to buttons
                if (hotkeys[key]) {
                    clickCheckButton(hotkeys[key]);
                }
                
                // Play/Pause with Space
                if (e.key === " ") {
                    e.preventDefault(); // Prevent scrolling
                    clickCheckButton('btn-play');
                }
                
                // Episode Navigation
                if (e.key === "ArrowLeft") {
                    let current = parseInt(episode_select.value);
                    if (current > 0) {
                        episode_select.value = (current - 1).toString();
                    }
                } else if (e.key === "ArrowRight") {
                    let current = parseInt(episode_select.value);
                    const max_idx = episode_select.options.length - 1;
                    if (current < max_idx) {
                        episode_select.value = (current + 1).toString();
                    }
                }
            });
        }
    """,
    )
    curdoc().js_on_event("document_ready", hotkey_js)
    hotkey_div = Div(
        text="<small>Hotkeys: [Space]Play/Pause [S]oy [R]ed [X]spill [G]host [Z]undo [W]save</small>",
        sizing_mode="stretch_width",
    )

    # --- Components ---

    # 1. Chat Box
    instruction_div = Div(
        text=INSTRUCTION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            task_instruction=data["task_instruction"],
        ),
        sizing_mode="stretch_width",
        align="end",
    )

    # 2. Narration Box (Robot)
    # We combine previous and current narration into one chat-like interface
    narration_div = Div(
        text=NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=data["previous_narrations"][0],
            current_narration=data["current_narration"][0],
        ),
        sizing_mode="stretch_width",
    )

    # 2. Camera Views
    image_plots = []
    for key in camera_keys:
        # Get dimensions from the first frame
        h, w = first_images[key].shape
        p = figure(
            title=f"Camera: {key}",
            x_range=Range1d(0, w),
            y_range=Range1d(0, h),
            height=CONFIG.image_height,
            aspect_ratio=w / h,
            sizing_mode="scale_width",
            tools="",
        )
        p.image_rgba(image="image", x=0, y=0, dw=w, dh=h, source=image_sources[key])
        p.axis.visible = False
        p.grid.visible = False
        image_plots.append(p)

    # 3. BON Probability Plot
    bon_plot = figure(
        title="Timeline Analysis: Probabilities & Events",
        x_axis_label="Time Step",
        y_axis_label="Probability",
        sizing_mode="stretch_width",
        height=CONFIG.bon_plot_height,
        x_range=Range1d(0, num_frames),
        y_range=Range1d(-0.1, 1.2),
    )
    bon_plot.scatter(
        "index",
        "prob_bon",
        source=timeline_source,
        size=4,
        color=CONFIG.bon_line_color,
        legend_label="BON",
    )
    bon_plot.scatter(
        "index",
        "prob_boa",
        source=timeline_source,
        size=4,
        color=CONFIG.boa_line_color,
        legend_label="BOA",
    )
    # Plot points where BON > BOA
    bon_plot.scatter(
        "index",
        "prob_bon",
        source=bon_gt_boa_source,
        size=3,
        color="red",
        legend_label="BON > BOA",
    )

    # Plot Events
    bon_plot.scatter(
        "index",
        "y",
        source=event_source,
        size=10,
        color="color",
        marker="inverted_triangle",
        legend_label="Events",
    )

    bon_plot.legend.location = "top_left"
    bon_plot.legend.click_policy = "hide"
    bon_plot.add_layout(bon_plot.legend[0], "right")

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
        sizing_mode="stretch_width",
    )

    # Speed Slider
    speed_slider = Slider(
        start=0.1,
        end=10.0,
        value=1.0,
        step=0.1,
        title="Speed",
        sizing_mode="stretch_width",
    )

    # 5. Play/Export Controls
    play_button = Button(label="Play", button_type="success", css_classes=["btn-play"])
    real_time_checkbox = CheckboxGroup(labels=["Sync with Timestamp"], active=[0])
    loop_checkbox = CheckboxGroup(labels=["Loop Playback"], active=[0])  # Default: Loop enabled
    save_imgs_button = Button(label="Save Images", button_type="primary")

    # 6. Timestamp Display
    timestamp_div = Div(
        text=TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][0],
        ),
        sizing_mode="stretch_width",
        height=CONFIG.timestamp_height,
    )

    # --- Callbacks ---
    def update_frame(attr, old, new):
        idx = int(new)
        data = state["data"]
        from_idx = state["from_idx"]
        camera_keys = state["camera_keys"]

        # Update text
        prev = data["previous_narrations"][idx]
        curr = data["current_narration"][idx]
        narration_div.text = NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=prev,
            current_narration=curr,
        )

        # Update images
        frame = dataset[from_idx + idx]
        images = process_frame(frame, camera_keys)
        for key in camera_keys:
            image_sources[key].data = {"image": [images[key]]}

        # Update time line
        time_line.location = idx

        # Update timestamp
        timestamp_div.text = TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][idx],
        )

    slider.on_change("value", update_frame)

    def change_episode(attr, old, new):
        nonlocal eval_manager
        ep_idx = int(new)
        logging.info(f"Switching to Episode: {ep_idx}")

        # Stop playback if running
        if play_button.label == "Pause":
            toggle_play()

        # Load new data
        new_data, _, new_from_idx = load_data(dataset, ep_idx)

        # Update state
        state.update(
            {
                "data": new_data,
                "from_idx": new_from_idx,
                "episode_index": ep_idx,
                "num_frames": len(new_data["index"]),
            }
        )

        num_frames = state["num_frames"]
        data = state["data"]

        # Update sources
        # Update sources
        inference_indices = [i for i, is_inf in enumerate(data["is_inference"]) if is_inf]
        timeline_source.data = {
            "index": [data["index"][i] for i in inference_indices],
            "prob_bon": [data["prob_bon"][i] for i in inference_indices],
            "prob_boa": [data["prob_boa"][i] for i in inference_indices],
        }

        bon_gt_boa_indices = [
            i
            for i, (bon, boa) in enumerate(zip(data["prob_bon"], data["prob_boa"], strict=True))
            if bon > boa
        ]
        bon_gt_boa_source.data = {
            "index": bon_gt_boa_indices,
            "prob_bon": [data["prob_bon"][i] for i in bon_gt_boa_indices],
        }

        # Update UI components
        instruction_div.text = INSTRUCTION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            task_instruction=data["task_instruction"],
        )

        # Reset slider and ranges
        slider.end = num_frames - 1
        slider.value = 0
        bon_plot.x_range.end = num_frames

        # Eval Managerの再初期化
        eval_manager = EvaluationManager(
            doc, repo_id, ep_idx, new_data["task_instruction"], event_source=event_source
        )
        eval_manager.status_div = eval_status_div
        eval_manager.update_display()
        eval_manager.update_event_source()
        btn_save_eval.label = "💾 Save Metrics"

        # Trigger frame update for frame 0
        update_frame(None, None, 0)

    episode_select.on_change("value", change_episode)

    # State for real-time playback
    playback_state = {
        "start_wall_time": 0.0,
        "start_frame_time": 0.0,
        "accumulator": 0.0,
    }

    def reset_playback_anchors():
        """Reset the reference time points for smooth speed changing."""
        idx = slider.value
        data = state["data"]
        # Ensure we are within bounds just in case
        if 0 <= idx < len(data["timestamp"]):
            current_ts = data["timestamp"][idx]
            playback_state["start_wall_time"] = time.time()
            playback_state["start_frame_time"] = current_ts

    speed_slider.on_change("value", lambda attr, old, new: reset_playback_anchors())

    def animate_update():
        nonlocal callback_id
        current_idx = slider.value
        data = state["data"]
        num_frames = state["num_frames"]

        # Check if "Real Speed" is active
        is_real_speed = 0 in real_time_checkbox.active
        # logging.info(f"Animate: RealSpeed={is_real_speed}, Speed={speed_slider.value}, CurrentIdx={current_idx}")

        if is_real_speed:
            speed = speed_slider.value
            now = time.time()
            elapsed_wall = now - playback_state["start_wall_time"]
            target_time = playback_state["start_frame_time"] + (elapsed_wall * speed)

            # Find the frame index corresponding to target_time
            # data["timestamp"] is expected to be sorted
            next_idx = bisect.bisect_left(data["timestamp"], target_time)

            # bisect returns insertion point, ensure we don't go out of bounds
            if next_idx >= num_frames:
                if 0 in loop_checkbox.active:
                    next_idx = 0
                    # Loop around: reset reference times
                    playback_state["start_wall_time"] = time.time()
                    playback_state["start_frame_time"] = data["timestamp"][0]
                else:
                    # Stop playback
                    next_idx = num_frames - 1
                    toggle_play()

            if slider.value != next_idx:
                slider.value = next_idx

        else:
            # Standard frame-by-frame playback (with speed support)
            speed = speed_slider.value
            playback_state["accumulator"] += speed
            step = int(playback_state["accumulator"])

            if step >= 1:
                playback_state["accumulator"] -= step
                frame = current_idx + step

                if frame >= num_frames:
                    if 0 in loop_checkbox.active:
                        frame = 0
                        playback_state["start_wall_time"] = time.time()
                        playback_state["start_frame_time"] = data["timestamp"][0]
                        playback_state["accumulator"] = 0.0
                    else:
                        frame = num_frames - 1
                        toggle_play()
                else:
                    target_time = data["timestamp"][frame]
                    playback_state["start_wall_time"] = time.time() - (
                        target_time - playback_state["start_frame_time"]
                    )
                slider.value = frame

    callback_id = None

    def toggle_play():
        nonlocal callback_id
        data = state["data"]
        if play_button.label == "Play":
            play_button.label = "Pause"

            # Initialize playback state for real-time mode
            playback_state["start_wall_time"] = time.time()
            playback_state["accumulator"] = 0.0
            # Handle potential None or missing timestamp gracefully, though we expect valid floats
            current_ts = data["timestamp"][slider.value]
            playback_state["start_frame_time"] = current_ts

            callback_id = doc.add_periodic_callback(animate_update, CONFIG.animation_interval_ms)
        else:
            play_button.label = "Play"
            if callback_id:
                with contextlib.suppress(ValueError):
                    doc.remove_periodic_callback(callback_id)
                callback_id = None

    def save_images():
        episode_index = state["episode_index"]
        from_idx = state["from_idx"]
        camera_keys = state["camera_keys"]

        save_dir = Path.cwd() / f"snvla_episode_{episode_index}_images"
        save_dir.mkdir(parents=True, exist_ok=True)

        frame = dataset[from_idx + slider.value]
        images = process_frame(frame, camera_keys)

        for key in camera_keys:
            img_array = images[key]
            h, w = img_array.shape

            # Convert back to RGBA uint8
            img_rgba = img_array.view(dtype=np.uint8).reshape((h, w, 4))
            # Flip vertically back
            img_rgba = np.flipud(img_rgba)

            from PIL import Image

            # img_rgba = np.array(
            #     Image.fromarray(img_rgba, mode="RGBA").resize((512, 512), Image.Resampling.LANCZOS)
            # )
            img_pil = Image.fromarray(img_rgba, mode="RGBA")
            img_pil.save(save_dir / f"{key}_frame_{slider.value:05d}.png")

        logging.info(f"Saved images to {save_dir}")

    play_button.on_click(toggle_play)
    save_imgs_button.on_click(save_images)

    # --- Layout ---
    controls = row(
        play_button,
        real_time_checkbox,
        loop_checkbox,
        slider,
        speed_slider,
        save_imgs_button,
        sizing_mode="stretch_width",
    )

    # Evaluation Panel
    eval_panel = column(
        Div(text="<h3>📝 Annotation Controls</h3>"),
        row(btn_soy, btn_red),
        row(btn_spill, btn_ghost),
        row(btn_undo, btn_save_eval),
        eval_status_div,
        hotkey_div,
        sizing_mode="stretch_width",
    )

    main_layout = column(
        episode_select,
        row(
            column(
                row(image_plots, sizing_mode="stretch_width"),
                bon_plot,
                timestamp_div,
                controls,
                sizing_mode="stretch_width",
            ),
            column(
                instruction_div,
                narration_div,
                # eval_panel,
                sizing_mode="stretch_width",
                max_width=600,
            ),
            sizing_mode="stretch_width",
            # toolbar_options={"logo": None},
        ),
        sizing_mode="stretch_width",
    )

    # Hide Bokeh logo
    hide_logo_css = InlineStyleSheet(css=".bk-logo { display: none !important; }")
    doc.add_root(flash_div)
    doc.add_root(main_layout)
    for model in doc.roots:
        model.stylesheets.append(hide_logo_css)
    doc.title = "SNVLA Evaluation Visualizer"


# To run this script:
# bokeh serve examples/snvla/vse.py --args --repo-id <repo_id> --episode-index <idx>

create_visualization(curdoc())
