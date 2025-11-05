from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.processor_pi05 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    CURRENT_NARRATION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    PREVIOUS_NARRATIONS,
)

from .configuration_snvla import SNVLAConfig

# 学習データセットが提供するキー
TASK_KEY = "task"


def make_prefix_prompt(task: str, previous_narrations: list[str], state_str: str) -> str:
    """Constructs the prefix prompt for SN-VLA."""
    narration_history = "\n".join(s.strip() for s in previous_narrations)
    if narration_history:
        narration_history = f"History: {narration_history}\n"

    prefix_str = f"Task: {task.strip()}\n{narration_history}State: {state_str};\nNext: "
    return prefix_str


@ProcessorStepRegistry.register(name="snvla_prepare_training_tokenizer_processor_step")
@dataclass
class SNVLAPrepareTrainingTokenizerProcessorStep(ProcessorStep):
    """Processor step for SN-VLA training."""

    config: SNVLAConfig
    tokenizer: Any = field(init=False)

    max_state_dim: int = 32

    task_key: str = TASK_KEY

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        self.begin_of_narration_token = self.tokenizer.convert_ids_to_tokens(
            self.config.begin_of_narration_token_id
        )
        self.begin_of_action_token = self.tokenizer.convert_ids_to_tokens(
            self.config.begin_of_action_token_id
        )

    def _discretize_state(self, state: torch.Tensor) -> str:
        """Discretizes state vector into a string."""
        state = pad_vector(state, self.max_state_dim)
        state_np = state.cpu().numpy()
        # 状態はNormalizerProcessorStepによって[-1, 1]に正規化済みと仮定
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_states))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for SN-VLA")

        task = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if task is None:
            raise ValueError(f"'{self.task_key}' not found in complementary data.")

        current_narration = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(CURRENT_NARRATION)
        if current_narration is None:
            raise ValueError(f"'{CURRENT_NARRATION}' (ground-truth) not found.")

        previous_narrations = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(PREVIOUS_NARRATIONS)
        if previous_narrations is None:
            raise ValueError(f"'{PREVIOUS_NARRATIONS}' (ground-truth) not found.")

        state_str = self._discretize_state(state)
        previous_narrations = previous_narrations.split("\n")

        # プレフィックス: コンテキスト
        prefix_str = make_prefix_prompt(task, previous_narrations, state_str)

        # サフィックス: 予測ターゲット
        current_narration = current_narration.strip()
        if current_narration:
            # ナレーション生成モード
            suffix_str = (
                f"{self.begin_of_narration_token}{current_narration.strip()}{self.tokenizer.eos_token}"
            )
        else:
            # 行動生成モード
            suffix_str = f"{self.begin_of_action_token}"

        prefix_data = self.tokenizer(
            prefix_str,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=False,  # 最大長は後で全体に適用
        )
        suffix_data = self.tokenizer(
            suffix_str,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=False,
        )

        input_ids = prefix_data["input_ids"] + suffix_data["input_ids"]
        attention_mask = prefix_data["attention_mask"] + suffix_data["attention_mask"]

        # ARマスクを作成: プレフィックス(0)は相互参照可, サフィックス(1)は自己回帰
        token_ar_mask = [0] * len(prefix_data["input_ids"]) + [1] * len(suffix_data["input_ids"])

        # 損失マスクを作成: サフィックス部分のみでテキスト損失を計算
        token_loss_mask = [0] * len(prefix_data["input_ids"]) + [1] * len(suffix_data["input_ids"])

        # 全体をパディング
        max_len = self.config.tokenizer_max_length
        pad_len = max_len - len(input_ids)

        if pad_len < 0:
            # トークンが長すぎる場合は切り捨て
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            token_ar_mask = token_ar_mask[:max_len]
            token_loss_mask = token_loss_mask[:max_len]
        else:
            # パディング
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            token_ar_mask += [0] * pad_len
            token_loss_mask += [0] * pad_len

        # Transitionオブジェクトに書き戻す
        obs = transition.get(TransitionKey.OBSERVATION, {})
        obs[OBS_LANGUAGE_TOKENS] = torch.tensor(input_ids, dtype=torch.long)
        obs[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(attention_mask, dtype=torch.bool)
        obs[OBS_LANGUAGE_TOKEN_AR_MASK] = torch.tensor(token_ar_mask, dtype=torch.bool)
        obs[OBS_LANGUAGE_TOKEN_LOSS_MASK] = torch.tensor(token_loss_mask, dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step adds the custom mask features.
        """
        # (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK はTokenizerProcessorStepと互換)
        max_len = self.config.tokenizer_max_length
        features["observation"][OBS_LANGUAGE_TOKEN_AR_MASK] = PolicyFeature(
            type=FeatureType.STATE, shape=(max_len,)
        )
        features["observation"][OBS_LANGUAGE_TOKEN_LOSS_MASK] = PolicyFeature(
            type=FeatureType.STATE, shape=(max_len,)
        )
        return features


def make_snvla_pre_post_processors(
    config: SNVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SN-VLA policy.

    The pre-processor is used **only for training** and uses
    `SNVLAPrepareTrainingTokenizerProcessorStep`.

    Inference (`select_action`) bypasses this and uses the internal tokenizer.
    """

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        SNVLAPrepareTrainingTokenizerProcessorStep(config=config, max_state_dim=config.max_state_dim),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
