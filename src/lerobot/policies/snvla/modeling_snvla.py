import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.policies.pi05.modeling_pi05 import (
    PaliGemmaWithExpertModel,
    PI05Policy,
    PI05Pytorch,
    get_gemma_config,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.utils.constants import (
    ACTION,
    COMPLEMENTARY_DATA,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .configuration_snvla import SNVLAConfig
from .processor_snvla import TASK_KEY, TOKEN_AR_MASK_KEY, TOKEN_LOSS_MASK_KEY, make_prefix_prompt


class SNVLACore(nn.Module):
    """Self-Narrating Vision-Language-Action (SN-VLA) core model."""

    gradient_checkpointing_enable = PI05Pytorch.gradient_checkpointing_enable
    gradient_checkpointing_disable = PI05Pytorch.gradient_checkpointing_disable
    _apply_checkpoint = PI05Pytorch._apply_checkpoint
    _prepare_attention_masks_4d = PI05Pytorch._prepare_attention_masks_4d
    sample_noise = PI05Pytorch.sample_noise
    sample_time = PI05Pytorch.sample_time
    embed_prefix = PI05Pytorch.embed_prefix
    embed_suffix = PI05Pytorch.embed_suffix
    denoise_step = PI05Pytorch.denoise_step

    def __init__(self, config: SNVLAConfig):
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.diffusion_loss_coeff = config.diffusion_loss_coeff

        self.gradient_checkpointing_enabled = False

    def forward(
        self,
        images,
        img_masks,
        language_tokens,
        language_padding_masks,
        language_attention_masks,
        actions,
        language_loss_masks,
        diffusion_loss_masks,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embeddings
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, language_tokens, language_padding_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        # Attention Masks
        prefix_att_masks[:, -language_attention_masks.shape[1] :] = language_attention_masks

        full_ar_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        full_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)

        att_2d_masks = make_att_2d_masks(full_pad_masks, full_ar_masks)
        position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Single Forward Pass
        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Loss Calculation

        # Action Loss (L_action)
        suffix_out_actions = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out_actions)
        action_loss_raw = F.mse_loss(u_t, v_t, reduction="none")
        # diffusion_loss_mask: (B,) -> (B, 1, 1)
        action_loss = (action_loss_raw * diffusion_loss_masks.view(-1, 1, 1)).mean()

        # Text Loss (L_narration)
        txt_logits = self.paligemma_with_expert.paligemma.lm_head(prefix_out)

        # ターゲットとロジットをシフト
        txt_targets = language_tokens[:, 1:]
        txt_logits = txt_logits[:, :-1]

        txt_loss_raw = F.cross_entropy(
            txt_logits.transpose(1, 2),  # (B, L, V) -> (B, V, L)
            txt_targets,
            reduction="none",
        )

        valid_loss_mask = language_loss_masks[:, 1:].float()
        txt_loss = (txt_loss_raw * valid_loss_mask).sum() / valid_loss_mask.sum().clamp(min=1)

        # Total Loss
        loss = txt_loss + self.diffusion_loss_coeff * action_loss

        info = {
            "loss": loss.item(),
            "text_loss": txt_loss.item(),
            "action_loss": action_loss.item(),
        }
        return loss, info


class SNVLAPolicy(PI05Policy):
    """SN-VLA Policy for LeRobot."""

    config_class = SNVLAConfig
    name = "snvla"

    def __init__(self, config: SNVLAConfig):
        # `PI05Policy` の __init__ を意図的にスキップ。`PreTrainedPolicy` の __init__ を呼び出す
        super(PI05Policy, self).__init__(config)
        config.validate_features()
        self.config = config

        self.model = SNVLACore(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        if config.compile_model:
            logging.info("Compiling SN-VLA inference steps...")
            self._prefill = torch.compile(self._prefill, mode=config.compile_mode, dynamic=True)
            self._narrate_step = torch.compile(self._narrate_step, mode=config.compile_mode, dynamic=True)
            self._act = torch.compile(self._act, mode=config.compile_mode, dynamic=True)

        self.reset()

    def reset(self):
        """Reset internal state - called when environment resets."""
        super().reset()  # `_action_queue` を初期化

        self._previous_narrations = []
        self._kv_cache = None

    def _discretize_state(self, state: torch.Tensor) -> str:
        state = pad_vector(state, self.config.max_state_dim)
        # 推論時、状態は正規化済みと仮定
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        # (B, D) -> B個の文字列。推論はB=1を仮定
        return " ".join(map(str, discretized_states[0]))

    def _build_prompt_and_tokenize(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # 初期指示はバッチから取得 (B=1を仮定 for inference)
        task = batch[COMPLEMENTARY_DATA][TASK_KEY][0]
        state = batch[OBS_STATE]

        state_str = self._discretize_state(state)

        prompt = make_prefix_prompt(task, self._previous_narrations, state_str)

        token_data = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
        )
        return {
            "input_ids": token_data["input_ids"].to(self.model.device),
            "attention_mask": token_data["attention_mask"].to(self.model.device).bool(),
        }

    @torch.no_grad()
    def _prefill(self, images, img_masks, tokens, masks) -> tuple[Tensor, Any]:
        """Runs the prefix (images + text history) to get KV cache and next-token logits."""
        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )

        # Build attention
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Run VLM forward
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), kv_cache = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # プレフィックスのみ
            use_cache=True,
        )

        # Get last non-padding token's logits
        # `prefix_position_ids` の最後の値が、パディングを除いた最後のトークンのインデックス
        last_token_idx = prefix_position_ids[:, -1]
        eop_pre_logit = prefix_out[torch.arange(prefix_out.shape[0]), last_token_idx]
        eop_logits = self.model.paligemma_with_expert.paligemma.lm_head(eop_pre_logit).unsqueeze(
            1
        )  # (B, 1, V)

        return eop_logits, kv_cache

    def _decide_mode(self, logits: Tensor) -> Tensor:
        """Decide the next mode (action or narration) from output logits."""

        valid_tokens = torch.tensor(
            [self.config.begin_of_action_token_id, self.config.begin_of_narration_token_id],
            device=logits.device,
        )
        valid_mask = torch.full_like(logits, -torch.inf)
        valid_mask[:, :, valid_tokens] = 0.0

        mode_logits = logits + valid_mask

        if self.config.narration_temperature > 0.0:
            probs = F.softmax(mode_logits / self.config.narration_temperature, dim=-1)
            mode_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
        else:
            mode_token = torch.argmax(mode_logits, dim=-1)

        return mode_token.view(-1, 1)

    @torch.no_grad()
    def _narrate_step(
        self, token: Tensor, kv_cache: tuple[tuple[Tensor]] | None
    ) -> tuple[Tensor, Tensor, Any]:
        """Performs a single autoregressive decoding step for narration generation."""

        token_embedding = self.model.paligemma_with_expert.paligemma.language_model.embed_tokens(token)

        (last_pre_logit, _), new_kv_cache = self.model.paligemma_with_expert.forward(
            attention_mask=None,
            position_ids=None,
            past_key_values=kv_cache,
            inputs_embeds=[token_embedding, None],
            use_cache=True,
        )

        new_logits = self.model.paligemma_with_expert.paligemma.lm_head(last_pre_logit)

        if self.config.narration_temperature > 0.0:
            probs = F.softmax(new_logits / self.config.narration_temperature, dim=-1)
            new_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
        else:
            new_token = torch.argmax(new_logits, dim=-1)

        return new_token.view(-1, 1), new_logits, new_kv_cache

    @torch.no_grad()
    def _act(self, kv_cache: Any, prefix_pad_masks: Tensor, bsize: int):
        """Generates an action chunk using the diffusion model."""

        # `BEGIN_OF_ACTION` トークンをフォワード
        device = self.model.device
        action_token = torch.full(
            (bsize, 1), self.config.begin_of_action_token_id, dtype=torch.long, device=device
        )
        action_emb = self.model.paligemma_with_expert.paligemma.language_model.embed_tokens(action_token)

        (_, _), act_kv_cache = self.model.paligemma_with_expert.forward(
            attention_mask=None,
            position_ids=None,
            past_key_values=kv_cache,
            inputs_embeds=[action_emb, None],
            use_cache=True,
        )

        # プレフィックスマスクにboaトークン分(1)を追加
        act_prefix_pad_masks = F.pad(prefix_pad_masks, (0, 1), value=True)

        # 拡散モデルのサンプリング
        num_steps = self.config.num_inference_steps
        dt = -1.0 / num_steps

        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.model.sample_noise(actions_shape, device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        # denoise_step loop
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            v_t = self.model.denoise_step(
                prefix_pad_masks=act_prefix_pad_masks,
                past_key_values=act_kv_cache,
                x_t=x_t,
                timestep=expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        actions = x_t

        # アクションをキューに追加
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions_unpadded = actions[:, : self.config.n_action_steps, :original_action_dim]

        self._action_queue.extend(actions_unpadded.transpose(0, 1))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # アクションキュー確認
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # 観測の準備
        images, img_masks = self._preprocess_images(batch)

        # 動的トークン化
        token_data = self._build_prompt_and_tokenize(batch)
        tokens = token_data["input_ids"]
        masks = token_data["attention_mask"]

        if self._kv_cache is None:
            logits, self._kv_cache = self._prefill(images, img_masks, tokens, masks)
        else:
            logging.warning("Reusing KV cache without reset it. This should not happen in normal flow.")
            logits, self._kv_cache = self._prefill(images, img_masks, tokens, masks)

        # モード決定
        mode = self._decide_mode(logits)
        current_token = mode

        # 実況ループ
        while mode.item() == self.config.begin_of_narration_token_id:
            logging.info("SN-VLA starting narration generation...")

            generated_tokens = []
            for _step in range(self.config.max_narration_length):
                # KVキャッシュを更新しながら1ステップデコード
                new_token, logits, self._kv_cache = self._narrate_step(current_token, self._kv_cache)
                current_token = new_token

                if new_token.item() == self.config.eos_token_id:
                    break
                generated_tokens.append(new_token.item())

            # 実況履歴を更新
            new_narration = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            self._previous_narrations.append(new_narration)
            logging.info(f"SN-VLA Narrated: {new_narration}")

            # 次のモードを決定
            mode = self._decide_mode(logits)
            current_token = mode

        # 行動生成
        bsize = images[0].shape[0]
        prefix_pad_masks = masks
        self._act(self.config, self._kv_cache, prefix_pad_masks, bsize)

        # KVキャッシュリセット
        self._kv_cache = None

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model for training."""

        images, img_masks = self._preprocess_images(batch)

        language_tokens = batch[OBS_LANGUAGE_TOKENS]
        language_attention_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        language_ar_masks = batch[TOKEN_AR_MASK_KEY]
        language_loss_masks = batch[TOKEN_LOSS_MASK_KEY]

        actions = self.prepare_action(batch)

        # 拡散損失マスク (データセットから来ると仮定)
        diffusion_loss_masks = batch.get("diffusion_loss_mask", torch.ones_like(actions[:, 0, 0]))

        return self.model.forward(
            images=images,
            img_masks=img_masks,
            language_tokens=language_tokens,
            language_padding_masks=language_attention_masks,
            language_attention_masks=language_ar_masks,
            actions=actions,
            language_loss_masks=language_loss_masks,
            diffusion_loss_masks=diffusion_loss_masks,
        )
