from unittest.mock import MagicMock

import pytest
import torch

# テスト対象のモジュールをインポート
from lerobot.policies.snvla.configuration_snvla import SNVLAConfig
from lerobot.policies.snvla.modeling_snvla import SNVLAPolicy
from lerobot.policies.snvla.processor_snvla import (
    CURRENT_NARRATION,
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    PREVIOUS_NARRATIONS,
    TASK_KEY,
    SNVLAPrepareTrainingTokenizerProcessorStep,
)

# lerobotのコアコンポーネントのダミー定義
# 本来はlerobotライブラリからインポートしますが、ここではテストに必要な最小限を定義します。
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    COMPLEMENTARY_DATA,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# --- テスト用の設定と共通コンポーネント (Fixtures) ---


@pytest.fixture
def test_config() -> SNVLAConfig:
    """テスト用の小さなSNVLAConfigインスタンスを作成します。"""
    config = SNVLAConfig()
    # テストが高速に終わるように設定を調整
    config.tokenizer_max_length = 64
    config.max_state_dim = 8
    config.max_action_dim = 4
    config.n_action_steps = 2
    config.chunk_size = 2
    config.compile_model = False  # テスト中はコンパイルしない
    config.device = "cpu"
    # output_featuresのダミー設定
    config.output_features = {ACTION: MagicMock(shape=(4,))}
    return config


@pytest.fixture
def dummy_transition(test_config: SNVLAConfig) -> EnvTransition:
    """プロセッサテスト用のダミーEnvTransitionを作成します。"""
    transition = EnvTransition()
    transition[TransitionKey.OBSERVATION] = {OBS_STATE: torch.rand(test_config.max_state_dim)}
    transition[TransitionKey.ACTION] = torch.rand(test_config.max_action_dim)
    transition[TransitionKey.COMPLEMENTARY_DATA] = {
        TASK_KEY: "pick up the red block",
        CURRENT_NARRATION: "approaching the block",
        PREVIOUS_NARRATIONS: ["first, I see a block", "now I will move my arm"],
    }
    return transition


@pytest.fixture
def dummy_inference_batch(test_config: SNVLAConfig) -> dict:
    """行動選択テスト用のダミーバッチデータを作成します。"""
    batch_size = 1
    return {
        # ダミー画像 (C, H, W)
        OBS_IMAGES: [torch.rand(3, 224, 224).unsqueeze(0)],
        OBS_STATE: torch.rand(batch_size, test_config.max_state_dim),
        COMPLEMENTARY_DATA: {TASK_KEY: ["pick up the red block"]},
    }


# --- テストケース ---


def test_processor_step(test_config: SNVLAConfig, dummy_transition: EnvTransition):
    """
    【プロセッサのテスト】
    SNVLAPrepareTrainingTokenizerProcessorStepが、入力Transitionを正しく
    トークン化し、必要なマスクを生成できるかを確認します。
    """
    # transitionの内容を確認
    print(dummy_transition)

    # 1. テスト対象のプロセッサを初期化
    processor = SNVLAPrepareTrainingTokenizerProcessorStep(config=test_config)

    # 2. プロセッサを実行
    processed_transition = processor(dummy_transition)
    processed_obs = processed_transition[TransitionKey.OBSERVATION]

    # 3. 結果を検証
    # 3.1. 必要なキーがすべて存在することを確認
    expected_keys = [
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKEN_AR_MASK,
        OBS_LANGUAGE_TOKEN_LOSS_MASK,
    ]
    for key in expected_keys:
        assert key in processed_obs, f"キー '{key}' がプロセッサの出力にありません"

    # 出力の内容を確認
    print(processed_obs)

    # 3.2. 出力の形状が正しいことを確認
    max_len = test_config.tokenizer_max_length
    assert processed_obs[OBS_LANGUAGE_TOKENS].shape == (max_len,), "トークンの形状が不正です"
    assert processed_obs[OBS_LANGUAGE_ATTENTION_MASK].shape == (max_len,), "Attentionマスクの形状が不正です"
    assert processed_obs[OBS_LANGUAGE_TOKEN_AR_MASK].shape == (max_len,), "ARマスクの形状が不正です"
    assert processed_obs[OBS_LANGUAGE_TOKEN_LOSS_MASK].shape == (max_len,), "Lossマスクの形状が不正です"

    # 3.3. Lossマスクのロジックが正しいことを確認（プレフィックス部分が0、サフィックス部分が1）
    # このテストケースでは、"Next: <bon>approaching the block<eos>" がサフィックス
    # トークナイザによって正確な位置は変動するが、大まかに後半が1になっているはず
    loss_mask = processed_obs[OBS_LANGUAGE_TOKEN_LOSS_MASK].int()
    assert torch.sum(loss_mask) > 0, "Lossマスクがすべて0です"
    assert torch.sum(loss_mask) < max_len, "Lossマスクがすべて1です"
    print("\nProcessor Test Passed!")


def test_forward_pass(test_config: SNVLAConfig):
    """
    【モデルのフォワードパステスト】
    SNVLAPolicyのforwardメソッドが、学習用のバッチデータを通してエラーなく
    損失を計算し、逆伝播が可能であることを確認します。
    """
    # 1. モデルを初期化
    policy = SNVLAPolicy(config=test_config)
    policy.train()  # 学習モードに設定

    # 2. ダミーの学習用バッチデータを作成
    batch_size = 2
    max_len = test_config.tokenizer_max_length
    batch = {
        OBS_IMAGES: [torch.rand(batch_size, 3, 224, 224)],
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (batch_size, max_len)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, max_len, dtype=torch.bool),
        OBS_LANGUAGE_TOKEN_AR_MASK: torch.zeros(batch_size, max_len, dtype=torch.bool),
        OBS_LANGUAGE_TOKEN_LOSS_MASK: torch.ones(batch_size, max_len, dtype=torch.bool),
        ACTION: torch.rand(batch_size, test_config.chunk_size, test_config.max_action_dim),
    }

    # 3. フォワードパスを実行
    loss, info = policy.forward(batch)

    # 4. 結果を検証
    # 4.1. 損失が計算されていることを確認
    assert isinstance(loss, torch.Tensor), "損失がTensorではありません"
    assert loss.requires_grad, "損失が勾配計算可能ではありません"
    assert "text_loss" in info and "action_loss" in info, "info辞書に必要な損失情報が含まれていません"

    # 4.2. 逆伝播（バックプロパゲーション）がエラーなく実行できることを確認
    try:
        loss.backward()
    except Exception as e:
        pytest.fail(f"loss.backward() でエラーが発生しました: {e}")

    # 4.3. パラメータに勾配が計算されていることを確認
    assert any(p.grad is not None for p in policy.model.parameters()), (
        "どのパラメータにも勾配が計算されていません"
    )
    print("Forward Pass Test Passed!")


def test_select_action(test_config: SNVLAConfig, dummy_inference_batch: dict):
    """
    【行動選択のテスト】
    select_actionメソッドが、推論用の観測データからエラーなく行動を
    生成できること、またアクションキューが機能していることを確認します。
    """
    # 1. モデルを初期化
    policy = SNVLAPolicy(config=test_config)
    policy.eval()  # 評価モードに設定

    # 2. 状態をリセット
    policy.reset()
    assert len(policy._action_queue) == 0
    assert not policy._previous_narrations

    # 3. 行動選択を実行
    # `_act`が呼ばれることを確認するためにモック化
    policy._act = MagicMock(wraps=policy._act)

    action = policy.select_action(dummy_inference_batch)

    # 4. 結果を検証
    # 4.1. 最初の行動が正しく生成されたか
    assert isinstance(action, torch.Tensor), "行動がTensorではありません"
    original_action_dim = test_config.output_features[ACTION].shape[0]
    assert action.shape == (original_action_dim,), f"行動の形状が不正です: {action.shape}"

    # 4.2. アクションキューが満たされたか
    # n_action_steps(2) - 1(popleft) = 1
    assert len(policy._action_queue) == test_config.n_action_steps - 1, "アクションキューの長さが不正です"

    # 4.3. _actが1回だけ呼ばれたことを確認
    policy._act.assert_called_once()

    # 5. 2回目の行動選択を実行
    _ = policy.select_action(dummy_inference_batch)

    # 6. アクションキューが機能していることを検証
    # 6.1. キューから取得されるため、_actは再度呼ばれない
    policy._act.assert_called_once()
    assert len(policy._action_queue) == 0, "アクションキューが空になっていません"

    # 7. 状態のリセットを検証
    policy.reset()
    assert len(policy._action_queue) == 0, "リセット後、アクションキューが空になっていません"
    print("Select Action Test Passed!")
