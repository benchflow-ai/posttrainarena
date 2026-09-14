from types import SimpleNamespace

import pytest
import torch

from posttrainarena.benchflow_pipeline.segmented_trainer import (
    segmented_grpo_trainer_class,
)


def _segment(prompt_ids, completion_ids, logprobs):
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
    )


class FakeAccelerator:
    num_processes = 2

    @staticmethod
    def gather(value):
        return torch.stack([value, value + 2])

    @staticmethod
    def backward(loss):
        loss.backward()


class FakeBaseTrainer:
    loss_type = "dapo"
    importance_sampling_level = "token"
    use_vllm = True
    vllm_importance_sampling_correction = True
    vllm_importance_sampling_mode = "token_truncate"
    vllm_importance_sampling_clip_min = 0.5
    vllm_importance_sampling_clip_max = 2.0
    beta = 0.0

    def __init__(self, rollouts):
        self.rollout_func = SimpleNamespace(last_rollouts=rollouts)
        self._tokenizer = SimpleNamespace(pad_token_id=99)
        self.accelerator = FakeAccelerator()
        self.model = SimpleNamespace(version=3.0)
        self.forward_rows = []

    def _generate_and_score_completions(self, inputs):
        count = len(inputs)
        return {
            "prompt_ids": torch.zeros((count, 1), dtype=torch.long),
            "prompt_mask": torch.ones((count, 1), dtype=torch.long),
            "completion_ids": torch.zeros((count, 1), dtype=torch.long),
            "completion_mask": torch.ones((count, 1), dtype=torch.long),
            "advantages": torch.tensor([1.5, -0.5]),
            "num_items_in_batch": torch.tensor(2),
        }

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size,
        compute_entropy=False,
    ):
        assert batch_size == input_ids.size(0) == 1
        self.forward_rows.append(attention_mask[:, -logits_to_keep:].sum().item())
        shape = (input_ids.size(0), logits_to_keep)
        return torch.full(shape, model.version), torch.zeros(shape), None

    def _prepare_inputs(self, generation_batch):
        return generation_batch


def test_segment_cache_is_generation_time_and_ids_select_shuffled_rollouts() -> None:
    rollouts = [
        SimpleNamespace(tokens=SimpleNamespace(segments=[_segment([1], [10], [2.5])])),
        SimpleNamespace(
            tokens=SimpleNamespace(
                segments=[
                    _segment([2, 3], [11, 12], [2.0, 2.0]),
                    _segment([4], [13], [2.0]),
                ]
            )
        ),
    ]
    trainer = segmented_grpo_trainer_class(FakeBaseTrainer)(rollouts)
    generated = trainer._generate_and_score_completions([{}, {}])

    assert generated["segment_rollout_ids"].tolist() == [0, 1]
    assert trainer._segment_cache[0]["old_per_token_logps"].tolist() == [[3.0, 0.0]]
    assert trainer.forward_rows == [1, 2, 1, 0, 0]
    assert trainer._segment_cache[1]["advantages"].tolist() == [-0.5, -0.5]
    assert trainer._segment_cache[0]["num_items_in_batch"].item() == 10

    trainer.model.version = 9.0
    shuffled = {**generated, "segment_rollout_ids": torch.tensor([1])}
    prepared = trainer._prepare_inputs(shuffled)

    assert prepared["prompt_ids"].tolist() == [[2, 3], [99, 4]]
    assert prepared["completion_ids"].tolist() == [[11, 12], [13, 99]]
    assert prepared["advantages"].tolist() == [-0.5, -0.5]
    assert prepared["old_per_token_logps"].tolist() == [[3.0, 3.0], [3.0, 0.0]]
    assert prepared["num_items_in_batch"].item() == 10
    assert set(prepared) == {
        "prompt_ids",
        "prompt_mask",
        "completion_ids",
        "completion_mask",
        "sampling_per_token_logps",
        "old_per_token_logps",
        "importance_sampling_ratio",
        "advantages",
        "num_items_in_batch",
    }


def test_sequence_mask_uses_one_ratio_across_original_rollout() -> None:
    rollouts = [
        SimpleNamespace(
            tokens=SimpleNamespace(
                segments=[
                    _segment([1], [10], [2.9]),
                    _segment([2], [11, 12], [2.8, 2.7]),
                ]
            )
        ),
        SimpleNamespace(tokens=SimpleNamespace(segments=[_segment([3], [13], [3.0])])),
    ]
    trainer = segmented_grpo_trainer_class(FakeBaseTrainer)(rollouts)
    trainer.vllm_importance_sampling_mode = "sequence_mask"

    trainer._generate_and_score_completions([{}, {}])

    first = trainer._segment_cache[0]
    active_ratios = first["importance_sampling_ratio"][first["completion_mask"].bool()]
    expected = torch.exp(torch.tensor(0.6)).item()
    assert active_ratios.tolist() == pytest.approx([expected] * 3)
    second = trainer._segment_cache[1]
    assert second["importance_sampling_ratio"][
        second["completion_mask"].bool()
    ].tolist() == pytest.approx([1.0])


def test_segmented_trainer_rejects_incompatible_loss() -> None:
    trainer = segmented_grpo_trainer_class(FakeBaseTrainer)([])
    trainer.loss_type = "grpo"

    try:
        trainer._generate_and_score_completions([])
    except RuntimeError as error:
        assert "only DAPO" in str(error)
    else:
        raise AssertionError("incompatible loss was accepted")


@pytest.mark.parametrize("sentinel", [None, "non-None value to disable scaling"])
def test_chunked_backward_matches_reference_and_counter_increments_once(
    sentinel,
) -> None:
    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.2))

        def forward(self, token_ids):
            return self.weight * token_ids.float()

    class ScalarBase(FakeBaseTrainer):
        def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size,
            compute_entropy=False,
        ):
            assert batch_size == 1
            self.forward_rows.append(attention_mask[:, -logits_to_keep:].sum().item())
            current = model(input_ids[:, -logits_to_keep:])
            return current, torch.zeros_like(current), None

    rollouts = [
        SimpleNamespace(
            tokens=SimpleNamespace(segments=[_segment([1], [1, 2], [0.2, 0.4])])
        ),
        SimpleNamespace(tokens=SimpleNamespace(segments=[_segment([2], [3], [0.6])])),
    ]
    trainer = segmented_grpo_trainer_class(ScalarBase)(rollouts)
    trainer.model = ScalarModel()
    trainer.args = SimpleNamespace(
        delta=None, n_gpu=1, optim="adamw", torch_empty_cache_steps=None
    )
    trainer.current_gradient_accumulation_steps = 2
    trainer.model_accepts_loss_kwargs = False
    trainer.compute_loss_func = sentinel
    trainer._step = 0
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    generated = trainer._generate_and_score_completions([{}, {}])
    cached = trainer._segment_cache[0]["old_per_token_logps"].clone()
    with torch.no_grad():
        trainer.model.weight.add_(0.01)
    trainer.forward_rows.clear()

    trainer.training_step(trainer.model, generated)

    # Fake remote rank reports local token count +2: whole-generation count=8.
    reference_weight = torch.tensor(0.21, requires_grad=True)
    tokens = torch.tensor([1.0, 2.0, 3.0])
    advantages = torch.tensor([1.5, 1.5, -0.5])
    ratio = torch.exp(reference_weight * tokens - 0.2 * tokens)
    reference = -torch.minimum(
        ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
    ).sum() / (8 / 2)
    if sentinel is None:
        reference = reference / 2
    reference.backward()
    assert torch.allclose(trainer.model.weight.grad, reference_weight.grad)
    assert trainer.forward_rows == [2, 1, 0, 0]
    assert trainer._step == 1
    assert torch.equal(trainer._segment_cache[0]["old_per_token_logps"], cached)


def test_real_grpo_initialization_disables_transformers_accumulation_scaling(
    tmp_path,
) -> None:
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "x": 3}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    args = GRPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        report_to="none",
        use_vllm=False,
        loss_type="dapo",
        num_generations=2,
        generation_batch_size=2,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
        max_completion_length=2,
    )
    trainer = segmented_grpo_trainer_class(GRPOTrainer)(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"prompt": ["x", "x"]}),
        reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
    )
    assert trainer.model_accepts_loss_kwargs is False
    assert trainer.compute_loss_func == "non-None value to disable scaling"
    assert trainer.accelerator.gradient_accumulation_steps == 1
