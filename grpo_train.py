"""Continue the causality SFT adapter with GRPO on news, tweets, and reports.

Model initialization
    BASE_MODEL_PATH inherits the base checkpoint configured in sft_train.py.
    SFT_LORA_PATH points to outputs/Causality_LoRA/final, including its saved
    tokenizer. Paths/model class/chat settings inherit the SFT configuration
    by default, and saved SFT metadata is checked before use. The SFT adapter
    is loaded with is_trainable=True and its existing
    parameters are optimized further. No randomly initialized replacement
    adapter is created, and the original SFT directory is not overwritten.

    With BETA > 0, TRL must snapshot the initial SFT adapter as a frozen 'ref'
    adapter. The script checks that its parameters equal the initial SFT
    policy and freezes them before training. Older TRL versions that fall back
    to the raw base model are rejected. On resume, GRPO checkpoint weights and
    optimizer state are restored after initializing from the same SFT adapter.

Data and prompts
    DATA_CONFIG names the three stage-1 FinAtlas_<text type>_Causality_GRPO.parquet
    files. Their required columns are text/Text, Chain_Complexity, and
    Causal_Paths. This consumes the raw 2:1/overflow corpus directly; no separate
    'GRPO_Ready' conversion is required. Majority labels are pseudo-labels.
    Invalid causal targets and overlong prompts are excluded before sampling.
    Values in DATA_CONFIG are per-source training sample caps; None uses all
    eligible rows. They are separate from the stage-1 GRPO collection capacity.
    The sampled News, Tweets, and Reports rows are combined and shuffled with
    SEED before one joint GRPO training run on the SFT adapter.

    Prompts reuse the causal definitions/blueprint from cot_process.py and the
    system instruction from sft_train.py. The tokenizer saved with SFT renders
    them. Targets are kept in separate reward columns and never added to the
    prompt. MULTIPLE requires a sequential chain with at least two explicit
    mechanisms; independent branches remain separate SINGLE paths.

Rewards
    Total reward = FORMAT_REWARD_WEIGHT * format score + CAUSALITY_REWARD_WEIGHT
    * causality score. Both component scores lie in [0, 1]. Format requires one
    non-empty <thinking> block followed by <output> containing valid target JSON.
    Causality requires a valid path structure and the correct complexity label;
    it then combines label correctness and normalized triad-set Jaccard overlap.
    NONE with an empty path list receives full task reward when the target is
    NONE. Text grounding is not established by this rule-based reward.

Editable settings and hardware
    Paths, sampling caps, epochs/steps, learning rate, batch/accumulation, group
    size, generation lengths, temperature, reward weights, KL coefficient,
    clipping, precision, and checkpoints are grouped below. LoRA rank and target
    modules are inherited from the SFT adapter, so edit them in sft_train.py.
    EPOCHS defaults to 8 for GRPO; MAX_STEPS > 0 overrides it.

    The experimental hardware is one A800 80 GB. BATCH_SIZE * GRAD_ACCUMULATION
    * GPU process count gives the completion batch per optimizer update, which
    must be divisible by GROUP_SIZE. GROUP_SIZE completions are generated per
    distinct prompt. The defaults 1 * 4 * 1 = 4 with GROUP_SIZE=4 correspond to
    one prompt group per update. Adjust memory use with batch size, group size,
    and prompt/completion lengths. Use a positive temperature for group diversity.

Output and dependencies
    OUTPUT_DIR contains GRPO checkpoints/metrics; OUTPUT_DIR/final contains the
    trained default LoRA adapter and tokenizer. This adapter includes both the
    SFT initialization and GRPO updates and loads on BASE_MODEL_PATH. A small
    training_sources.json records initialization and training settings.
    RESUME_FROM_CHECKPOINT='auto' resumes the latest checkpoint in OUTPUT_DIR;
    None starts fresh; a path resumes that checkpoint. Use separate output
    folders for different initialization models or experiments.

    Keep this file beside three_model_consensus.py, cot_process.py, and
    sft_train.py. Dependencies: torch, transformers, peft, datasets, trl,
    accelerate, pyarrow. Use compatible releases supporting Qwen3.5 and TRL's
    pretrained-PEFT reference snapshot. No torch.load/security/FSDP patches are
    applied. Reference: https://huggingface.co/docs/trl/grpo_trainer
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

from cot_process import build_inference_prompt, normalize_record
from sft_train import (
    ATTENTION_IMPLEMENTATION as SFT_ATTENTION_IMPLEMENTATION,
    CHAT_TEMPLATE_KWARGS as SFT_CHAT_TEMPLATE_KWARGS,
    LOCAL_FILES_ONLY as SFT_LOCAL_FILES_ONLY,
    MODEL_CLASS as SFT_MODEL_CLASS,
    MODEL_PATH as SFT_MODEL_PATH,
    OUTPUT_DIR as SFT_OUTPUT_DIR,
    PRECISION as SFT_PRECISION,
    SEED as SFT_SEED,
    SYSTEM_PROMPT,
    TRUST_REMOTE_CODE as SFT_TRUST_REMOTE_CODE,
)
from three_model_consensus import OUTPUT_DIR as ALLOCATION_DIR, PROJECT_DIR, validate_causal_paths

# ---------------------------------------------------------------------------
# USER CONFIGURATION: edit these settings before running the script.
# ---------------------------------------------------------------------------

# Same base model checkpoint used in the SFT stage.
BASE_MODEL_PATH = SFT_MODEL_PATH
# REQUIRED trained SFT adapter and tokenizer; GRPO continues these exact weights.
SFT_LORA_PATH = SFT_OUTPUT_DIR / "final"
# Model loader matching the base checkpoint architecture, as in sft_train.py.
MODEL_CLASS = SFT_MODEL_CLASS
# Restrict base/tokenizer loading to local or cached files.
LOCAL_FILES_ONLY = SFT_LOCAL_FILES_ONLY
# Allow custom implementation code if required by the selected checkpoint.
TRUST_REMOTE_CODE = SFT_TRUST_REMOTE_CODE
# Attention backend supported by the model and PyTorch installation.
ATTENTION_IMPLEMENTATION = SFT_ATTENTION_IMPLEMENTATION
# Separate GRPO checkpoint/output directory; never use the SFT adapter directory.
OUTPUT_DIR = PROJECT_DIR / "outputs" / "Causality_GRPO"
# "auto" resumes the latest GRPO checkpoint; None starts fresh; a path resumes it.
RESUME_FROM_CHECKPOINT = "auto"

# All three sources are combined and shuffled for one joint GRPO training run.
# Per-source training sample limits, applied after validity/length filtering.
# Defaults to 1,000 rows per source; adjust dataset sizes as needed for experiments.
# If fewer eligible rows are available, all available rows are used.
# Set a limit to None to use all eligible rows from that source.
DATA_CONFIG = {
    # Financial news GRPO corpus produced by the consensus stage.
    ALLOCATION_DIR / "FinAtlas_News_Causality_GRPO.parquet": 1000,
    # Financial tweet GRPO corpus produced by the same pipeline for tweets.
    ALLOCATION_DIR / "FinAtlas_Tweets_Causality_GRPO.parquet": 1000,
    # Financial report GRPO corpus produced by the same pipeline for reports.
    ALLOCATION_DIR / "FinAtlas_Reports_Causality_GRPO.parquet": 1000,
}

# Number of GRPO training passes; independent of the SFT epoch setting.
EPOCHS = 8
# Positive values override EPOCHS with a fixed optimizer-update budget; -1 uses epochs.
MAX_STEPS = -1
# Learning rate for the existing SFT LoRA parameters during GRPO.
LEARNING_RATE = 5e-6
# Completion sequences per GPU/process per training microbatch; tune for memory.
BATCH_SIZE = 1
# Microbatches per optimizer update; on one GPU, 1 * 4 = 4 completion sequences.
GRAD_ACCUMULATION = 4
# Completions per prompt; the global completion batch must be divisible by this.
GROUP_SIZE = 4
# Maximum rendered prompt tokens; longer texts are skipped, not silently truncated.
MAX_PROMPT_LENGTH = 6144
# Maximum generated tokens per completion, covering rationale and causal JSON.
MAX_COMPLETION_LENGTH = 2048
# Positive sampling temperature to produce diverse completions within a group.
TEMPERATURE = 1.0
# Nucleus sampling threshold; 1.0 leaves the full distribution available.
TOP_P = 1.0
# Seed for sampling input records and initializing the training run.
SEED = SFT_SEED

# Relative weight of the [0, 1] format-validity reward.
FORMAT_REWARD_WEIGHT = 0.2
# Relative weight of the [0, 1] causal-complexity/path reward.
CAUSALITY_REWARD_WEIGHT = 0.8
# Share of the task score assigned to a correct complexity label.
COMPLEXITY_REWARD_SHARE = 0.5
# Share of the task score assigned to triad overlap when the label is correct.
PATH_REWARD_SHARE = 0.5
# KL penalty against the frozen initial SFT policy; zero disables this penalty.
BETA = 0.04
# GRPO probability-ratio clipping parameter.
EPSILON = 0.2

# Training precision; bf16 requires a compatible GPU.
PRECISION = SFT_PRECISION
# Reduce activation memory by recomputing activations in the backward pass.
GRADIENT_CHECKPOINTING = True
# Optimizer implementation understood by TRL/Transformers.
OPTIMIZER = "adamw_torch"
# Learning-rate schedule.
LR_SCHEDULER = "constant"
# Learning-rate warmup optimizer updates; zero disables warmup.
WARMUP_STEPS = 0
# Use Transformers generation by default; True requires a compatible vLLM setup.
USE_VLLM = False
# Exclude cut-off completions from the policy loss.
MASK_TRUNCATED_COMPLETIONS = True
# OpenMP threads; applied before importing the training runtime.
OMP_NUM_THREADS = 8
# PyTorch CUDA allocator settings, applied before importing PyTorch.
CUDA_ALLOCATOR_CONFIG = "expandable_segments:True"
# Checkpoint saving interval, measured in optimizer updates.
SAVE_STEPS = 30
# Maximum retained GRPO checkpoints.
SAVE_TOTAL_LIMIT = 3
# Training logging interval, measured in optimizer updates.
LOGGING_STEPS = 2
# Disable external reporting services by default.
REPORT_TO = "none"
# Match the SFT chat-template configuration; FinCoT remains in the generated content.
CHAT_TEMPLATE_KWARGS = SFT_CHAT_TEMPLATE_KWARGS.copy()

LOGGER = logging.getLogger(__name__)


def completion_text(completion: Any) -> str:
    """Handle both plain-text and conversational completions from TRL."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(
            message.get("content", "")
            for message in completion
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        )
    return ""


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys")
        result[key] = value
    return result


def parse_completion(completion: Any) -> dict[str, Any]:
    """Require the same wrappers and JSON schema used during causality SFT."""
    text = completion_text(completion).strip()
    match = re.fullmatch(
        r"<thinking>\s*(.*?)\s*</thinking>\s*<output>\s*(.*?)\s*</output>",
        text,
        re.DOTALL,
    )
    if not match or not match.group(1).strip():
        raise ValueError("Missing rationale or output block")
    if re.search(r"</?(?:thinking|output)\b", match.group(1)):
        raise ValueError("Nested or repeated wrappers")
    parsed = json.loads(match.group(2), object_pairs_hook=unique_json_object)
    if not isinstance(parsed, dict) or set(parsed) != {"Chain_Complexity", "Causal_Paths"}:
        raise ValueError("Invalid target JSON keys")
    if parsed["Chain_Complexity"] not in {"NONE", "SINGLE", "MULTIPLE"}:
        raise ValueError("Invalid complexity label")
    if not isinstance(parsed["Causal_Paths"], list):
        raise ValueError("Causal_Paths must be a list")
    for triad in parsed["Causal_Paths"]:
        if not isinstance(triad, dict) or set(triad) != {"Cause", "Mechanism", "Effect"}:
            raise ValueError("Invalid triad keys")
        if not all(isinstance(value, str) and value.strip() for value in triad.values()):
            raise ValueError("Invalid triad values")
    return parsed


def format_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards = []
    for completion in completions:
        try:
            parse_completion(completion)
            rewards.append(1.0)
        except (ValueError, TypeError):
            rewards.append(0.0)
    return rewards


def triad_set(paths: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    """Compare wording with case/whitespace normalization, without semantic guessing."""
    return {
        tuple(" ".join(path[key].casefold().split()) for key in ("Cause", "Mechanism", "Effect"))
        for path in paths
    }


def causality_reward_func(
    completions: list[Any], Chain_Complexity: list[str], Causal_Paths: list[str], **kwargs: Any
) -> list[float]:
    if not (len(completions) == len(Chain_Complexity) == len(Causal_Paths)):
        raise ValueError("Reward targets must align with generated completions")
    rewards = []
    for completion, label, raw_paths in zip(completions, Chain_Complexity, Causal_Paths):
        try:
            parsed = parse_completion(completion)
            validate_causal_paths(parsed["Chain_Complexity"], parsed["Causal_Paths"])
            if parsed["Chain_Complexity"] != label:
                rewards.append(0.0)
                continue
            target_paths = json.loads(raw_paths)
            predicted = triad_set(parsed["Causal_Paths"])
            target = triad_set(target_paths)
            union = predicted | target
            overlap = len(predicted & target) / len(union) if union else 1.0
            rewards.append(COMPLEXITY_REWARD_SHARE + PATH_REWARD_SHARE * overlap)
        except (ValueError, TypeError, KeyError):
            rewards.append(0.0)
    return rewards


def prepare_grpo_row(row: dict[str, Any], tokenizer: Any) -> dict[str, str] | None:
    record = normalize_record(row)
    # Use the exact same user prompt builder as the CoT -> SFT stage.
    user_prompt = build_inference_prompt(record["text"])
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    if len(tokenizer(rendered, add_special_tokens=False)["input_ids"]) > MAX_PROMPT_LENGTH:
        return None
    return {
        "prompt": rendered,
        "Chain_Complexity": record["Chain_Complexity"],
        "Causal_Paths": json.dumps(record["Causal_Paths"], ensure_ascii=False),
    }


def load_training_data(tokenizer: Any) -> Any:
    from datasets import Dataset, concatenate_datasets, load_dataset

    rng = random.Random(SEED)
    sources = []
    for path, cap in DATA_CONFIG.items():
        source = load_dataset("parquet", data_files=str(path), split="train")
        rows = []
        invalid = overlong = 0
        for index, row in enumerate(source, 1):
            try:
                prepared = prepare_grpo_row(row, tokenizer)
                if prepared is None:
                    overlong += 1
                else:
                    rows.append(prepared)
            except (ValueError, TypeError, SyntaxError):
                invalid += 1
                LOGGER.debug("Invalid target excluded: %s row %s", path.name, index)
        rng.shuffle(rows)
        if cap is not None:
            rows = rows[:cap]
        LOGGER.info("%s: sampled %s; invalid %s; overlong %s", path.name, len(rows), invalid, overlong)
        if rows:
            sources.append(Dataset.from_list(rows))
    if not sources:
        raise ValueError("No eligible GRPO samples remain; check targets and MAX_PROMPT_LENGTH")
    return concatenate_datasets(sources).shuffle(seed=SEED)


def build_training_arguments(config_class: Any) -> Any:
    return config_class(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        num_generations=GROUP_SIZE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        beta=BETA,
        epsilon=EPSILON,
        reward_weights=[FORMAT_REWARD_WEIGHT, CAUSALITY_REWARD_WEIGHT],
        remove_unused_columns=False,
        bf16=PRECISION == "bf16",
        fp16=PRECISION == "fp16",
        optim=OPTIMIZER,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if GRADIENT_CHECKPOINTING else None,
        use_vllm=USE_VLLM,
        mask_truncated_completions=MASK_TRUNCATED_COMPLETIONS,
        sync_ref_model=False,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        report_to=REPORT_TO,
        seed=SEED,
        data_seed=SEED,
    )


def verify_sft_reference(model: Any, torch: Any) -> None:
    """Prevent KL regularization against a raw base instead of the SFT policy."""
    model.set_adapter("default")
    if BETA == 0:
        return
    if "ref" not in model.peft_config:
        raise RuntimeError("TRL must snapshot the pretrained SFT adapter as 'ref'; install a compatible TRL/PEFT release")
    checked = 0
    for name, parameter in model.named_parameters():
        if ".default." in name:
            reference = model.get_parameter(name.replace(".default.", ".ref."))
            if not torch.equal(parameter.detach(), reference.detach()):
                raise RuntimeError("The initial reference adapter does not match the SFT policy")
            reference.requires_grad_(False)
            checked += 1
    if not checked:
        raise RuntimeError("No SFT adapter parameters found for reference verification")
    if any(parameter.requires_grad and ".default." not in name for name, parameter in model.named_parameters()):
        raise RuntimeError("Frozen SFT reference requires adapter-only training; shared base parameters must stay frozen")
    LOGGER.info("Verified %s frozen reference tensors against the initial SFT policy", checked)


def validate_config() -> None:
    if not (SFT_LORA_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(f"SFT adapter is missing: {SFT_LORA_PATH}; run sft_train.py first")
    if not (SFT_LORA_PATH / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Use the tokenizer saved alongside the SFT adapter")
    metadata_path = SFT_LORA_PATH / "training_sources.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("stage") != "sft":
            raise ValueError("SFT_LORA_PATH must point to the output of sft_train.py")
        saved_base = metadata.get("base_model")
        if saved_base != BASE_MODEL_PATH and Path(str(saved_base)).resolve() != Path(BASE_MODEL_PATH).resolve():
            raise ValueError("GRPO base model differs from the base recorded by SFT")
        if metadata.get("model_class") != MODEL_CLASS:
            raise ValueError("GRPO model loader differs from the loader recorded by SFT")
        if metadata.get("system_prompt") != SYSTEM_PROMPT or metadata.get("chat_template_kwargs") != CHAT_TEMPLATE_KWARGS:
            raise ValueError("GRPO chat settings differ from the settings recorded by SFT")
    if OUTPUT_DIR.resolve() == SFT_LORA_PATH.resolve():
        raise ValueError("GRPO output must differ from the SFT adapter directory")
    if not DATA_CONFIG:
        raise ValueError("DATA_CONFIG must contain GRPO input files")
    for path, cap in DATA_CONFIG.items():
        if path.suffix.lower() != ".parquet" or not path.is_file():
            raise FileNotFoundError(f"Required GRPO Parquet file is missing: {path}")
        if cap is not None and (not isinstance(cap, int) or cap < 0):
            raise ValueError("Training sample caps must be non-negative integers or None")
    if not any(cap is None or cap > 0 for cap in DATA_CONFIG.values()):
        raise ValueError("At least one data source must have a positive or unlimited cap")
    if min(BATCH_SIZE, GRAD_ACCUMULATION, MAX_PROMPT_LENGTH, MAX_COMPLETION_LENGTH, OMP_NUM_THREADS) < 1:
        raise ValueError("Batch, accumulation, lengths and thread count must be positive")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if GROUP_SIZE < 2 or (BATCH_SIZE * GRAD_ACCUMULATION * world_size) % GROUP_SIZE:
        raise ValueError("Global completion batch must be divisible by GROUP_SIZE, which must be at least two")
    if EPOCHS <= 0 or MAX_STEPS == 0 or MAX_STEPS < -1 or LEARNING_RATE <= 0:
        raise ValueError("Invalid epochs, maximum steps, or learning rate")
    if TEMPERATURE <= 0 or not 0 < TOP_P <= 1:
        raise ValueError("GRPO requires a positive temperature and TOP_P in (0, 1]")
    if PRECISION not in {"bf16", "fp16", "fp32"} or BETA < 0 or EPSILON <= 0:
        raise ValueError("Invalid precision, KL coefficient, or clipping setting")
    if MODEL_CLASS not in {"AutoModelForCausalLM", "AutoModelForImageTextToText"}:
        raise ValueError("MODEL_CLASS must match a supported checkpoint architecture")
    if min(FORMAT_REWARD_WEIGHT, CAUSALITY_REWARD_WEIGHT, COMPLEXITY_REWARD_SHARE, PATH_REWARD_SHARE) < 0:
        raise ValueError("Reward weights must be non-negative")
    if abs(FORMAT_REWARD_WEIGHT + CAUSALITY_REWARD_WEIGHT - 1.0) > 1e-8:
        raise ValueError("Format and causality reward weights must sum to one")
    if abs(COMPLEXITY_REWARD_SHARE + PATH_REWARD_SHARE - 1.0) > 1e-8:
        raise ValueError("Complexity and path reward shares must sum to one")
    if min(SAVE_STEPS, SAVE_TOTAL_LIMIT, LOGGING_STEPS) < 1 or WARMUP_STEPS < 0:
        raise ValueError("Invalid checkpoint, logging, or warmup settings")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_config()
    os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOCATOR_CONFIG

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    if PRECISION in {"bf16", "fp16"} and not torch.cuda.is_available():
        raise RuntimeError("The configured mixed precision requires a compatible CUDA GPU")
    if PRECISION == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This device does not support the configured bf16 precision")
    set_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(
        str(SFT_LORA_PATH), trust_remote_code=TRUST_REMOTE_CODE, local_files_only=LOCAL_FILES_ONLY
    )
    if not tokenizer.chat_template:
        raise ValueError("The SFT tokenizer must contain a chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The SFT tokenizer needs a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = load_training_data(tokenizer)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[PRECISION]
    base_model = getattr(transformers, MODEL_CLASS).from_pretrained(
        BASE_MODEL_PATH, dtype=dtype, attn_implementation=ATTENTION_IMPLEMENTATION,
        trust_remote_code=TRUST_REMOTE_CODE, local_files_only=LOCAL_FILES_ONLY,
    )
    # Continue SFT weights directly; creating a fresh LoRA here would discard them.
    model = PeftModel.from_pretrained(base_model, str(SFT_LORA_PATH), is_trainable=True)
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
        model.config.text_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    trainer = GRPOTrainer(
        model=model,
        args=build_training_arguments(GRPOConfig),
        reward_funcs=[format_reward_func, causality_reward_func],
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    verify_sft_reference(model, torch)
    checkpoint = RESUME_FROM_CHECKPOINT
    if checkpoint == "auto":
        checkpoint = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.is_dir() else None
    elif checkpoint is not None:
        checkpoint = str(checkpoint)
    LOGGER.info("Continuing SFT adapter %s with %s GRPO prompts; resume: %s", SFT_LORA_PATH, len(dataset), checkpoint)
    result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save only the trained policy adapter; the frozen 'ref' adapter is not the policy.
    if trainer.is_world_process_zero():
        final_dir = OUTPUT_DIR / "final"
        policy = trainer.accelerator.unwrap_model(trainer.model)
        policy.save_pretrained(str(final_dir), selected_adapters=["default"], safe_serialization=True)
        tokenizer.save_pretrained(str(final_dir))
        provenance = {
            "stage": "grpo",
            "base_model": BASE_MODEL_PATH,
            "model_class": MODEL_CLASS,
            "system_prompt": SYSTEM_PROMPT,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "sft_adapter": str(SFT_LORA_PATH.resolve()),
            "grpo_sources": {str(path): cap for path, cap in DATA_CONFIG.items()},
            "group_size": GROUP_SIZE,
            "epochs": EPOCHS,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "beta": BETA,
            "epsilon": EPSILON,
            "reward_weights": [FORMAT_REWARD_WEIGHT, CAUSALITY_REWARD_WEIGHT],
            "seed": SEED,
        }
        (final_dir / "training_sources.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    LOGGER.info("Final causality GRPO adapter and tokenizer: %s", OUTPUT_DIR / "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
