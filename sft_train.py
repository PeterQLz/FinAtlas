"""Train a causality LoRA adapter from FinAtlas's three FinCoT SFT datasets.

Task and input
    This is stage 3 of the Causal Relationship pipeline, after consensus
    allocation and cot_process.py. It combines English financial news,
    tweets, and reports from the three configured .parquet files. Each row
    must contain prompt and response strings; legacy Prompt/Response column
    names are also supported. All configured files are required. Extra columns
    are ignored, blank pairs are excluded, and sources are combined without
    resampling before a shuffle with SEED=42.

Editable configuration
    All routine settings are grouped below the imports. Set MODEL_PATH to a
    local Hugging Face checkpoint folder, DATA_PATHS to the three SFT files,
    and OUTPUT_DIR to the training output folder. Default paths are anchored
    to the folder containing these scripts. To use a Hub model ID, set
    LOCAL_FILES_ONLY=False. Change MODEL_CLASS and LoRA target module names
    to match the replacement model's architecture.

    EPOCHS, LEARNING_RATE, BATCH_SIZE, GRAD_ACCUMULATION, MAX_SEQ_LENGTH,
    precision, optimizer, scheduler, LoRA settings, and checkpoint settings
    are editable. Defaults are 10 epochs, a 2e-5 learning rate, batch size 2,
    accumulation 8, and an 8,192-token sequence limit.

    The experiments used one NVIDIA A800 with 80 GB of GPU memory. BATCH_SIZE
    is the number of samples processed simultaneously on that GPU, while
    GRAD_ACCUMULATION is the number of microbatches accumulated before each
    optimizer update. Effective batch size is BATCH_SIZE * GRAD_ACCUMULATION
    * GPU process count, so the current single-GPU settings give 2 * 8 = 16.
    Adjust both settings for the actual model, sequence lengths, precision,
    and available memory; these defaults are not a measured memory guarantee.
    If memory is insufficient, batch size 1 with accumulation 16 preserves an
    effective batch of 16. If memory permits, batch size 4 with accumulation 4
    or batch size 8 with accumulation 2 also gives 16. More accumulation uses
    sequential microbatches rather than holding the full effective batch at once.

Training behavior
    The tokenizer's chat template formats system/user/assistant messages.
    Only the assistant response and its closing tokens contribute to loss;
    prompt and padding labels are -100. CHAT_TEMPLATE_KWARGS disables Qwen's
    native thinking prefill because FinCoT is already inside the response.
    This setting does not remove the dataset's <thinking> rationale.

    OVERLENGTH_POLICY='skip' preserves complete responses by excluding long
    samples. 'truncate' restores right truncation, which may cut the response.
    Samples with no remaining supervised response tokens are always excluded.
    Increase MAX_SEQ_LENGTH when reports exceed the configured context budget,
    subject to the model's context window and available memory.

    Training uses Transformers Trainer + PEFT LoRA, with dynamic right padding
    and optional gradient checkpointing. Trainer handles device placement for
    single-GPU or standard distributed training. The base model is frozen.
    No validation split or benchmark evaluation is performed by this script.

Output
    OUTPUT_DIR contains training checkpoints and metrics. OUTPUT_DIR/final
    contains the final LoRA adapter, tokenizer, and training_sources.json. The
    metadata records the base model and chat settings for the GRPO handoff.
    It does not contain merged base-model weights.
    Set RESUME_FROM_CHECKPOINT to a checkpoint path, or True for the latest
    checkpoint; None starts a new run. No API keys are needed for local training.

Dependencies: Python 3.10+; torch, datasets, transformers, peft, accelerate,
and pyarrow. Use a Transformers release that supports the selected model.
Model architecture reference: https://huggingface.co/Qwen/Qwen3.5-9B
Chat templates: https://huggingface.co/docs/transformers/chat_templating
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

from cot_process import OUTPUT_PARQUET as COT_OUTPUT_PARQUET
from three_model_consensus import PROJECT_DIR

# ---------------------------------------------------------------------------
# USER CONFIGURATION: edit these settings before running the script.
# ---------------------------------------------------------------------------

# Local model checkpoint folder, or a Hub model ID when LOCAL_FILES_ONLY=False.
MODEL_PATH = str(PROJECT_DIR / "models" / "Qwen3.5-9B")
# Full Qwen3.5 checkpoints use this class; ordinary text-only checkpoints may
# instead need "AutoModelForCausalLM". Match the class to the model's config.
MODEL_CLASS = "AutoModelForImageTextToText"
# Read local/cached model files only; disable to allow Hub downloads.
LOCAL_FILES_ONLY = True
# Allow custom model/tokenizer implementation code when required by a checkpoint.
TRUST_REMOTE_CODE = True
# Attention backend supported by the selected model and installed PyTorch.
ATTENTION_IMPLEMENTATION = "sdpa"

# Input files generated by the CoT stage; all listed files must exist.
DATA_PATHS = [
    # News FinCoT pairs for the Causal Relationship dimension.
    COT_OUTPUT_PARQUET.parent / "FinAtlas_Causality_SFT_News.parquet",
    # Tweet FinCoT pairs for the same dimension.
    COT_OUTPUT_PARQUET.parent / "FinAtlas_Causality_SFT_Tweets.parquet",
    # Report FinCoT pairs for the same dimension.
    COT_OUTPUT_PARQUET.parent / "FinAtlas_Causality_SFT_Reports.parquet",
]
# Folder for training checkpoints, metrics, and the final adapter/tokenizer.
OUTPUT_DIR = PROJECT_DIR / "outputs" / "Causality_LoRA"
# None starts fresh; use a checkpoint path or True to resume the latest checkpoint.
RESUME_FROM_CHECKPOINT = None

# Number of training passes through the combined dataset.
EPOCHS = 10
# Initial learning rate for the trainable LoRA parameters.
LEARNING_RATE = 2e-5
# Samples processed simultaneously per GPU/process in each forward/backward pass.
# Experiments used one A800 80 GB; adjust this according to actual GPU memory use.
# Larger values generally need more memory; reduce this first if training runs OOM.
# On one GPU, 2 samples * 8 accumulation steps gives an effective batch of 16.
BATCH_SIZE = 2
# Forward/backward microbatches accumulated before one optimizer update.
# Accumulation combines gradients across sequential passes, not all samples in memory.
# For a single-GPU effective batch of 16, use (batch, accumulation) = (1,16),
# (2,8), (4,4), or (8,2), subject to memory. Changing GPU count changes the total.
GRAD_ACCUMULATION = 8
# Maximum total tokens, including system prompt, user prompt, response, and wrappers.
MAX_SEQ_LENGTH = 8192
# "skip" preserves full responses; "truncate" cuts overlong sequences on the right.
OVERLENGTH_POLICY = "skip"
# Dataset shuffle and training random seed.
SEED = 42

# Floating-point precision: "bf16", "fp16", or "fp32"; bf16 needs compatible hardware.
PRECISION = "bf16"
# Optimizer implementation understood by Transformers TrainingArguments.
OPTIMIZER = "adamw_torch"
# Learning-rate schedule applied during training.
LR_SCHEDULER = "cosine"
# Number of initial optimizer updates used for learning-rate warmup; zero disables it.
WARMUP_STEPS = 0
# AdamW weight decay for trainable parameters.
WEIGHT_DECAY = 0.0
# Gradient clipping threshold.
MAX_GRAD_NORM = 1.0
# Recompute activations during backward passes to reduce GPU memory use.
GRADIENT_CHECKPOINTING = True

# LoRA low-rank matrix size.
LORA_RANK = 16
# LoRA scaling parameter.
LORA_ALPHA = 32
# Adapter dropout probability.
LORA_DROPOUT = 0.05
# Module names to adapt; match the chosen model. "all-linear" broadens coverage.
LORA_TARGET_MODULES = ["q_proj", "k_proj"]
# Bias training mode understood by PEFT: "none", "all", or "lora_only".
LORA_BIAS = "none"

# OpenMP thread count; applied before importing PyTorch and other native libraries.
OMP_NUM_THREADS = 8
# Number of dataset tokenization processes; set to 1 for serial preprocessing.
TOKENIZE_NUM_PROC = 4
# DataLoader worker processes; zero is portable across Windows and Linux.
DATALOADER_NUM_WORKERS = 0
# Print training logs every this many optimizer updates.
LOGGING_STEPS = 5
# Checkpoint schedule: "epoch", "steps", or "no".
SAVE_STRATEGY = "epoch"
# Checkpoint interval in optimizer updates when SAVE_STRATEGY="steps".
SAVE_STEPS = 500
# Maximum retained checkpoints; None keeps all checkpoints.
SAVE_TOTAL_LIMIT = None
# External training integrations; "none" disables reporting services.
REPORT_TO = "none"
# Enable only when distributed training has trainable modules unused in forward passes.
DDP_FIND_UNUSED_PARAMETERS = False

# Extra chat-template options; update these when switching model families.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
# Shared system instruction for news, tweets, and reports; targets remain in responses.
SYSTEM_PROMPT = (
    "You are a financial NLP expert performing Financial Chain-of-Thought (FinCoT) "
    "for the Causal Relationship dimension. Extract source-supported causal paths "
    "and classify complexity as NONE, SINGLE, or MULTIPLE. MULTIPLE requires a "
    "strictly sequential chain with at least two explicitly stated intermediate "
    "mechanisms. Independent A -> B and A -> C relationships are separate SINGLE "
    "paths. Do not invent missing mechanisms. Follow the user's output protocol "
    "and return a <thinking> rationale followed by <output> structured JSON."
)

# ---------------------------------------------------------------------------
# DATA PREPARATION AND RESPONSE-ONLY LOSS MASKING
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def valid_training_pair(example: dict[str, Any]) -> bool:
    return all(
        isinstance(example.get(key), str) and bool(example[key].strip())
        for key in ("prompt", "response")
    )


def load_training_data() -> Any:
    """Align the three sources to the lowercase two-column CoT output schema."""
    from datasets import Features, Value, concatenate_datasets, load_dataset

    sources = []
    schema = Features({"prompt": Value("string"), "response": Value("string")})
    for path in DATA_PATHS:
        source = load_dataset("parquet", data_files=str(path), split="train")
        if {"prompt", "response"}.issubset(source.column_names):
            source = source.select_columns(["prompt", "response"])
        elif {"Prompt", "Response"}.issubset(source.column_names):
            source = source.select_columns(["Prompt", "Response"]).rename_columns(
                {"Prompt": "prompt", "Response": "response"}
            )
        else:
            raise ValueError(f"Missing prompt/response column pair in {path}")
        original_size = len(source)
        source = source.filter(valid_training_pair).cast(schema)
        LOGGER.info("%s: %s pairs; %s blank/invalid rows excluded", path.name, len(source), original_size - len(source))
        if len(source):
            sources.append(source)
    if not sources:
        raise ValueError("No valid training pairs in the configured input files")
    return concatenate_datasets(sources).shuffle(seed=SEED)


def tokenize_and_mask(example: dict[str, str], tokenizer: Any) -> dict[str, Any]:
    """Mask the prompt without assuming Qwen-specific role marker strings."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["response"]}],
        tokenize=False,
        add_generation_prompt=False,
        **CHAT_TEMPLATE_KWARGS,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template does not preserve the generation prefix; review CHAT_TEMPLATE_KWARGS")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt token boundary changed in the full chat; response-only masking would be ambiguous")

    # Preserve the entire response by default, rather than training on partial JSON.
    skip_reason = ""
    if len(input_ids) > MAX_SEQ_LENGTH:
        if OVERLENGTH_POLICY == "skip":
            skip_reason = "overlength"
        else:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
    if len(prompt_ids) >= len(input_ids):
        skip_reason = skip_reason or "no_response_tokens"
    if skip_reason:
        return {"input_ids": [], "attention_mask": [], "labels": [], "skip_reason": skip_reason}

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "skip_reason": "",
    }


def prepare_training_data(dataset: Any, tokenizer: Any) -> Any:
    tokenized = dataset.map(
        tokenize_and_mask,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=TOKENIZE_NUM_PROC,
        desc="Tokenizing causality FinCoT pairs",
    )
    counts = Counter(tokenized["skip_reason"])
    LOGGER.info("Tokenization results: %s", dict(counts))
    tokenized = tokenized.filter(lambda row: not row["skip_reason"]).remove_columns("skip_reason")
    if not len(tokenized):
        raise ValueError("No supervised samples remain; review MAX_SEQ_LENGTH and the input pairs")
    return tokenized


def validate_config() -> None:
    if not DATA_PATHS:
        raise ValueError("DATA_PATHS must list at least one input file")
    for path in DATA_PATHS:
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Training input must be .parquet: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Required SFT input is missing: {path}")
    if EPOCHS <= 0 or LEARNING_RATE <= 0:
        raise ValueError("EPOCHS and LEARNING_RATE must be positive")
    if min(BATCH_SIZE, GRAD_ACCUMULATION, MAX_SEQ_LENGTH, TOKENIZE_NUM_PROC, OMP_NUM_THREADS) < 1:
        raise ValueError("Batch, accumulation, sequence length, process and thread settings must be positive")
    if PRECISION not in {"bf16", "fp16", "fp32"}:
        raise ValueError("PRECISION must be bf16, fp16, or fp32")
    if OVERLENGTH_POLICY not in {"skip", "truncate"}:
        raise ValueError("OVERLENGTH_POLICY must be skip or truncate")
    if MODEL_CLASS not in {"AutoModelForCausalLM", "AutoModelForImageTextToText"}:
        raise ValueError("Select a supported MODEL_CLASS for the checkpoint architecture")
    if LORA_RANK < 1 or LORA_ALPHA <= 0 or not 0 <= LORA_DROPOUT < 1 or not LORA_TARGET_MODULES:
        raise ValueError("Invalid LoRA rank, alpha, dropout, or target modules")
    if SAVE_STRATEGY not in {"epoch", "steps", "no"} or SAVE_STEPS < 1 or LOGGING_STEPS < 1:
        raise ValueError("Invalid checkpoint or logging settings")
    if SAVE_TOTAL_LIMIT is not None and SAVE_TOTAL_LIMIT < 1:
        raise ValueError("SAVE_TOTAL_LIMIT must be positive or None")
    if min(DATALOADER_NUM_WORKERS, WARMUP_STEPS, WEIGHT_DECAY, MAX_GRAD_NORM) < 0:
        raise ValueError("Worker count, warmup, weight decay and gradient norm must be non-negative")


def build_training_arguments(training_arguments_class: Any) -> Any:
    """Read editable settings directly; no separate defaults table is needed."""
    return training_arguments_class(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        bf16=PRECISION == "bf16",
        fp16=PRECISION == "fp16",
        optim=OPTIMIZER,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if GRADIENT_CHECKPOINTING else None,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=REPORT_TO,
        seed=SEED,
        data_seed=SEED,
        ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_config()
    # Apply thread settings before importing libraries that initialize native runtimes.
    os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    if PRECISION in {"bf16", "fp16"} and not torch.cuda.is_available():
        raise RuntimeError("The configured mixed precision requires CUDA; use a compatible GPU or PRECISION='fp32'")
    if PRECISION == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bf16; adjust PRECISION for the training hardware")
    set_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dataset = load_training_data()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=LOCAL_FILES_ONLY
    )
    if not tokenizer.chat_template:
        raise ValueError("The selected tokenizer must provide a chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a padding token nor an end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenized = prepare_training_data(dataset, tokenizer)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[PRECISION]
    model = getattr(transformers, MODEL_CLASS).from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        trust_remote_code=TRUST_REMOTE_CODE,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
        model.config.text_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # The collator pads input tokens dynamically and masks padding labels with -100.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=build_training_arguments(TrainingArguments),
        train_dataset=tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    LOGGER.info("Training causality LoRA on %s pairs", len(tokenized))
    result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    trainer.save_model(str(OUTPUT_DIR / "final"))
    if trainer.is_world_process_zero():
        final_dir = OUTPUT_DIR / "final"
        tokenizer.save_pretrained(str(final_dir))
        provenance = {
            "stage": "sft",
            "base_model": MODEL_PATH,
            "model_class": MODEL_CLASS,
            "sft_sources": [str(path) for path in DATA_PATHS],
            "system_prompt": SYSTEM_PROMPT,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        }
        (final_dir / "training_sources.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    LOGGER.info("Final causality adapter and tokenizer: %s", OUTPUT_DIR / "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
