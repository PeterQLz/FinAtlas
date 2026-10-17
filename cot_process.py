"""Generate FinCoT training pairs for FinAtlas's Causal Relationship dimension.

Task and text type
    This is stage 2 for one English financial text type per run: News, Tweets,
    or Reports, selected by TEXT_TYPE. It reads the unanimous SFT
    allocation produced by three_model_consensus.py, asks one model to explain
    the supplied causal annotation, and assembles supervised training pairs.
    The original Chain_Complexity and Causal_Paths are preserved in the final
    response. GRPO samples are handled separately by the previous stage.

Editable configuration
    TEXT_TYPE inherits the stage-1 selection and may be overridden here. Set
    INPUT_PARQUET, OUTPUT_PARQUET, the API key, model ID, API base URL,
    CONCURRENCY_LIMIT, and generation settings in the configuration block below.
    To use a stronger model from the same provider, change MODEL_NAME to its
    exact API model ID. To change providers, also update API_KEY and BASE_URL;
    the endpoint must support OpenAI-compatible chat completions.
    Non-empty DEEPSEEK_API_KEY, DEEPSEEK_MODEL, and DEEPSEEK_BASE_URL environment
    variables can override these three settings. Keep the key placeholder in
    the GitHub version. Run the file after editing the configuration.

    CONCURRENCY_LIMIT is the maximum simultaneous generation requests. Lower
    it when rate limits are reached. SEED defaults to 42 and seeds retry jitter;
    input order is preserved and model responses are not deterministic.

Input and output
    Input is one .parquet file: <TEXT_TYPE>_Causality_allocate.parquet in the
    stage-1 output folder. Run stages 1 and 2 for each type before combined SFT.
    Required columns: text (or Text), Chain_Complexity, and Causal_Paths.
    Causal_Paths may be a JSON string or a list of dictionaries. Legacy Python
    literal lists are also parsed safely without replacing apostrophes.
    Other input columns are ignored. Invalid annotations and failed/empty
    generations are excluded and counted in the final log.

    Output is FinAtlas_Causality_SFT_<TEXT_TYPE>.parquet in the same folder,
    with exactly two string columns: prompt and response. The prompt includes
    the task, blueprint, and source text, without the target annotation. The
    response contains <thinking> rationale followed by <output> target JSON.
    Default paths inherit the stage-1 output directory anchored to these scripts.
    Each run regenerates the dataset and replaces the output after completion.

    SINGLE may contain several independent direct causal triads. MULTIPLE
    requires a strictly sequential chain with at least two explicitly stated
    intermediate mechanisms; branching alone stays SINGLE. NONE requires an
    empty path list. Code checks counts/connectivity and rejects generic
    Direct impact placeholders in MULTIPLE; explicit source support is a model
    judgment. This matches the previous stage's prompt.
    Generated rationales still require review for factual and annotation quality.

Dependencies: Python 3.10+; pip install openai pandas pyarrow tqdm
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from three_model_consensus import (
    CAUSAL_DEFINITIONS,
    DEEPSEEK_API_KEY as ANNOTATION_API_KEY,
    DEEPSEEK_BASE_URL as ANNOTATION_BASE_URL,
    DEEPSEEK_MODEL as ANNOTATION_MODEL,
    OUTPUT_DIR as ALLOCATION_DIR,
    TEXT_TYPE as ALLOCATION_TEXT_TYPE,
    TEXT_TYPES,
    normalize_causal_paths,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# USER CONFIGURATION: edit these settings before running the script.
# ---------------------------------------------------------------------------

# One type per run; inherits three_model_consensus.py or may be set explicitly here.
TEXT_TYPE = ALLOCATION_TEXT_TYPE
# Stage 1 SFT allocation for this type; accepts a different absolute path if needed.
INPUT_PARQUET = ALLOCATION_DIR / f"{TEXT_TYPE}_Causality_allocate.parquet"
# Final two-column training file: FinAtlas_<dimension>_SFT_<text type>.parquet.
OUTPUT_PARQUET = ALLOCATION_DIR / f"FinAtlas_Causality_SFT_{TEXT_TYPE}.parquet"

# API credential for local use; keep the placeholder when publishing to GitHub.
API_KEY = ANNOTATION_API_KEY
# Exact API model ID; replace it with another model available at BASE_URL.
MODEL_NAME = ANNOTATION_MODEL
# OpenAI-compatible API base URL; change it when switching providers.
BASE_URL = ANNOTATION_BASE_URL

# Maximum simultaneous CoT generation requests; each text makes one request.
CONCURRENCY_LIMIT = 16
# Maximum attempts per record, including the initial request.
MAX_ATTEMPTS = 3
# Sampling temperature for rationale generation.
TEMPERATURE = 0.2
# Timeout in seconds for each API attempt.
REQUEST_TIMEOUT = 30.0
# Base wait in seconds between attempts; increases with the attempt number.
RETRY_BASE_DELAY = 3.0
# Random seed for retry jitter; does not make model responses deterministic.
SEED = 42
# Requested rationale length in words; these are prompt instructions, not filters.
COT_MIN_WORDS = 100
COT_MAX_WORDS = 200

# ---------------------------------------------------------------------------
# SHARED CAUSALITY PROMPTS FOR NEWS, TWEETS, AND REPORTS
# ---------------------------------------------------------------------------

DISTILLATION_SYSTEM_PROMPT = """You are a financial analyst explaining causal annotations of financial texts (news, tweets, and reports).
You receive the source text and an existing target annotation consisting of Chain_Complexity and Causal_Paths.
Write a concise, evidence-grounded rationale explaining the annotation using the supplied economic transmission blueprint.

[EXECUTION RULES]
1. Follow the blueprint and put the entire rationale inside one <thinking> block.
2. Ground every claim in the source text. Do not invent entities, numbers, events, mechanisms, or effects to justify a target.
3. Explain the cause, transmission mechanism, effect, and topology supported by the text. For NONE, explain why no substantive causal business or market impact is established.
4. SINGLE may contain one or more independent direct triads: A -> B and A -> C are two SINGLE relationships. MULTIPLE requires a strictly sequential chain with at least two explicitly stated intermediate mechanisms; it may contain more than two connected triads. Explain the source evidence for each mechanism, verify every adjacent Effect/Cause match, and do not invent missing steps. Generic Direct impact placeholders and branching do not establish MULTIPLE.
5. Return only the <thinking> block. Do not output the target JSON, Markdown fences, or additional wrappers."""

BLUEPRINT = """[Expert Blueprint Hint]
graph TD
A["Step 1: Event Ingestion & Token-Level Triggers"] --> B["Step 2: Isolate Direct Root Cause"]
B --> C["Step 3: Map Transmission Mechanism / Market Bridge"]
C --> D["Step 4: Quantify Final Economic Effect / Asset Impact"]
D --> E{{"Step 5: Synthesize Topology & Determine Complexity (NONE/SINGLE/MULTIPLE)"}}
E --> F["Step 6: Emit Unified Causal Proof Structure"]"""

DISTILLATION_USER_PROMPT = """{blueprint}

[Input {text_type}]
{text}

[Target Causality Structure]
Chain_Complexity: {complexity}
Causal_Paths: {causal_paths}

[Response Format]
<thinking>
Follow the blueprint and explain how the source evidence supports the target complexity and causal paths. Write approximately {min_words}-{max_words} words.
</thinking>"""

# Training prompts expose the source text and task, while targets stay in responses.
INFERENCE_PROMPT_TEMPLATE = """You are a top-tier financial logic analyst.
Analyze the financial text and extract the causal chain (Cause -> Mechanism -> Effect).

""" + CAUSAL_DEFINITIONS + """

{blueprint}

[Input Financial Text]
{text}

[Output Protocol]
First write a concise rationale inside <thinking> tags. Then write an <output> block containing a valid JSON object with exactly two keys: Chain_Complexity and Causal_Paths.
Each causal path must contain Cause, Mechanism, and Effect. Do not use Markdown code fences."""

LOGGER = logging.getLogger(__name__)
VALID_LABELS = frozenset({"NONE", "SINGLE", "MULTIPLE"})


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    """Validate the supplied target before spending an API call on its rationale."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        text = row.get("Text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing or blank financial text")
    label = row.get("Chain_Complexity")
    if not isinstance(label, str) or label not in VALID_LABELS:
        raise ValueError("Unknown Chain_Complexity")

    paths = row.get("Causal_Paths")
    if isinstance(paths, str):
        try:
            paths = json.loads(paths)
        except json.JSONDecodeError:
            paths = ast.literal_eval(paths)
    # Arrow list columns can become NumPy arrays when loaded through pandas.
    if hasattr(paths, "tolist"):
        paths = paths.tolist()
    paths = normalize_causal_paths(label, paths)
    return {"text": text, "Chain_Complexity": label, "Causal_Paths": paths}


def extract_rationale(raw: str | None) -> str:
    """Keep the rationale and discard surrounding response wrappers."""
    if not raw or not raw.strip():
        raise ValueError("Empty generation")
    blocks = re.findall(r"<thinking>\s*(.*?)\s*</thinking>", raw, re.DOTALL | re.IGNORECASE)
    if blocks:
        if len(blocks) != 1:
            raise ValueError("Expected one thinking block")
        rationale = blocks[0].strip()
    else:
        if re.search(r"</?(?:thinking|output)\b", raw, re.IGNORECASE):
            raise ValueError("Invalid response wrappers")
        rationale = raw.strip()
        fence = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", rationale, re.DOTALL)
        if fence:
            rationale = fence.group(1).strip()
    if not rationale or re.search(r"</?(?:thinking|output)\b", rationale, re.IGNORECASE):
        raise ValueError("Empty or malformed rationale")
    return rationale


def assemble_training_pair(record: dict[str, Any], rationale: str) -> dict[str, str]:
    """Use the original validated target, rather than asking the model to relabel it."""
    target = json.dumps(
        {"Chain_Complexity": record["Chain_Complexity"], "Causal_Paths": record["Causal_Paths"]},
        ensure_ascii=False,
        indent=2,
    )
    return {
        "prompt": build_inference_prompt(record["text"]),
        "response": f"<thinking>\n{rationale}\n</thinking>\n<output>\n{target}\n</output>",
    }


def build_inference_prompt(text: str) -> str:
    """Keep the SFT user prompt identical to the GRPO user prompt."""
    return INFERENCE_PROMPT_TEMPLATE.format(blueprint=BLUEPRINT, text=text)


async def generate_causality_cot(
    record: dict[str, Any],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    rng: random.Random,
) -> dict[str, str] | None:
    async with semaphore:
        user_message = DISTILLATION_USER_PROMPT.format(
            blueprint=BLUEPRINT,
            text_type=TEXT_TYPE,
            text=record["text"],
            complexity=record["Chain_Complexity"],
            causal_paths=json.dumps(record["Causal_Paths"], ensure_ascii=False),
            min_words=COT_MIN_WORDS,
            max_words=COT_MAX_WORDS,
        )
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    timeout=REQUEST_TIMEOUT,
                )
                rationale = extract_rationale(response.choices[0].message.content)
                return assemble_training_pair(record, rationale)
            except Exception as error:
                if attempt == MAX_ATTEMPTS - 1:
                    # Exclude failed generations instead of putting error text into SFT.
                    LOGGER.warning("CoT generation excluded after %s attempts (%s)", MAX_ATTEMPTS, type(error).__name__)
                    return None
                delay = RETRY_BASE_DELAY * (attempt + 1) + rng.uniform(0.5, 2.0)
                await asyncio.sleep(delay)
    return None


async def process_records(
    records: list[dict[str, Any]], api_key: str, base_url: str, model: str
) -> list[dict[str, str]]:
    from openai import AsyncOpenAI
    from tqdm import tqdm

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    rng = random.Random(SEED)
    pairs: list[dict[str, str]] = []
    try:
        # Bounded batches avoid scheduling the entire dataset at once.
        with tqdm(total=len(records), desc=f"{TEXT_TYPE} Causality FinCoT", dynamic_ncols=True) as progress:
            for start in range(0, len(records), CONCURRENCY_LIMIT):
                batch = records[start : start + CONCURRENCY_LIMIT]
                results = await asyncio.gather(
                    *(generate_causality_cot(record, client, model, semaphore, rng) for record in batch)
                )
                pairs.extend(result for result in results if result is not None)
                progress.update(len(batch))
    finally:
        await client.close()
    return pairs


def save_training_pairs(pairs: list[dict[str, str]]) -> None:
    import pandas as pd

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PARQUET.with_suffix(".parquet.tmp")
    # Explicit names and string types also give empty outputs the correct schema.
    pd.DataFrame(pairs, columns=["prompt", "response"], dtype="string").to_parquet(
        temporary_path, index=False, engine="pyarrow"
    )
    temporary_path.replace(OUTPUT_PARQUET)


def validate_config() -> None:
    if TEXT_TYPE not in TEXT_TYPES:
        raise ValueError("TEXT_TYPE must be News, Tweets, or Reports")
    if INPUT_PARQUET.suffix.lower() != ".parquet" or OUTPUT_PARQUET.suffix.lower() != ".parquet":
        raise ValueError("Input and output must be .parquet files")
    if INPUT_PARQUET.resolve() == OUTPUT_PARQUET.resolve():
        raise ValueError("Input and output paths must differ")
    if not INPUT_PARQUET.is_file():
        raise FileNotFoundError(f"Input allocation not found: {INPUT_PARQUET}")
    if CONCURRENCY_LIMIT < 1 or MAX_ATTEMPTS < 1:
        raise ValueError("Concurrency and maximum attempts must be positive")
    if REQUEST_TIMEOUT <= 0 or RETRY_BASE_DELAY < 0:
        raise ValueError("Timeout must be positive and retry delay non-negative")
    if not 0 < COT_MIN_WORDS <= COT_MAX_WORDS:
        raise ValueError("CoT word limits must be positive and ordered")


def main() -> int:
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_config()
    frame = pd.read_parquet(INPUT_PARQUET)
    required = {"Chain_Complexity", "Causal_Paths"}
    if not required.issubset(frame.columns) or not {"text", "Text"}.intersection(frame.columns):
        raise ValueError("Input requires text/Text, Chain_Complexity, and Causal_Paths columns")

    records = []
    invalid_count = 0
    for index, row in enumerate(frame.to_dict(orient="records"), 1):
        try:
            records.append(normalize_record(row))
        except (ValueError, TypeError, SyntaxError) as error:
            invalid_count += 1
            LOGGER.warning("Input row %s excluded: %s", index, error)

    pairs: list[dict[str, str]] = []
    if records:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or API_KEY.strip()
        if not api_key or api_key.startswith("YOUR_"):
            raise ValueError("Set API_KEY in the top configuration block or set DEEPSEEK_API_KEY")
        model = os.getenv("DEEPSEEK_MODEL", MODEL_NAME)
        base_url = os.getenv("DEEPSEEK_BASE_URL", BASE_URL)
        pairs = asyncio.run(process_records(records, api_key, base_url, model))
    save_training_pairs(pairs)
    LOGGER.info(
        "Input: %s; saved: %s; invalid annotations: %s; failed generations: %s",
        len(frame), len(pairs), invalid_count, len(records) - len(pairs),
    )
    LOGGER.info("Output: %s", OUTPUT_PARQUET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
