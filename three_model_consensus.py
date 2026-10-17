"""Construct consensus annotations for FinAtlas's Causal Relationship dimension.

Task and text type
    This pipeline handles one English financial text type per run: News,
    Tweets, or Reports, selected by TEXT_TYPE. It labels causal
    complexity as SINGLE, MULTIPLE, or NONE and extracts Cause -> Mechanism ->
    Effect paths. A DeepSeek precheck is followed by three model votes. Records
    enter SFT when all three complexity labels agree and the matching category
    quota is available. Valid 2:1 votes and unanimous votes whose SFT category
    is full enter GRPO, subject to its total capacity. Run this stage and
    cot_process.py once for each type before starting the combined SFT stage.

Editable configuration
    All routine settings are grouped immediately below the imports. Set the
    input/output paths, three API keys, model IDs, API base URLs, category CAPS,
    GRPO_CAP, concurrency, and seed there, then run the file. The SFT target is
    the sum of the three CAPS values. GRPO_CAP defaults to 8,000 records without
    per-label quotas. Processing continues until both datasets are full or
    inputs are exhausted. Set GRPO_CAP to zero to disable GRPO collection.

    To select a stronger model from the same provider, change its *_MODEL to
    the exact model ID accepted by that provider's API. To use another
    provider in that slot, also change its *_BASE_URL and *_API_KEY. Endpoints
    must support OpenAI-compatible chat completions; the DeepSeek and Qwen
    slots also require JSON-object response format. DeepSeek remains the
    precheck slot and the first choice for extracted paths.

    CONCURRENCY_LIMIT controls simultaneous texts, not individual API calls.
    Each full vote sends three requests in parallel, allowing up to three
    times that many simultaneous voting requests. Lower it if rate limits
    are reached. SEED defaults to 42 and controls input shuffling and retry
    jitter; API responses and concurrent completion order are not deterministic.

    Non-empty *_API_KEY environment variables override the key settings.
    *_MODEL and *_BASE_URL environment variables can also override the file.
    Optional command-line flags override paths, caps, GRPO capacity, concurrency,
    and seed.
    Keep key placeholders in the GitHub version.

Input and output
    INPUT_DIR must contain .parquet files directly; subfolders are not scanned.
    Each file must have a Text or text column containing English financial
    text strings. Other input columns are ignored; null/blank texts are skipped.
    Default paths are anchored to the folder containing these four scripts.

    OUTPUT_DIR receives <TEXT_TYPE>_Causality_allocate.parquet and
    FinAtlas_<TEXT_TYPE>_Causality_GRPO.parquet, plus two type-specific resume logs.
    News keeps the filename News_Causality_allocate.parquet. Shared fields
    are text, Chain_Complexity, Causal_Paths, Votes, and Consensus_Level.
    GRPO also records GRPO_Reason and all three Model_Outputs. Causal_Paths
    Votes, Causal_Paths, and Model_Outputs are JSON strings. The logs track
    completed input files and attempted text hashes. Reuse the directory to
    resume the same run; choose a fresh directory after changing the corpus,
    models, prompts, or quotas. Previously discarded votes cannot be recovered
    from logs; use a fresh output directory to reprocess them for GRPO.

Consensus scope
    Consensus applies to Chain_Complexity. Positive records use the first
    non-empty path list from a model supporting the winning label, in
    DeepSeek/Qwen/GLM order. Missing/failed votes and 1:1:1 disagreements are
    discarded. While GRPO has room, scan all valid texts; afterward, use the
    original quota-dependent keyword filters to fill remaining SFT quotas.
    MULTIPLE requires a strictly sequential chain with at least two explicitly
    stated intermediate mechanisms. Independent A -> B and A -> C branches
    are separate SINGLE triads. Path counts and chain connectivity are checked
    in code; whether mechanisms are explicitly supported remains a model
    judgment and requires annotation review.

Dependencies: Python 3.10+; pip install openai pandas pyarrow tqdm
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from tqdm import tqdm

# ---------------------------------------------------------------------------
# USER CONFIGURATION: edit these settings before running the script.
# ---------------------------------------------------------------------------

# Common root for default paths; running a script from another folder is safe.
PROJECT_DIR = Path(__file__).resolve().parent
# Process one type per run; exact supported names: "News", "Tweets", "Reports".
# cot_process.py inherits this selection by default.
TEXT_TYPE = "News"
# Folder containing input .parquet files for the selected type; may be overridden.
INPUT_DIR = PROJECT_DIR / "data" / TEXT_TYPE.lower()
# Folder for the generated Parquet dataset and resume logs; created if absent.
OUTPUT_DIR = PROJECT_DIR / "causality"

# DeepSeek API credential; replace the placeholder locally before running.
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
# Exact API model ID for the DeepSeek slot; also used for the initial precheck.
DEEPSEEK_MODEL = "deepseek-chat"
# OpenAI-compatible API base URL for the DeepSeek slot.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Alibaba Cloud DashScope credential for the Qwen slot.
QWEN_API_KEY = "YOUR_QWEN_API_KEY"
# Exact API model ID for the Qwen slot; change this to use another available model.
QWEN_MODEL = "qwen-plus"
# OpenAI-compatible API base URL for the Qwen slot.
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Zhipu API credential for the GLM slot.
GLM_API_KEY = "YOUR_GLM_API_KEY"
# Exact API model ID for the GLM slot; change this to use another available model.
GLM_MODEL = "glm-4-flash"
# OpenAI-compatible API base URL for the GLM slot.
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# Maximum SFT records per label for the selected text type; adjust for each run.
# Their sum is this type's SFT target, not the combined three-source dataset size.
CAPS = {
    # One or more independent direct causal triads; branching alone stays SINGLE.
    "SINGLE": 500,
    # Strictly sequential chain with at least two explicitly stated mechanisms.
    "MULTIPLE": 300,
    # No substantive causal business impact.
    "NONE": 200,
}
# Maximum GRPO records in total: 2:1 votes plus 3:0 votes with a full SFT category.
# Set to zero to disable this output.
GRPO_CAP = 8000
# Maximum simultaneous texts; each voting text makes three parallel API requests.
CONCURRENCY_LIMIT = 16
# Default random seed for file/row shuffling and retry jitter.
SEED = 42
# Maximum attempts per full model vote, including retries for HTTP 429 errors.
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# PIPELINE IMPLEMENTATION
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)
PROVIDER_NAMES = ("DeepSeek", "Qwen", "GLM")
VALID_LABELS = frozenset({"SINGLE", "MULTIPLE", "NONE"})
TEXT_TYPES = ("News", "Tweets", "Reports")
SFT_COLUMNS = ("text", "Chain_Complexity", "Causal_Paths", "Votes", "Consensus_Level")
GRPO_COLUMNS = SFT_COLUMNS + ("GRPO_Reason", "Model_Outputs")

PROBE_PROMPT = """You are a strict binary classification tool.
Analyze the text and determine if there is a clear causal business/market logic.
Respond ONLY with a valid JSON object containing a single boolean field "has_causality". Do not output any thinking process.
Example: {"has_causality": true} or {"has_causality": false}"""

CAUSAL_DEFINITIONS = """[STRICT DEFINITIONS]
- SINGLE: A direct causal relationship without a sequential chain meeting the MULTIPLE definition. Output one triad for each independent direct relationship. A -> B and A -> C are TWO SINGLE triads, not MULTIPLE. Multiple independent relationships or effects do not by themselves imply MULTIPLE. If a direct link has no explicitly stated mechanism, use "Direct impact"; do not invent one.
- MULTIPLE: The text contains a strictly sequential causal chain in which a cause leads to a final effect through AT LEAST TWO EXPLICITLY STATED intermediate mechanisms. Output AT LEAST TWO connected triads in source-supported causal order; longer chains are allowed. Each triad's Effect MUST be EXACTLY the same string as the next triad's Cause. Identify the source evidence for each mechanism; do not infer or invent missing intermediate steps. Generic "Direct impact" placeholders do not count as explicit mechanisms. Branching or independent A -> B / A -> C paths NEVER satisfies this definition; encode those as separate SINGLE triads. When a qualifying chain exists, output that chain without appending unrelated branches.
- NONE: No substantive causal business impact. Output an empty array []."""

CAUSALITY_PROMPT = """You are a top-tier financial logic analyst.
Analyze the financial text and extract the causal chain (Cause -> Mechanism -> Effect).

""" + CAUSAL_DEFINITIONS + """

[OUTPUT FORMAT] (Must be valid JSON)
{
    "Chain_Complexity": "MULTIPLE",
    "Causal_Paths": [
        {
            "Cause": "Shortage of raw material supply",
            "Mechanism": "Scarcer raw materials increase procurement expenditure",
            "Effect": "Production costs increased by 15%"
        },
        {
            "Cause": "Production costs increased by 15%",
            "Mechanism": "Cost pass-through to consumers",
            "Effect": "Retail prices hiked by 5%"
        }
    ]
}"""

CAUSAL_PATTERN = re.compile(
    r"\b(?:due to|led to|resulted in|driven by|consequently|trigger|"
    r"attribute to|stem from|impact|cause|therefore|affect|because|prompt|"
    r"furthermore|subsequently|meanwhile|in turn)\b",
    re.IGNORECASE,
)
COMPLEX_PATTERN = re.compile(
    r"\b(?:chain reaction|ripple effect|cascade|domino effect|spillover|"
    r"not only|coupled with|two-fold|multiple|compound|exacerbate)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR
    text_type: str = TEXT_TYPE
    caps: dict[str, int] = field(default_factory=lambda: CAPS.copy())
    grpo_cap: int = GRPO_CAP
    concurrency: int = CONCURRENCY_LIMIT
    seed: int = SEED

    @property
    def target(self) -> int:
        return sum(self.caps.values())

    @property
    def dataset_path(self) -> Path:
        return self.output_dir / f"{self.text_type}_Causality_allocate.parquet"

    @property
    def grpo_path(self) -> Path:
        return self.output_dir / f"FinAtlas_{self.text_type}_Causality_GRPO.parquet"

    @property
    def processed_log_path(self) -> Path:
        return self.output_dir / f"{self.text_type.lower()}_causality_processed_files.log"

    @property
    def seen_log_path(self) -> Path:
        return self.output_dir / f"seen_{self.text_type.lower()}_causality_hashes.log"


def text_hash(text: str) -> str:
    """Use the original MD5 fingerprint for deduplication, not security."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_log(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_log(path: Path, entries: list[str]) -> None:
    if entries:
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(f"{entry}\n" for entry in entries)


def parse_json_object(raw: str | None) -> dict[str, Any]:
    """Accept a JSON object, optionally wrapped in a Markdown code fence."""
    if not raw:
        raise ValueError("Empty model response")
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    parsed = json.loads(match.group(1).strip() if match else raw.strip())
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def validate_causal_paths(label: str, paths: Any) -> None:
    """Check path structure; source-grounding of mechanisms is a model judgment."""
    if label not in VALID_LABELS or not isinstance(paths, list):
        raise ValueError("Invalid causality label or path list")
    if label == "NONE" and paths:
        raise ValueError("NONE requires an empty path list")
    if label == "SINGLE" and not paths:
        raise ValueError("SINGLE requires at least one direct causal triad")
    if label == "MULTIPLE" and len(paths) < 2:
        raise ValueError("MULTIPLE requires at least two sequential mechanisms")
    for triad in paths:
        if not isinstance(triad, dict) or any(
            not isinstance(triad.get(key), str) or not triad[key].strip()
            for key in ("Cause", "Mechanism", "Effect")
        ):
            raise ValueError("Each path needs non-empty Cause, Mechanism, and Effect strings")
    if label == "MULTIPLE":
        if any(triad["Mechanism"].strip().casefold() == "direct impact" for triad in paths):
            raise ValueError("Direct impact placeholders cannot establish explicit intermediate mechanisms")
        if any(left["Effect"] != right["Cause"] for left, right in zip(paths, paths[1:])):
            raise ValueError("MULTIPLE paths must form one strictly sequential chain")


def normalize_causal_paths(label: str, paths: Any) -> list[dict[str, str]]:
    """Keep the exact three triad keys consumed by both SFT and GRPO."""
    validate_causal_paths(label, paths)
    return [
        {key: triad[key] for key in ("Cause", "Mechanism", "Effect")}
        for triad in paths
    ]


def select_consensus(results: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Accept a 3:0 or 2:1 label vote only when all three models succeeded."""
    if len(results) != 3 or any(result is None for result in results):
        return None
    valid_results = [result for result in results if result is not None]
    votes = [result["Chain_Complexity"] for result in valid_results]
    label, winning_count = Counter(votes).most_common(1)[0]
    if winning_count < 2:
        return None

    paths: list[Any] = []
    if label != "NONE":
        # This is label consensus, not a vote over the extracted causal paths.
        paths = next(
            (
                result["Causal_Paths"]
                for result in valid_results
                if result["Chain_Complexity"] == label and result["Causal_Paths"]
            ),
            [],
        )
        if not paths:
            return None
    paths = normalize_causal_paths(label, paths)
    return {
        "Chain_Complexity": label,
        "Causal_Paths": json.dumps(paths, ensure_ascii=False),
        "Votes": json.dumps(votes, ensure_ascii=False),
        "Consensus_Level": "STRICT_3_TO_0" if winning_count == 3 else "MAJORITY_2_TO_1",
    }


class PipelineState:
    """Own both datasets and protect their capacity updates from concurrent tasks."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.records: list[dict[str, Any]] = []
        self.grpo_records: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.lock = asyncio.Lock()

    @property
    def sft_full(self) -> bool:
        return len(self.records) >= self.config.target

    @property
    def grpo_full(self) -> bool:
        return len(self.grpo_records) >= self.config.grpo_cap

    @property
    def is_full(self) -> bool:
        return self.sft_full and self.grpo_full

    def load_checkpoint(self) -> None:
        import pandas as pd

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.dataset_path.exists():
            self.records = pd.read_parquet(self.config.dataset_path).to_dict(orient="records")
        if self.config.grpo_path.exists():
            self.grpo_records = pd.read_parquet(self.config.grpo_path).to_dict(orient="records")
        if len(self.grpo_records) > self.config.grpo_cap:
            raise ValueError("GRPO checkpoint exceeds GRPO_CAP; increase the capacity or use a fresh output directory")
        self.counts = Counter(record["Chain_Complexity"] for record in self.records)
        for label, count in self.counts.items():
            if label not in VALID_LABELS or count > self.config.caps[label]:
                raise ValueError(
                    f"Checkpoint label/count conflicts with configured quotas: {label}={count}"
                )
        LOGGER.info("Loaded %s accepted records; quotas: %s", len(self.records), self.config.caps)
        LOGGER.info("Loaded %s/%s GRPO records", len(self.grpo_records), self.config.grpo_cap)

    def save_checkpoint(self) -> None:
        import pandas as pd

        for records, path in (
            (self.records, self.config.dataset_path),
            (self.grpo_records, self.config.grpo_path),
        ):
            columns = GRPO_COLUMNS if path == self.config.grpo_path else SFT_COLUMNS
            # Replace only after a complete write, preserving the previous checkpoint on failure.
            temporary_path = path.with_suffix(".parquet.tmp")
            # Empty outputs still carry the complete string schema for the next stage.
            pd.DataFrame(records, columns=columns, dtype="string").to_parquet(temporary_path, index=False)
            temporary_path.replace(path)


class ConsensusPipeline:
    def __init__(self, config: PipelineConfig, state: PipelineState) -> None:
        self.config = config
        self.state = state
        self.clients: dict[str, AsyncOpenAI] = {}
        self.models: dict[str, str] = {}
        self.rng = random.Random(config.seed)

    def create_clients(self) -> None:
        file_keys = {
            "DeepSeek": DEEPSEEK_API_KEY,
            "Qwen": QWEN_API_KEY,
            "GLM": GLM_API_KEY,
        }
        # Prefer a non-empty environment value; otherwise use the editable key above.
        api_keys = {
            name: os.getenv(f"{name.upper()}_API_KEY", "").strip() or key.strip()
            for name, key in file_keys.items()
        }
        missing = [
            f"{name.upper()}_API_KEY"
            for name, key in api_keys.items()
            if not key or key.startswith("YOUR_")
        ]
        if missing:
            raise ValueError(
                "Missing API keys: " + ", ".join(missing)
                + ". Replace the placeholders in the API key section at the top "
                "of this file, or set the matching environment variables."
            )

        from openai import AsyncOpenAI

        self.models = {
            "DeepSeek": os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL),
            "Qwen": os.getenv("QWEN_MODEL", QWEN_MODEL),
            "GLM": os.getenv("GLM_MODEL", GLM_MODEL),
        }
        base_urls = {
            "DeepSeek": os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
            "Qwen": os.getenv("QWEN_BASE_URL", QWEN_BASE_URL),
            "GLM": os.getenv("GLM_BASE_URL", GLM_BASE_URL),
        }
        for name in PROVIDER_NAMES:
            self.clients[name] = AsyncOpenAI(
                api_key=api_keys[name],
                base_url=base_urls[name],
                max_retries=0,  # Retries are managed explicitly below.
            )

    async def probe(self, text: str) -> bool:
        """Use a cheap precheck; failure falls through to the full three-model vote."""
        try:
            response = await self.clients["DeepSeek"].chat.completions.create(
                model=self.models["DeepSeek"],
                messages=[
                    {"role": "system", "content": PROBE_PROMPT},
                    {"role": "user", "content": f"Text type: {self.config.text_type}\n{text[:2000]}"},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=15.0,
            )
            value = parse_json_object(response.choices[0].message.content)["has_causality"]
            if not isinstance(value, bool):
                raise ValueError("has_causality must be a JSON boolean")
            return value
        except Exception as error:
            LOGGER.debug("Probe failed (%s); proceeding to full vote", type(error).__name__)
            return True

    async def call_model(self, name: str, text: str) -> dict[str, Any] | None:
        for attempt in range(MAX_ATTEMPTS):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.models[name],
                    "messages": [
                        {"role": "system", "content": CAUSALITY_PROMPT},
                        {"role": "user", "content": f"Text type: {self.config.text_type}\n{text}"},
                    ],
                    "temperature": 0.1,
                    "timeout": 25.0,
                }
                if name in ("DeepSeek", "Qwen"):
                    kwargs["response_format"] = {"type": "json_object"}
                response = await self.clients[name].chat.completions.create(**kwargs)
                result = parse_json_object(response.choices[0].message.content)
                label = result["Chain_Complexity"].upper()
                paths = normalize_causal_paths(label, result["Causal_Paths"])
                return {"Chain_Complexity": label, "Causal_Paths": paths}
            except Exception as error:
                if getattr(error, "status_code", None) == 429 and attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep((attempt + 1) * 3 + self.rng.uniform(0.5, 1.5))
                    continue
                # Do not turn API/JSON failures into artificial NONE votes.
                LOGGER.warning("%s vote unavailable (%s)", name, type(error).__name__)
                return None
        return None

    async def process_item(self, text: str, semaphore: asyncio.Semaphore, progress: tqdm) -> None:
        async with semaphore:
            if self.state.is_full:
                return
            has_causality = await self.probe(text)
            if (
                not has_causality
                and self.state.counts["NONE"] >= self.config.caps["NONE"]
                and self.state.grpo_full
            ):
                return

            results = await asyncio.gather(*(self.call_model(name, text) for name in PROVIDER_NAMES))
            consensus = select_consensus(results)
            if consensus is None:
                return
            label = consensus["Chain_Complexity"]

            # Check and increment together: concurrent votes must not exceed a quota.
            async with self.state.lock:
                if self.state.is_full:
                    return
                record = {"text": text, **consensus}
                unanimous = consensus["Consensus_Level"] == "STRICT_3_TO_0"
                if unanimous and self.state.counts[label] < self.config.caps[label]:
                    self.state.records.append(record)
                    self.state.counts[label] += 1
                elif not self.state.grpo_full:
                    record["GRPO_Reason"] = "SFT_QUOTA_FULL" if unanimous else "MAJORITY_2_TO_1"
                    record["Model_Outputs"] = json.dumps(
                        dict(zip(PROVIDER_NAMES, results)), ensure_ascii=False
                    )
                    self.state.grpo_records.append(record)
                else:
                    return
                progress.update(1)
                progress.set_postfix({**self.state.counts, "GRPO": len(self.state.grpo_records)})

    def filter_candidates(self, frame: Any) -> Any:
        """Scan all GRPO candidates; focus on remaining SFT labels once GRPO fills."""
        if not self.state.grpo_full:
            return frame
        needs_multiple = self.state.counts["MULTIPLE"] < self.config.caps["MULTIPLE"]
        needs_single = self.state.counts["SINGLE"] < self.config.caps["SINGLE"]
        if not needs_single and needs_multiple:
            # Short tweets can contain a complete sequential chain.
            if self.config.text_type != "Tweets":
                frame = frame[frame["Text"].str.len() > 250]
            dense = frame["Text"].str.count(CAUSAL_PATTERN) >= 2
            complex_text = frame["Text"].str.contains(COMPLEX_PATTERN, na=False)
            return frame[dense | complex_text]
        if needs_multiple or needs_single:
            return frame[frame["Text"].str.contains(CAUSAL_PATTERN, na=False)]
        return frame

    async def process_batch(self, texts: list[str], semaphore: asyncio.Semaphore, progress: tqdm) -> None:
        await asyncio.gather(*(self.process_item(text, semaphore, progress) for text in texts))
        # Commit accepted records before marking their inputs as attempted.
        # Otherwise a restart could skip records absent from the last checkpoint.
        self.state.save_checkpoint()
        append_log(self.config.seen_log_path, [text_hash(text) for text in texts])

    async def run(self) -> None:
        import pandas as pd
        from tqdm import tqdm

        try:
            self.create_clients()
            files = sorted(self.config.input_dir.glob("*.parquet"))
            self.rng.shuffle(files)
            processed = load_log(self.config.processed_log_path)
            seen = load_log(self.config.seen_log_path)
            seen.update(
                text_hash(record["text"])
                for records in (self.state.records, self.state.grpo_records)
                for record in records
            )
            semaphore = asyncio.Semaphore(self.config.concurrency)

            with tqdm(
                total=self.config.target + self.config.grpo_cap,
                initial=len(self.state.records) + len(self.state.grpo_records),
                desc=f"{self.config.text_type} Causality SFT + GRPO",
                dynamic_ncols=True,
            ) as progress:
                for path in files:
                    if self.state.is_full:
                        break
                    if path.name in processed:
                        continue
                    LOGGER.info("Reading %s", path.name)
                    try:
                        frame = pd.read_parquet(path)
                    except Exception as error:
                        LOGGER.warning("Skipping unreadable file %s (%s)", path.name, type(error).__name__)
                        continue
                    if "Text" not in frame.columns and "text" in frame.columns:
                        frame = frame.rename(columns={"text": "Text"})
                    elif "Text" in frame.columns and "text" in frame.columns:
                        valid_primary = frame["Text"].map(
                            lambda value: isinstance(value, str) and bool(value.strip())
                        )
                        frame["Text"] = frame["Text"].where(valid_primary, frame["text"])
                    if "Text" not in frame.columns:
                        LOGGER.warning("Skipping %s: missing Text/text column", path.name)
                        continue

                    # Do not send nulls, blank strings, or non-text values to APIs.
                    valid_text = frame["Text"].map(
                        lambda value: isinstance(value, str) and bool(value.strip())
                    )
                    frame = frame[valid_text]
                    frame = frame.sample(
                        frac=1, random_state=self.rng.randrange(2**32)
                    ).reset_index(drop=True)
                    frame = self.filter_candidates(frame)
                    batch: list[str] = []
                    with tqdm(
                        total=len(frame),
                        desc=f"Scanning {path.name}",
                        leave=False,
                        dynamic_ncols=True,
                    ) as scan:
                        for text in frame["Text"]:
                            if self.state.is_full:
                                break
                            scan.update(1)
                            fingerprint = text_hash(text)
                            if fingerprint in seen:
                                continue
                            seen.add(fingerprint)
                            batch.append(text)
                            if len(batch) >= self.config.concurrency:
                                await self.process_batch(batch, semaphore, progress)
                                batch.clear()
                        if batch:
                            await self.process_batch(batch, semaphore, progress)
                    if not self.state.is_full:
                        append_log(self.config.processed_log_path, [path.name])
                        processed.add(path.name)
        finally:
            await asyncio.gather(
                *(client.close() for client in self.clients.values()),
                return_exceptions=True,
            )


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text-type", choices=TEXT_TYPES, default=TEXT_TYPE)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--single-cap", type=int, default=CAPS["SINGLE"])
    parser.add_argument("--multiple-cap", type=int, default=CAPS["MULTIPLE"])
    parser.add_argument("--none-cap", type=int, default=CAPS["NONE"])
    parser.add_argument("--grpo-cap", type=int, default=GRPO_CAP, help="Total GRPO capacity; zero disables collection.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY_LIMIT,
        help="Concurrent texts; each full vote makes three parallel requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed input shuffling; does not make API responses deterministic.",
    )
    args = parser.parse_args(argv)
    config = PipelineConfig(
        input_dir=args.input_dir or (
            INPUT_DIR if args.text_type == TEXT_TYPE else PROJECT_DIR / "data" / args.text_type.lower()
        ),
        output_dir=args.output_dir,
        text_type=args.text_type,
        caps={"SINGLE": args.single_cap, "MULTIPLE": args.multiple_cap, "NONE": args.none_cap},
        grpo_cap=args.grpo_cap,
        concurrency=args.concurrency,
        seed=args.seed,
    )
    if config.text_type not in TEXT_TYPES:
        parser.error("TEXT_TYPE must be News, Tweets, or Reports.")
    if (
        any(cap < 0 for cap in config.caps.values())
        or config.grpo_cap < 0
        or config.target + config.grpo_cap == 0
    ):
        parser.error("SFT quotas and GRPO capacity must be non-negative with a positive combined target.")
    if config.concurrency < 1:
        parser.error("--concurrency must be positive.")
    if not config.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {config.input_dir}")
    if not any(config.input_dir.glob("*.parquet")):
        parser.error("Input directory contains no .parquet files.")
    if config.input_dir.resolve() == config.output_dir.resolve():
        parser.error("Input and output directories must differ to avoid reading generated records as input.")
    return config


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    state = PipelineState(config)
    # Load before entering the save-on-exit block, so an invalid checkpoint is not rewritten.
    state.load_checkpoint()
    if state.is_full:
        LOGGER.info("Both SFT (%s) and GRPO (%s) are already full.", config.target, config.grpo_cap)
        return 0
    try:
        asyncio.run(ConsensusPipeline(config, state).run())
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; saving accepted records.")
        return 130
    finally:
        state.save_checkpoint()
        LOGGER.info("Accepted %s/%s records: %s", len(state.records), config.target, dict(state.counts))
        LOGGER.info("GRPO records: %s/%s", len(state.grpo_records), config.grpo_cap)
    if not state.is_full:
        LOGGER.warning("Input exhausted before all quotas were filled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
