# FinAtlas: A Multi-Task Financial Sentiment Analysis System

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-FinAtlas-blue)](#)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)

FinAtlas is a comprehensive system designed for complex financial sentiment understanding, extending traditional polarity classification into multi-dimensional semantic parsing. This repository provides the instruction-tuning and benchmark datasets reconstructed from a massive-scale raw candidate corpus of approximately 80 million financial texts.

## 📌 Author

**FinAtlas** is constructed by:

- **Di Han** — dihan@gduf.edu.cn
- **Liangzhou Qu** — liangzhqu2-c@my.cityu.edu.hk
- **Junjie Mao** — junjie.mao@audencia.com
- **Yulin He<sup>*</sup>** — yulinhe@gml.ac.cn
- **Qixian Li** — 3260000547@student.must.edu.mo
- **Canwei Dai** — 18318717968@163.com

<sup>*</sup> Corresponding author

## 📌 System Components

The FinAtlas system integrates three core components:

1. **FINS (Data Evaluation Framework):** A task-driven evaluation framework that deconstructs financial sentiment signals across five dimensions.
2. **FinAtlas-SFT (Instruction-Tuning Dataset):** The finalized high-quality supervised fine-tuning dataset. *(Note: This dataset is the final output of our multi-model consensus and structured CoT reasoning pipeline).*
3. **FinAtlas-Bench (Benchmark):** A standardized dataset containing 2,000 independent samples for evaluating the capabilities of various LLMs in comprehending complex financial semantics.

## 🚀 The 5-Dimensional FINS Framework

Unlike datasets focused solely on sentiment polarity, FinAtlas aligns and evaluates models across the following five dimensions:

* **Macro Context:** Identifies the external market conditions or macro environment surrounding the text.
* **Risk Exposure:** Assesses the degree to which entities, assets, or markets are exposed to negative events and abnormal volatility.
* **Causal Relationship:** Extracts and evaluates the causal links and event transmission mechanisms.
* **Entity Relationship:** Identifies financial entities and extracts their interactive relationship networks.
* **Wording Difference:** Judges how shifts in expression impact sentiment intensity and signal semantics.

![FINS Framework Diagram](FINS.png)

---

## 📂 Datasets Access & Structure

We provide 5 SFT datasets corresponding to the 5 dimensions, and 1 unified benchmark dataset. The datasets are hosted in `.parquet` format for efficient loading.

### Quick Start
You can load the dataset directly using `pandas` or the Hugging Face `datasets` library:

```python
import pandas as pd

# Option 1: pandas (only requires pandas + pyarrow)
df_bench = pd.read_parquet("FinAtlas_Bench.parquet")
print(df_bench.head())

from datasets import load_dataset

# Option 2: Hugging Face datasets
ds_bench = load_dataset("parquet", data_files="FinAtlas_Bench.parquet")
print(ds_bench.head())

```

### Data Dictionary

Each record follows the multidimensional JSON schema introduced in Chapter 3 of the paper (a complete example is provided in the Sample Record below, reproduced from Appendix 3 of the paper):

| Field | Description | Type / Values |
| --- | --- | --- |
| `Text` | The original financial text (News, Tweet, or Report). | String |
| `Fundamental_Labels.Timestamp` | The trading date or event date corresponding to the text. | String (e.g., `2022/12/19`) |
| `Fundamental_Labels.Type` | Corpus source type. | String: `NEWS`, `TWEET`, `REPORT` |
| `Fundamental_Labels.Sentiment_Polarity` | Overall polarity of the text. | String: `BEARISH`, `BULLISH`, `NEUTRAL` |
| `Fundamental_Labels.Sentiment_Probability` | Model-generated confidence for the sentiment polarity label. | Float (e.g., `0.9765625`) |
| `Macro_Context.Market_Regime` | External market phase in which the text is situated. | String: `BULL`, `BEAR` |
| `Macro_Context.Macro_Event` | Major external events discussed in the text (e.g., COVID-19, monetary policy adjustments, regulatory changes). | List of Dict: `{anchor, confidence}` |
| `Risk_Exposure.Risk_Category` | Nature of the risk. A row may contain more than one category. | List of String: `Market Risk`, `Credit Risk`, `Liquidity Risk`, `Operational Risk`, `None` |
| `Risk_Exposure.Risk_Level` | Severity or potential impact of the risk. Actual values: `NO RISK`, `LOW RISK`, `HIGH RISK`; no `MEDIUM` level exists in this file. | String (e.g., `HIGH RISK`) |
| `Risk_Exposure.Risk_Triplets` | Risk source and affected target for each identified risk. | List of Dict: `{risk_category, risk_source, affected_object}` |
| `Causal_Relationship.Chain_Complexity` | Complexity of the causal transmission structure. | String: no causal relationship, `SINGLE`, `MULTIPLE` |
| `Causal_Relationship.Causal_Paths` | Causal chains preserving the cause, mechanism, and effect. | List of Dict: `{Chain_ID, Cause, Mechanism, Effect}` |
| `Entity_Relationship.Entity_Recognition` | Recognized financial entities (companies, institutions, individuals, financial indicators, stock tickers). | List of Dict: `{Entity_Name, Type}` with Type: `ORG`, `LOC`, `PER`, `FIN` |
| `Entity_Relationship.Relation_Extraction` | Relationship type and direction between entities, covering 9 categories (e.g., `ACQUISITION_MERGER`, `REGULATORY_ACTION`). | List of Dict: `{Relation, Clean_Entities}` |
| `Wording_Difference.Rhetoric_Type` | Expression mode of the text. | String: `OBJECTIVE_FACT`, `SUBJECTIVE_PREDICTION`, `EMOTIONAL_EXAGGERATION`, `VAGUE_RUMOR`, `NO_SPECIFIC_RHETORIC` |
| `Wording_Difference.Fuzzy_Text_Confidence` | Model uncertainty in rhetoric and sentiment judgment, derived from the triangular fuzzy number. | String / Float (e.g., `HIGH`) |

### Sample Record in FinAtals-Bench

The following JSON object is a real FinAtlas-Bench sample reproduced from Appendix 3 of the paper. It illustrates the complete multidimensional extraction results, including fundamental labels, macro context, risk exposure, causal relationship, entity relationship, and wording difference:

```json
{
  "Text": "Wall Street Closes Lower for a 4th Day | United States Stock Market | US stocks closed lower for a fourth straight session Monday, as concerns over a recession in the US mounted after the Fed took a more hawkish tone than expected. Chair Powell said last week the central bank would continue its monetary policy tightening campaign, despite the ongoing recession risks and expectations that inflation is peaking. Among single stocks, Meta Platforms dropped 4.1% after the European Commission said it could impose a fine of up to 10% of its annual global turnover if evidence showed an infringement of the EU's antitrust laws, while L3 Harris Technologies fell 3.7% after saying it would buy Aerojet Rocketdyne Holdings for $4.7 billion. On the economic data front, US homebuilder sentiment fell for a 12th month as high mortgage rates weighed on affordability. The Dow Jones fell 163 points, or 0.5%, to 32,758, the S&P 500 lost 0.9% to 3,818, and the Nasdaq Composite shed 1.5% to 10,546. Stocks are on track to end December lower after two consecutive weeks in the red.",
  "Fundamental_Labels": {
    "Timestamp": "2022/12/19",
    "Type": "NEWS",
    "Sentiment_Polarity": "BEARISH",
    "Sentiment_Probability": "0.9765625"
  },
  "Macro_Context": {
    "Market_Regime": "BULL",
    "Macro_Event": [
      {
        "anchor": "Fed hawkish tightening amid recession risks",
        "confidence": "High"
      }
    ]
  },
  "Risk_Exposure": {
    "Risk_Category": [
      "Market Risk",
      "Operational Risk"
    ],
    "Risk_Level": "HIGH RISK",
    "Risk_Triplets": [
      {
        "risk_category": "Market Risk",
        "risk_source": "Recession concerns and hawkish Fed tone",
        "affected_object": "US stock market indices"
      },
      {
        "risk_category": "Operational Risk",
        "risk_source": "Potential EU antitrust fine",
        "affected_object": "Meta Platforms"
      },
      {
        "risk_category": "Market Risk",
        "risk_source": "Acquisition announcement",
        "affected_object": "L3 Harris Technologies"
      }
    ]
  },
  "Causal_Relationship": {
    "Chain_Complexity": "SINGLE",
    "Causal_Paths": [
      {
        "Chain_ID": 1,
        "Cause": "Fed took a more hawkish tone than expected",
        "Mechanism": "raising concerns over a recession in the US",
        "Effect": "US stocks closed lower for a fourth straight session"
      }
    ]
  },
  "Entity_Relationship": {
    "Entity_Recognition": [
      {
        "Entity_Name": "Wall Street",
        "Type": "LOC"
      },
      {
        "Entity_Name": "United States",
        "Type": "LOC"
      },
      {
        "Entity_Name": "US",
        "Type": "LOC"
      },
      {
        "Entity_Name": "Fed",
        "Type": "ORG"
      },
      {
        "Entity_Name": "Powell",
        "Type": "PER"
      },
      {
        "Entity_Name": "Meta Platforms",
        "Type": "ORG"
      },
      {
        "Entity_Name": "European Commission",
        "Type": "ORG"
      },
      {
        "Entity_Name": "EU",
        "Type": "ORG"
      },
      {
        "Entity_Name": "L3 Harris Technologies",
        "Type": "ORG"
      },
      {
        "Entity_Name": "Aerojet Rocketdyne Holdings",
        "Type": "ORG"
      },
      {
        "Entity_Name": "Dow Jones",
        "Type": "FIN"
      },
      {
        "Entity_Name": "S&P 500",
        "Type": "FIN"
      },
      {
        "Entity_Name": "Nasdaq Composite",
        "Type": "FIN"
      }
    ],
    "Relation_Extraction": [
      {
        "Relation": "ACQUISITION_MERGER",
        "Clean_Entities": [
          "L3 Harris Technologies",
          "Aerojet Rocketdyne Holdings"
        ]
      },
      {
        "Relation": "REGULATORY_ACTION",
        "Clean_Entities": [
          "European Commission",
          "Meta Platforms"
        ]
      }
    ]
  },
  "Wording_Difference": {
    "Rhetoric_Type": "OBJECTIVE_FACT",
    "Fuzzy_Text_Confidence": "HIGH"
  }
}
```

---

### FinAtlas-Bench EDA

#### 1. Time Span and Yearly Distribution

- Time span: `2018-01-01` to `2024-09-01`.
- All 2,000 records have valid `Timestamp` values; no missing or blank timestamps were found.
- Yearly sample counts map:

![FinAtlas-Bench yearly sample counts](yearly_sample_counts.png)

#### 2. Crisis-Window Coverage of S&P 500 Stress Episodes

The file starts in 2018, so earlier S&P 500 stress episodes (e.g., the 2008 global financial crisis and the 2000 dot-com bust) are **outside the covered range**. For episodes inside 2018-2024, we report both the number of records whose timestamps fall inside the window and the number of records whose text or event labels explicitly mention the episode. Direct mentions are estimated with a case-insensitive keyword scan over `Text`, `Macro_Event`, `Risk_Categories`, and `Risk_Triplets`, so the counts should be treated as lower-bound evidence rather than an exact classification.

| S&P 500 stress episode | Analysis window | Samples in window | Direct mentions | Coverage assessment |
| --- | --- | ---: | ---: | --- |
| 2018 Q4 sell-off / near bear market | 2018-10-01 to 2018-12-31 | 44 | 0 | Temporal coverage exists, but macro-event content is not concentrated in this window. |
| 2019 US-China trade escalation and growth scare | 2019-05-01 to 2019-08-31 | 37 | 0 | Weak content coverage. |
| 2020 COVID-19 pandemic shock and market crash | 2020-02-01 to 2020-04-30 | 604 | 251 | Strong coverage; 2020-02/03 carries the largest sample mass. |
| 2021 Evergrande / Fed taper risk-off episode | 2021-08-01 to 2021-12-31 | 108 | 3 | Only limited general inflation/Fed-related text; no direct Evergrande mention. |
| 2022 Russia-Ukraine shock, Fed tightening, and S&P 500 bear market | 2022-01-01 to 2022-12-31 | 594 | Russia/Ukraine/sanctions: 42; inflation/Fed: 172 | Strong temporal and thematic coverage. |
| 2023 banking-system stress (Silicon Valley Bank / Credit Suisse) | 2023-03-01 to 2023-04-30 | 16 | 0 | Almost no direct coverage: only one retrospective mention (2023-09-29) exists in the whole file. |
| 2023 US debt-ceiling standoff / Fitch downgrade | 2023-05-01 to 2023-08-31 | 28 | 0 | No direct coverage found. |
| 2024 yen-carry-trade unwind / global risk-off | 2024-08-01 to 2024-09-30 | 3 | 0 | No direct coverage; 2024 has only 34 samples in total. |

Overall, the 2018-2024 replaced benchmark strongly covers the **2020 COVID-19 shock** and the **2022 S&P 500 bear market**, while the **2023 SVB/Credit Suisse banking-stress window** is temporally sparse and not directly represented by event-specific text.

#### 3. Source Composition and Time Coverage

| Type | Samples | Share |
| --- | ---: | ---: |
| `NEWS` | 1,173 | 58.65% |
| `TWEETS` | 494 | 24.70% |
| `REPORT` | 333 | 16.65% |
| **Total** | **2,000** | **100%** |

Note: the actual `Type` value in the parquet is `TWEETS` (plural), while the Data Dictionary above writes `TWEET`. The parquet spelling is used here.

| Type | First sample | Last sample | Month-level coverage |
| --- | --- | --- | --- |
| `NEWS` | `2018-01-01` | `2023-12-01` | 62 non-empty months out of a 72-month span; no 2024 samples. |
| `TWEETS` | `2018-01-24` | `2023-01-13` | 22 non-empty months out of a 61-month span; sparse and no 2024 samples. |
| `REPORT` | `2018-01-10` | `2024-09-01` | 37 non-empty months out of an 81-month span; the only source with 2024 samples. |

To show the monthly sample-volume perspective, the faceted line chart below retains zero months and gives each source its own panel, so the much larger `NEWS` peaks do not compress `TWEETS` and `REPORT`.

![Monthly sample count by source](source_monthly_counts_lines.png)

#### 4. Dimension Label Coverage and Distribution

The coverage table below counts a list field as valid when it contains at least one element. `NONE` and `NO RISK` are treated as valid labels, not missing values.

| Field | Valid / non-empty rows | Coverage |
| --- | ---: | ---: |
| `Sentiment_Polarity` | 2,000 | 100.0% |
| `Macro_Market_Regime` | 2,000 | 100.0% |
| `Macro_Event` | 846 | 42.3% |
| `Risk_Categories` | 2,000 | 100.0% |
| `Risk_Level` | 2,000 | 100.0% |
| `Risk_Triplets` | 2,000 | 100.0% |
| `Causal_Chain_Complexity` | 2,000 | 100.0% |
| `Causal_Paths` | 800 | 40.0% |
| `Entity_Recognition` | 1,960 | 98.0% |
| `Entity_Relation_Extraction` | 556 | 27.8% |
| `Wording_Rhetoric_Type` | 2,000 | 100.0% |
| `Wording_Fuzzy_Text_Confidence` | 2,000 | 100.0% |

`Risk_Categories` and `Risk_Triplets` are always present as lists, but 651 rows are labeled as no-risk (`None`); 1,349 rows (67.45%) contain at least one actual risk category.

**Sentiment polarity**

| `Sentiment_Polarity` | Count | Share |
| --- | ---: | ---: |
| `NEUTRAL` | 1,156 | 57.80% |
| `BEARISH` | 732 | 36.60% |
| `BULLISH` | 112 | 5.60% |

**Macro context**

| `Macro_Market_Regime` | Count | Share |
| --- | ---: | ---: |
| `BULL` | 1,240 | 62.00% |
| `BEAR` | 760 | 38.00% |

**Risk exposure**

| `Risk_Level` | Count | Share |
| --- | ---: | ---: |
| `NO RISK` | 651 | 32.55% |
| `LOW RISK` | 855 | 42.75% |
| `HIGH RISK` | 494 | 24.70% |

| `Risk_Categories` | Rows containing category | Row hit rate |
| --- | ---: | ---: |
| `Market Risk` | 993 | 49.65% |
| `Operational Risk` | 660 | 33.00% |
| `Liquidity Risk` | 190 | 9.50% |
| `Credit Risk` | 172 | 8.60% |
| `None` | 651 | 32.55% |

One record can contain multiple risk categories, so the row hit rates do not sum to 100%.

**Causal relationship**

| `Causal_Chain_Complexity` | Count | Share |
| --- | ---: | ---: |
| `NONE` | 1,200 | 60.00% |
| `SINGLE` | 551 | 27.55% |
| `MULTIPLE` | 249 | 12.45% |

`Causal_Paths` is non-empty for the 800 records with `SINGLE` or `MULTIPLE` causal chains.

**Entity relationship**

| Field | Count | Share |
| --- | ---: | ---: |
| `Entity_Recognition` non-empty | 1,960 / 2,000 | 98.00% |
| `Entity_Relation_Extraction` non-empty | 556 / 2,000 | 27.80% |

Entity-type distribution is based on all extracted entity elements (21,874 in total):

| Entity type | Elements | Element share |
| --- | ---: | ---: |
| `ORG` | 7,557 | 34.55% |
| `FIN` | 7,388 | 33.78% |
| `LOC` | 3,629 | 16.59% |
| `PER` | 3,300 | 15.09% |

Relation-type distribution is based on all extracted relations (1,053 in total):

| Relation type | Elements | Element share |
| --- | ---: | ---: |
| `REGULATORY_ACTION` | 256 | 24.31% |
| `INVESTMENT_FINANCING` | 187 | 17.76% |
| `ACQUISITION_MERGER` | 164 | 15.57% |
| `COMPETITION` | 147 | 13.96% |
| `SUPPLY_CHAIN` | 85 | 8.07% |
| `STRATEGIC_PARTNERSHIP` | 75 | 7.12% |
| `EXECUTIVE_CHANGE` | 72 | 6.84% |
| `LEGAL_DISPUTE` | 67 | 6.36% |

**Wording difference**

| `Wording_Rhetoric_Type` | Count | Share |
| --- | ---: | ---: |
| `OBJECTIVE_FACT` | 1,036 | 51.80% |
| `SUBJECTIVE_PROJECTION` | 560 | 28.00% |
| `EMOTIONAL_HYPERBOLE` | 344 | 17.20% |
| `AMBIGUOUS_RUMOR` | 60 | 3.00% |

The panic/rumor signal combined by `EMOTIONAL_HYPERBOLE` + `AMBIGUOUS_RUMOR` covers 404 records (20.20%).

| `Wording_Fuzzy_Text_Confidence` | Count | Share |
| --- | ---: | ---: |
| `HIGH` | 1,526 | 76.30% |
| `MEDIUM` | 395 | 19.75% |
| `LOW` | 79 | 3.95% |




## 🛠️Training Pipeline

The training pipeline is illustrated using the **Causal Relationship** dimension as an example, covering English financial **News**, **Tweets**, and **Reports**. The four scripts connect three-model consensus annotation, structured chain-of-thought (CoT) generation, supervised fine-tuning (SFT), and Group Relative Policy Optimization (GRPO).

### Stages and Files

Keep the four scripts in the same directory. Default paths are relative to that directory. In the table below, `<TYPE>` is exactly `News`, `Tweets`, or `Reports`.

| Stage / Script | Input | Output / Behavior |
| --- | --- | --- |
| 1. [three_model_consensus.py](three_model_consensus.py) | `.parquet` files directly inside `data/<type>/`, where `<type>` is lowercase; each file needs a `Text` or `text` column. | Three models vote on `Chain_Complexity`. Unanimous 3:0 samples enter `causality/<TYPE>_Causality_allocate.parquet` while the matching SFT category has room. Valid 2:1 samples and unanimous samples whose SFT category is full enter `causality/FinAtlas_<TYPE>_Causality_GRPO.parquet`. |
| 2. [cot_process.py](cot_process.py) | `causality/<TYPE>_Causality_allocate.parquet` | Generates a rationale for the supplied annotation and saves `causality/FinAtlas_Causality_SFT_<TYPE>.parquet`, containing exactly two string columns: `prompt` and `response`. |
| 3. [sft_train.py](sft_train.py) | The three CoT files: `FinAtlas_Causality_SFT_News.parquet`, `FinAtlas_Causality_SFT_Tweets.parquet`, and `FinAtlas_Causality_SFT_Reports.parquet`, under `causality/`. | Combines their `prompt` / `response` pairs and trains a LoRA adapter. Loss is computed on the response tokens. The final adapter, tokenizer, and training metadata are saved to `outputs/Causality_LoRA/final/`. |
| 4. [grpo_train.py](grpo_train.py) | The three GRPO files: `FinAtlas_News_Causality_GRPO.parquet`, `FinAtlas_Tweets_Causality_GRPO.parquet`, and `FinAtlas_Reports_Causality_GRPO.parquet`, under `causality/`, plus the SFT adapter in `outputs/Causality_LoRA/final/`. | By default, samples up to 1,000 eligible rows per text type, combines all three sources into one dataset of up to 3,000 rows, and shuffles it with seed 42. Jointly trains the SFT adapter on this combined dataset using GRPO format and causality rewards. Saves the final adapter and tokenizer to `outputs/Causality_GRPO/final/`. |


### Configuration and Defaults

Editable settings, with comments, are grouped near the top of each script.

| Script | Settings to Review | Current Defaults |
| --- | --- | --- |
| `three_model_consensus.py` | `TEXT_TYPE`, `INPUT_DIR`, `OUTPUT_DIR`, provider API keys, `*_MODEL`, `*_BASE_URL`, `CAPS`, `GRPO_CAP`, `CONCURRENCY_LIMIT` | `News`; `deepseek-chat`, `qwen-plus`, `glm-4-flash`; SFT category caps: `SINGLE=500`, `MULTIPLE=300`, `NONE=200`; GRPO collection capacity: 8,000 per text type; 16 concurrent texts. |
| `cot_process.py` | `TEXT_TYPE`, `INPUT_PARQUET`, `OUTPUT_PARQUET`, `API_KEY`, `MODEL_NAME`, `BASE_URL`, `CONCURRENCY_LIMIT` | Inherits the text type, output directory, and DeepSeek API settings from stage 1; 16 concurrent generation requests. These settings can be overridden in the file. |
| `sft_train.py` | `MODEL_PATH`, `MODEL_CLASS`, `DATA_PATHS`, `OUTPUT_DIR`, epochs, learning rate, batch size, gradient accumulation, sequence length, LoRA settings | Local `models/Qwen3.5-9B` checkpoint; 10 epochs; learning rate `2e-5`; batch size 2; gradient accumulation 8; sequence limit 8,192 tokens. |
| `grpo_train.py` | `SFT_LORA_PATH`, `DATA_CONFIG`, `OUTPUT_DIR`, epochs, learning rate, batch size, gradient accumulation, `GROUP_SIZE`, generation limits, reward weights | Continues `outputs/Causality_LoRA/final/`; up to 1,000 eligible samples each from News, Tweets, and Reports; 8 epoch; learning rate `5e-6`; batch size 1; gradient accumulation 4; 4 completions per prompt. |

Adjust dataset sizes as needed: stage-1 `CAPS` and `GRPO_CAP` control collection, while stage-4 `DATA_CONFIG` controls training sampling after validity and length filtering. If a source has fewer eligible rows than its training cap, all available rows are used; set its cap to `None` to use all eligible rows. The default seed is 42.

To change an annotation model, edit its model ID; changing providers also requires the corresponding API base URL and key. Stages 1 and 2 use OpenAI-compatible chat endpoints. Enter API keys locally or through the supported environment variables, and retain placeholders in published code. Stage-1 concurrency counts texts, with three voting requests per text; stage-2 concurrency counts individual generation requests.

Training settings reflect experiments on a single **NVIDIA A800 80 GB**. Adjust batch size and gradient accumulation to suit your hardware. For SFT, the default effective batch size on one GPU is `2 × 8 = 16`. For GRPO, the global completion batch must be divisible by `GROUP_SIZE`. GRPO inherits the base-model and chat settings from SFT, and its LoRA configuration comes from the trained SFT adapter. Both final outputs are adapters used with the configured base model.

### Running the Pipeline

Use Python 3.10+ in a separate virtual environment. For GPU training, first install a CUDA-enabled PyTorch build matching your machine using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), within the version bounds in [requirements.txt](requirements.txt). Then install the repository dependencies from its root directory:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` covers dataset loading and all four scripts. It constrains the training libraries for the current code, including [Qwen3.5 support in Transformers 5.2](https://github.com/huggingface/transformers/tree/v5.2.0/src/transformers/models/qwen3_5) and [TRL 0.29.1's SFT reference-adapter snapshot](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py). With the default positive KL coefficient, GRPO checks that its frozen reference adapter matches the initial SFT adapter. These constraints are not an exact lockfile of the paper's experimental environment; a complete GPU training run has not been validated against this dependency file.

1. Place raw Parquet files in `data/news/`, `data/tweets/`, and `data/reports/`, or change `INPUT_DIR`. Configure API credentials and model settings.
2. Set `TEXT_TYPE = "News"` in `three_model_consensus.py`, then run the following commands from the repository directory:

   ```bash
   python three_model_consensus.py
   python cot_process.py
   ```

3. Change `TEXT_TYPE` to `"Tweets"` and repeat both commands, then do the same for `"Reports"`. CoT inherits this selection unless explicitly overridden. The first two stages process **one text type per run**; prepare all three sources before combined training.
4. Configure the local base-model path and SFT settings, then run:

   ```bash
   python sft_train.py
   python grpo_train.py
   ```

Stage 1 uses type-specific resume logs. Use a fresh output directory when changing the corpus, annotation models, prompts, or collection quotas. CoT regenerates and replaces its output file. GRPO defaults to resuming the latest checkpoint in its output directory; set `RESUME_FROM_CHECKPOINT = None` for a fresh run. Review generated annotations and rationales before training: the structural checks do not establish that every causal claim is supported by the source text.

## 🗂️ Repository Structure

This repository contains the datasets, documentation, and causal relationship training scripts:

| File | Description |
| --- | --- |
| `FinAtlas_Bench.parquet` | The unified FinAtlas-Bench benchmark dataset (2,000 independent samples). |
| `FinAtlas_Macro_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Macro Context dimension. |
| `FinAtlas_Risk_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Risk Exposure dimension. |
| `FinAtlas_Casuality_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Causal Relationship dimension. |
| `FinAtlas_Entity_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Entity Relationship dimension. |
| `FinAtlas_Wording_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Wording Difference dimension. |
| `FinAtlas_Experiment_settings.md` | Experiments setting  detail. |
| [three_model_consensus.py](three_model_consensus.py) | Three-model causal annotation and allocation of SFT/GRPO samples. |
| [cot_process.py](cot_process.py) | CoT generation and conversion to `prompt` / `response` SFT pairs. |
| [sft_train.py](sft_train.py) | LoRA SFT using the News, Tweets, and Reports CoT datasets. |
| [grpo_train.py](grpo_train.py) | GRPO training initialized from the trained causality SFT adapter. |
| [requirements.txt](requirements.txt) | Python dependencies and version constraints for dataset loading and the complete training pipeline. |
| `README.md` | Dataset documentation, schema, a sample record, and the four-stage training workflow. |

---

## ⚖️ Data Sources & Acknowledgements

The FinAtlas datasets are derived and reconstructed from the following open-source financial corpora. We strictly respect their original licenses and distribute our annotations under non-commercial terms.

| Original Dataset | Source / Reference | Url / Cite |
| --- | --- | --- |
| **financial-news-multisource** | Brianferrell787 | https://huggingface.co/datasets/Brianferrell787/financial-news-multisource |
| **financial-reports-sec** | JanosAudran | https://huggingface.co/datasets/JanosAudran/financial-reports-sec |
| **sp500-edgar-10k-markdown** | BEE-spoke-data | https://huggingface.co/datasets/BEE-spoke-data/sp500-edgar-10k-markdown |
| **sp500_dataset_earnings_sec** | hfmlsoc | https://huggingface.co/datasets/hfmlsoc/sp500_dataset_earnings_sec |
| **financial-tweets** | StephanAkkerman | https://huggingface.co/datasets/StephanAkkerman/financial-tweets |
| **stock_market_tweets** | mjw | https://huggingface.co/datasets/mjw/stock_market_tweets |
| **StockTwits** | StockTwits | https://www.mdpi.com/2571-5577/4/1/13 |
| **SP500-Financial-News-Articles-Time-Series** | KrossKinetic | https://huggingface.co/datasets/KrossKinetic/SP500-Financial-News-Articles-Time-Series |
| **Bloomberg_Financial_News** | danidanou | https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News |
| **Financial Tweets** | Financial Tweets | https://www.kaggle.com/datasets/davidwallach/financial-tweets |
| **S&P 500 with Financial News Headlines** | S&P 500 with Financial News Headlines | https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024 |
---

## 📖 Citation

If you use FinAtlas or the FINS framework in your research, please cite our paper:

```bibtex
coming soon

```

## 📜 License

* **Code:** Any code or configuration files in this repository are licensed under the [MIT License](https://www.google.com/search?q=LICENSE-CODE).
* **Dataset:** The FinAtlas datasets are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://www.google.com/search?q=LICENSE-DATA). They are intended strictly for academic and non-commercial research purposes.
