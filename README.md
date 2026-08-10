# FinAtlas: A Multi-Task Financial Sentiment Analysis System

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-FinAtlas-blue)](#)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)

FinAtlas is a comprehensive system designed for complex financial sentiment understanding, extending traditional polarity classification into multi-dimensional semantic parsing. 

This repository provides the instruction-tuning and benchmark datasets reconstructed from a massive-scale raw candidate corpus of approximately 80 million financial texts.

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
ds_bench = Dataset.from_parquet("FinAtlas_Bench.parquet")
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
| `Risk_Exposure.Risk_Category` | Nature of the risk. | List of String: `Market Risk`, `Credit Risk`, `Liquidity Risk`, `Operational Risk`, `No Risk` |
| `Risk_Exposure.Risk_Level` | Severity or potential impact of the risk (high / medium / low / none). | String (e.g., `HIGH RISK`) |
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
  "Text": "Wall Street Closes Lower for a 4th Day Wall Street Closes Lower for a 4th DayUnited States Stock MarketUS stocks closed lower for a fourth straight session Monday, as concerns over a recession in the US mounted after the Fed took a more hawkish tone than expected. Chair Powell said last week the central bank would continue its monetary policy tightening campaign, despite the ongoing recession risks and expectations that inflation is peaking. Among single stocks, Meta Platforms dropped 4.1% after the European Commission said it could impose a fine of up to 10% of its annual global turnover if evidence showed an infringement of the EU's antitrust laws, while L3 Harris Technologies fell 3.7% after saying it would buy Aerojet Rocketdyne Holdings for $4.7 billion. On the economic data front, US homebuilder sentiment fell for a 12th month as high mortgage rates weighed on affordability. The Dow Jones fell 163 points, or 0.5%, to 32,758, the S&P 500 lost 0.9% to 3,818, and the Nasdaq Composite shed 1.5% to 10,546. Stocks are on track to end December lower after two consecutive weeks in the red.2022-12-19T21:12:56.66",
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

## 🗂️ Repository Structure

This repository is data-focused and contains the following files:

| File | Description |
| --- | --- |
| `FinAtlas_Bench.parquet` | The unified FinAtlas-Bench benchmark dataset (2,000 independent samples). |
| `FinAtlas_Macro_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Macro Context dimension. |
| `FinAtlas_Risk_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Risk Exposure dimension. |
| `FinAtlas_Casuality_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Causal Relationship dimension. |
| `FinAtlas_Entity_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Entity Relationship dimension. |
| `FinAtlas_Wording_SFT.parquet` | FinAtlas-SFT instruction-tuning samples for the Wording Difference dimension. |
| `README.md` | Dataset documentation, schema, and a sample record. |

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

---

## 📖 Citation

If you use FinAtlas or the FINS framework in your research, please cite our paper:

```bibtex
coming soon

```

## 📜 License

* **Code:** Any code or configuration files in this repository are licensed under the [MIT License](https://www.google.com/search?q=LICENSE-CODE).
* **Dataset:** The FinAtlas datasets are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://www.google.com/search?q=LICENSE-DATA). They are intended strictly for academic and non-commercial research purposes.
