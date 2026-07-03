# FinAtlas: A Multi-Task Financial Sentiment Analysis System



FinAtlas is a comprehensive system designed for complex financial sentiment understanding, extending traditional polarity classification into multi-dimensional semantic parsing. 

This repository provides the evaluation framework, as well as the instruction-tuning and benchmark datasets reconstructed from a massive-scale financial corpus.

## 📌 System Components

The FinAtlas system integrates three core components:

1. **FINS (Data Evaluation Framework):** A task-driven evaluation framework that deconstructs financial sentiment signals across five dimensions.
2. **FinAtlas-SFT (Instruction-Tuning Dataset):** The finalized high-quality supervised fine-tuning dataset. *(Note: This dataset is the final output of our multi-model consensus and structured CoT reasoning pipeline.).*
3. **FinAtlas-Bench (Benchmark):** A standardized dataset for evaluating the capabilities of various LLMs in comprehending complex financial semantics.

## 🚀 The 5-Dimensional FINS Framework

Unlike datasets focused solely on sentiment polarity, FinAtlas aligns and evaluates models across the following five dimensions:

* **Macro Context:** Identifies the external market conditions or macro environment surrounding the text.
* **Risk Exposure:** Assesses the degree to which entities, assets, or markets are exposed to negative events and abnormal volatility.
* **Causal Relationship:** Extracts and evaluates the causal links and event transmission mechanisms.
* **Entity Relationship:** Identifies financial entities and extracts their interactive relationship networks.
* **Wording Difference:** Judges how shifts in expression impact sentiment intensity and signal semantics.

![FINS Framework Diagram](assets/fins_framework.png)

## 📂 Datasets Access & Structure

We construct 5 SFT datasets corresponding to the 5 dimentions and 1 bench dataset.
