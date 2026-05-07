# Auditing DP-ICL

This is the code repository for the paper **SnapAudit: Active Auditing of Differentially Private In-Context Learning via Snapshot-Based Simulation**.

We implement auditing code for four DP-ICL mechanisms across both text classification and text generation tasks, including active (bootstrap-based) and passive (full-process) auditing, DP flaw detection, and the MSGS distant sentence search algorithm.

# Datasets

We here list the datasets used in our paper.
|Dataset|URL|Type|
| :----: | :----: | :----: |
|AGnews|https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset|text-classification|
|TREC|https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi|text-classification|
|Samsum|https://huggingface.co/datasets/knkarthick/samsum|text-generation|
|Xsum|https://huggingface.co/datasets/EdinburghNLP/xsum|text-generation|

# Implementations

## Text Classification (`text-classification/`)

Auditing code for **Private Voting (PV)** and **Product-of-Experts (PoE)** mechanisms.

|File|Description|
| :----: | :----: |
|`auditing_utils.py`|Shared utilities: LLM setup, prompt building, DP estimators (GDP, pure-DP), Bernstein+MC error bounds, data loading|
|`pv_passive.py`|PV passive auditing (full-process): runs end-to-end PV pipeline per trial, Clopper-Pearson CI, GDP estimator|
|`pv_active.py`|PV active auditing (bootstrap): collects clean snapshots, bootstraps with Gaussian noise, two-layer Bernstein+MC error bound, GDP estimator|
|`poe_passive.py`|PoE passive auditing (full-process): runs end-to-end PoE pipeline per trial, Clopper-Pearson CI, pure-DP estimator|
|`poe_active.py`|PoE active auditing (bootstrap): collects clean utility vectors, bootstraps with Gumbel noise, two-layer Bernstein+MC error bound, pure-DP estimator|
|`pv_flaw_classical_gaussian.py`|DP flaw detection: demonstrates classical Gaussian calibration is insufficient at high epsilon|

## Text Generation (`text-generation/`)

Auditing code for **Keyword Space Aggregation (KSA)** and **Embedding Space Aggregation (ESA)** mechanisms.

|File|Description|
| :----: | :----: |
|`auditing_utils.py`|Shared utilities: LLM setup, prompt building, DP estimators (GDP, pure-DP), Bernstein+MC error bounds, data loading|
|`ksa_passive.py`|KSA passive auditing (full-process): runs end-to-end KSA pipeline per trial, Clopper-Pearson CI, pure-DP estimator|
|`ksa_active.py`|KSA active auditing (bootstrap): collects clean vote histograms, bootstraps with Gumbel noise, two-layer Bernstein+MC error bound, pure-DP estimator|
|`esa_passive.py`|ESA passive auditing (full-process): runs end-to-end ESA pipeline per trial, Clopper-Pearson CI, GDP estimator|
|`esa_active.py`|ESA active auditing (bootstrap): collects clean embeddings, bootstraps with Gaussian noise (analytic calibration, sensitivity=2/K), two-layer Bernstein+MC error bound, GDP estimator|
|`esa_flaw_wrong_sensitivity.py`|DP flaw detection: demonstrates ESA with wrong sensitivity (1.0 instead of correct 2/K) leads to privacy violation|
|`msgs.py`|MSGS (Multi-Sweep Greedy Search): finds maximally distant sentence pairs in embedding space for signal sentence selection|
