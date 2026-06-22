# MuDi-HTR

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## Project Title
**MuDi-HTR: Multi-Modal Digital Handwriting Text Recognition**

## Abstract
MuDi-HTR is a research-oriented framework for combining online stroke trajectories and offline handwritten image features for robust handwriting text recognition.

## Repository structure

```
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── cda_similarity
│   ├── __init__.py
│   └── minhash_similarity.py
├── data
│   ├── README.md
│   ├── processed
│   │   └── online
│   │       └── didi
│   │           ├── test.pt
│   │           ├── train.pt
│   │           └── valid.pt
│   └── raw
│       └── didi_dataset
│           ├── diagrams_20200131.ndjson
│           ├── diagrams_wo_text_20200131.ndjson
│           ├── dot
│           ├── png
│           └── xdot
├── demo
│   ├── __init__.py
│   └── streamlit_app.py
├── docs
├── experiments
│   ├── figures
│   ├── notebooks
│   │   └── eda.ipynb
│   └── results
├── models
│   ├── __init__.py
│   ├── fusion
│   │   ├── __init__.py
│   │   └── model.py
│   ├── offline
│   │   ├── __init__.py
│   │   └── model.py
│   └── online
│       ├── __init__.py
│       ├── config_pretrain.yaml
│       ├── dataset.py
│       ├── finetune.py
│       ├── model.py
│       ├── pretrain.py
│       ├── train.py # might be obsolete
│       ├── utils.py # might be obsolete
│       └── visualize.py # might be out of date
├── preprocessing
│   ├── __init__.py
│   ├── didi_preprocess.py
│   ├── iam_ondb_preprocess.py
│   ├── offline_preprocess.py
│   └── online_preprocess.py
├── scripts
│   ├── run_offline_pipeline.py
│   └── train.py
└── tests
    ├── __init__.py
    └── test_preprocessing.py
```

## Installation

1. Get started
```bash
git clone https://github.com/kanitafro/MuDi-HTR.git
cd MuDi-HTR
pip install -r requirements.txt
```

2. Get data:
    * DIDI - online branch
    * OpenHand-Synth - offline branch
    * DATASET3 - offline branch (finetuning)

3. Data Preparation
    - Place source assets under `data/`.
    - Keep raw material in `data/raw/` (ignored by git).
    - Implement dataset parsing inside `preprocessing/`.

4. Training (Online & Offline)
    - Online pipeline modules live in `models/online/`.
    - Offline pipeline modules live in `models/offline/`.
    - Use scripts in `scripts/` to launch experiments.

    ```
    python -m models.online.train
    ```

## Fusion
Fusion components combining online/offline signals are in `models/fusion/`.

## Demo (Streamlit)
Run the interactive demo from `demo/` with:
```bash
streamlit run demo/streamlit_app.py
```

## CDA Similarity (MinHash)
Approximate similarity utilities using MinHash are in `cda_similarity/`.

## Results
Store experiment outputs under `experiments/results/` and visualizations under `experiments/figures/`.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contributors
- @kanitafro
