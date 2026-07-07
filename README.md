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

2. **Get data**:
    * [DIDI](https://github.com/google-research/google-research/tree/master/didi_dataset) - online branch (pretraining)
    * [IAM-OnDB](https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database) - online branch (finetuning)
    * [OpenHand-Synth](https://huggingface.co/datasets/to-be/OpenHand-Synth) - offline branch (pretraining)
    * [GNHK](https://www.goodnotes.com/gnhk) - offline branch (finetuning)

3. **Data Preparation**
    - Place source assets under `data/`.
    - Keep raw material in `data/raw/` (ignored by git).
    - Implement dataset parsing inside `preprocessing/`.

    3a. ONLINE data prep
    
    You can call the script `online_preprocess.py` with parameter `--dataset` (from the root folder):
    ```
    python -m preprocessing.online_preprocess --dataset didi
    ```
    ```
    python -m preprocessing.online_preprocess --dataset iam_ondb
    ```

    Or, if you want to process both datasets in one go:

    ```
    python -m preprocessing.online_preprocess
    ```

    3b. OFFLINE data prep

    Stage 1 uses OpenHand-Synth and stays on the existing Hugging Face preprocessing path. Stage 2 uses GNHK and is preprocessed from local JSON/image pairs.

    The GNHK dataset ships as `train` and `test` folders with one JSON file per image and the matching image file next to it.

    Convert it into the processed `.pt` format used by the offline trainer:

    ```
    python -m preprocessing.gnhk_preprocess --source-root path/to/gnhk_dataset
    ```

    This creates `data/processed/offline/gnhk/{train,test}` and matching `manifest.json` files.

    Offline training is pretrain-first by default. The GNHK stage is optional in the offline trainer and can be enabled explicitly when needed.

    For stage 1 preprocessing, keep using `python scripts/run_offline_pipeline.py` or the existing offline data-prep workflow for OpenHand-Synth.

4. **Training (Online & Offline)**
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
