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
├── cda_similarity/
│   ├── __init__.py
|   ├── minhash_similarity.py
│   └── synthetic_data.py
├── data/
│   ├── README.md
│   ├── processed/
│   │   └── online/
│   │   └── offline/
│   └── raw/
│       └── didi_dataset/
│       └── isgl/
│       └── iam-ondb/
├── demo/
│   ├── __init__.py
│   └── streamlit_app.py
├── docs/
├── experiments/
│   ├── figures
│   ├── notebooks
│   │   └── eda.ipynb
│   └── results
├── models/
│   ├── __init__.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── offline/
│   │   ├── __init__.py
│   │   └── model.py
│   └── online/
│       ├── __init__.py
│       ├── compute_stats.py
│       ├── config_isgl.yaml
│       ├── dataset.py
│       ├── feature_stats.pt
│       ├── generate_online_beams.py
│       ├── model.py
│       ├── train.py 
│       └── visualize.py
│       └── obsolete/  # unused code in the final version
├── runs/
├── preprocessing/
│   ├── __init__.py
│   ├── didi_preprocess.py
│   ├── iam_ondb_preprocess.py
│   ├── offline_preprocess.py
│   └── online_preprocess.py
├── scripts
│   ├── analyze_offline_data.py
│   ├── evaluate_fusion.py
│   ├── evaluate_offline.py
│   ├── minhash_scalability.py
│   ├── run_offline_pipeline.py
│   ├── run_online_pipeline.py
│   ├── train_offline.py
│   ├── train_online.py
│   └── train.py
└── tests
    ├── __init__.py
    ├── find_isgl_files.py
    ├── test_beam.py
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

    3a. **ONLINE data prep**
    
    You can call the script `run_online_pipeline.py` with parameter `--dataset` (from the root folder):
    ```
    python -m scripts.online_preprocess --dataset didi
    ```
    ```
    python -m scripts.online_preprocess --dataset iam_ondb
    ```
    ```
    python -m scripts.online_preprocess --dataset isgl
    ```

    Or, if you want to process all 3 datasets in one go, omit the argument or set it to `--dataset all`.

    3b. **OFFLINE data prep**

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
    python -m scripts.train_online
    ```

    ```
    python -m scripts.train_offline
    ```

## Fusion
Fusion components combining online/offline signals are in `models/fusion/`.

## Demo (Streamlit)
Run the interactive demo from isnide `demo/` folder with:
```
streamlit run app.py
```

## CDA Similarity (MinHash)
Approximate similarity utilities using MinHash are in `cda_similarity/`.

## Results
Store experiment outputs under `experiments/results/` and visualizations under `experiments/figures/`.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contributors
- @kanitafro
- @dzankk
