# **MuDi-HTR**: Multi-Modal Digital Handwriting Text Recognition

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)


MuDi-HTR is a research-oriented framework for combining online stroke trajectories and offline handwritten image features for robust handwriting text recognition.

## Contributors
- Kanita Tafro ([@kanitafro](https://github.com/kanitafro))
- Džana Kopić ([@dzankk](https://github.com/dzankk))


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
│   ├── MuDi-HTR poster.png
│   └── report.pdf
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
├── preprocessing/
│   ├── __init__.py
│   ├── didi_preprocess.py
│   ├── gnhk_preprocess.py
│   ├── iam_ondb_preprocess.py
│   ├── inference_preprocess.py
│   ├── isgl_preprocess.py
│   ├── kaggle_handwriting_preprocess.py
│   ├── offline_preprocess.py
│   ├── online_preprocess.py
│   └── utils.py
├── scripts/
│   ├── analyze_offline_data.py
│   ├── evaluate_fusion.py
│   ├── evaluate_offline.py
│   ├── minhash_scalability.py
│   ├── run_offline_pipeline.py
│   ├── run_online_pipeline.py
│   ├── train_offline.py
│   ├── train_online.py
│   └── train.py
└── tests/
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
    * [ISGL](https://data.mendeley.com/datasets/n7kmd7t7yx/1) - online branch (accepted version)
    * [OpenHand-Synth](https://huggingface.co/datasets/to-be/OpenHand-Synth) - offline branch (pretraining)
    * [GNHK](https://www.goodnotes.com/gnhk) - offline branch (finetuning)

3. **Data Preparation**
    - Place source assets under `data/`.
    - Keep raw material in `data/raw/` (ignored by git).
    - For ISGL dataset, it's necessary to extract all zip folders and make sure the parent folder of this dataset is `data/raw/isgl/` (rename from `data/raw/ICRGL/`)

    3a. **ONLINE data prep**
    
    You can call the script `run_online_pipeline.py` with parameter `--dataset` (from the root folder):
    ```
    python -m preprocessing.online_preprocess --dataset didi
    ```
    ```
    python -m preprocessing.online_preprocess --dataset iam_ondb
    ```
    ```
    python -m preprocessing.online_preprocess --dataset isgl
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

## Demo (Streamlit)
Run the interactive demo from inside `demo/` folder with:
```
streamlit run app.py
```

## CDA Similarity (MinHash)
Approximate similarity utilities using MinHash are in `cda_similarity/`.

## Results

### Online Branch

**Method 1: Transfer Learning from DIDI**
- **Validation CER:** 0.5231
- **Test CER:** 0.8484
- **Test WER:** 1.000
- Model consistently predicted prefix `"OCRCSR"` + random characters
- Catastrophic forgetting due to domain mismatch between GraphViz and English text

**Method 2: Training from Scratch (Original)**
- **Best validation CER:** 0.9587
- **Test WER:** 1.000
- Model predicted `"OCRCSR"` prefix + random characters
- LR scheduler reduced LR to zero by epoch 22

**Method 3: Training from Scratch (Improved CNN-BiLSTM)**
- **Best validation CER:** 0.9592
- **Test CER:** 0.9600
- **Test WER:** 1.000
- Training loss dropped to 0.326 but validation CER stuck at ~0.96
- LR scheduler reduced LR to zero by epoch 31

**Method 4: ISGL Dataset (Final)**
- **Raw coordinates:** Validation CER consistently >60% (failed)
- **Derivative features (6 features):**
  - **Best validation CER:** 0.4640
  - **Best validation WER:** 0.4842
  - **Test CER:** 0.4725
  - **Test WER:** 0.4928
- Key features: Δx, Δy, log(r), angle, pen_up, pen_down
- Data augmentation: scaling (0.9–1.1), rotation (±8°), noise
- LR scheduler with patience 15 prevented premature LR drops

---

### Offline Branch

**Synthetic Pretraining (OpenHand-Synth)**
- **Best validation CER:** 0.4732
- **Best validation WER:** 0.6954
- Synthetic data alone insufficient for real-world generalization

**Intermediate Transfer (IAM FineVision)**
- **Best validation CER:** 0.7700
- **Validation WER:** 0.9968
- Domain gap too severe; complex layouts and unconstrained handwriting

**Fine-Tuning on Real-World Data (Kaggle)**
- **Best validation CER:** 0.1439
- **Best validation WER:** 0.4356
- **Test CER:** 0.1439
- Training loss: 14.7440 → 0.1960
- Training CER: 1.0750 → 0.0587
- Common substitutions: E↔I, P↔R, M↔L
- Minor spacing errors at word boundaries

---

### MinHash + LSH Similarity Search

**Scalability**
- Query time scales sublinearly:
  - 1,000 docs: 2.06 ms
  - 50,000 docs: 17.84 ms
- Suitable for real-time applications (<20 ms latency)

**Recall Evaluation**
- LSH recall at threshold 0.3: **86%**
- High accuracy with significantly reduced search space

**Threshold Tuning**

| Threshold | Recall | Query Time (ms) |
|-----------|--------|-----------------|
| 0.2 | **0.990** | 2.49 |
| 0.3 | 0.880 | 2.16 |
| 0.4 | 0.180 | 1.85 |
| 0.5 | 0.180 | 1.79 |

**Optimal Configuration:** Threshold 0.2, 128 permutations, 3-character shingles

---

### Summary

| Branch | Best Test CER | Best Test WER | Key Finding |
|--------|---------------|---------------|-------------|
| Online (ISGL) | **0.473** | 0.493 | Derivative features essential; raw coords failed |
| Offline (CRNN) | **0.144** | 0.436 | Discriminative fine-tuning critical for adaptation |
| MinHash + LSH | N/A | N/A | 99% recall, sub-20 ms latency on 50k docs |


## License
This project is licensed under the [MIT License](LICENSE).

