# MuDi-HTR

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## Project Title
**MuDi-HTR: Multi-Modal Digital Handwriting Text Recognition**

## Abstract
MuDi-HTR is a research-oriented framework for combining online stroke trajectories and offline handwritten image features for robust handwriting text recognition.

## Installation
```bash
git clone https://github.com/kanitafro/MuDi-HTR.git
cd MuDi-HTR
pip install -e .
```

## Data Preparation
- Place source assets under `data/`.
- Keep raw material in `data/raw/` (ignored by git).
- Implement dataset parsing inside `preprocessing/`.

## Training (Online & Offline)
- Online pipeline modules live in `models/online/`.
- Offline pipeline modules live in `models/offline/`.
- Use scripts in `scripts/` to launch experiments.

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
