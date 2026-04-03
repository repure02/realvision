# RealVision

RealVision is a computer vision project that detects whether an image is **AI-generated** or a **real photograph**. The goal is to build a portfolio-grade, interview-ready system that goes beyond a simple binary classifier by focusing on **generalization, robustness, calibration, interpretability, and deployment**.

## Project Goal

The main objective of RealVision is to investigate whether a model trained on diverse real photos and AI-generated images from multiple modern generators can reliably distinguish between the two, and how well it generalizes to **unseen or newer generators**.

The system will take an input image and output:
- probability of **AI-generated**
- probability of **real**
- predicted label
- confidence score
- optional explanation heatmap for suspicious regions

## Why This Project Matters

AI image generators have improved rapidly, making fake images increasingly photorealistic. This creates a challenging and evolving ML problem: not just achieving high accuracy on known data, but building a detector that remains useful as image generators change over time.

This project is designed to demonstrate:
- deep learning for computer vision
- dataset design and curation
- rigorous evaluation under distribution shift
- model calibration and threshold analysis
- failure analysis
- interpretability
- ML engineering best practices
- deployment with an interactive demo

## Current Status

This repository is in the **baseline training + evaluation phase**, with a working dataset, training pipeline, and held-out generator evaluation.

Completed so far:
- dataset assembled and processed with metadata and splits (`data/metadata/processed_metadata.csv`)
- splits implemented for `random_split` and `heldout_generator_split`
- baseline training pipeline in place (ResNet18 + transfer learning)
- checkpoints saved for both split strategies
- evaluation pipeline implemented (metrics, confusion matrix, ROC/PR curves)
- held-out generator evaluation run with threshold sweep and failure analysis

Latest held-out generator test results (from `reports/heldout_generator_split_test_predictions.csv`):
- Accuracy: **0.7318**
- Precision: **0.8684**
- Recall: **0.6346**
- F1: **0.7333**
- ROC-AUC: **0.8286**
- PR-AUC: **0.8619**

Threshold sweep (target recall 0.80) selected:
- Threshold: **0.10**
- Recall: **0.8221**
- Precision: **0.7738**
- Accuracy: **0.7570**
- F1: **0.7972**

Artifacts generated:
- Predictions: `reports/heldout_generator_split_test_predictions.csv`
- Threshold sweep: `reports/heldout_generator_split_threshold_sweep.csv`
- Chosen threshold: `reports/heldout_generator_split_chosen_threshold.txt`
- Figures: `reports/figures/heldout_generator_*`
- Failure analysis: `reports/failures/heldout_false_negatives_t025.csv`, `reports/failures/heldout_false_positives_t025.csv`

## Results

Held-out generator evaluation (test split):

| Metric | Value |
| --- | --- |
| Accuracy | 0.7318 |
| Precision | 0.8684 |
| Recall | 0.6346 |
| F1 | 0.7333 |
| ROC-AUC | 0.8286 |
| PR-AUC | 0.8619 |

Key artifacts:
- Predictions: `reports/heldout_generator_split_test_predictions.csv`
- Threshold sweep: `reports/heldout_generator_split_threshold_sweep.csv`
- Chosen threshold: `reports/heldout_generator_split_chosen_threshold.txt`
- ROC/PR + confusion matrix: `reports/figures/heldout_generator_*`

## Planned Research Question

**Can a detector trained on a diverse set of known AI generators distinguish AI-generated images from real photographs, and how well does it generalize to unseen or newer generators?**

This project will also explore:
- performance on held-out generators
- robustness to resizing and compression
- calibration quality
- failure modes by generator and image type
- adaptation when new generator data is added

## Project Roadmap

### 1. Dataset Design
- collect diverse real photographs
- collect diverse AI-generated images from modern generators
- build metadata-rich dataset
- remove duplicates and low-quality samples
- create robust train/validation/test splits
- create a held-out generator split for generalization testing

### 2. Exploratory Data Analysis
- inspect class and source balance
- analyze image sizes, formats, and aspect ratios
- visually inspect samples
- detect dataset artifacts and shortcut risks
- audit for leakage and near-duplicates

### 3. Model Training
- build a strong baseline with transfer learning
- test architectures such as ResNet, EfficientNet, or ConvNeXt
- train using reproducible configuration-based experiments
- track metrics and checkpoints

### 4. Evaluation
- accuracy, precision, recall, F1
- ROC-AUC and PR-AUC
- confusion matrix
- threshold analysis
- calibration analysis
- generator-wise evaluation
- held-out generator evaluation

### 5. Failure Analysis
- inspect false positives and false negatives
- identify generator-specific weaknesses
- understand hard cases such as photorealistic AI images or real images with synthetic-looking patterns

### 6. Interpretability
- Grad-CAM or similar visualization methods
- suspicious-region heatmaps
- analysis of what cues the model uses

### 7. Deployment
- build a Streamlit app for image upload and inference
- display label, class probabilities, confidence, and optional heatmap

## Repository Structure

```text
realvision/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── metadata/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
├── app/
├── reports/
├── tests/
├── checkpoints/
└── outputs/
````

## Tech Stack

Planned tools and libraries:

* Python
* PyTorch
* torchvision / timm
* scikit-learn
* pandas
* numpy
* matplotlib / seaborn
* OpenCV / Pillow
* Streamlit

## Key Risks and Challenges

This project is intentionally designed around real ML challenges, including:

* **dataset bias**: the model may learn source artifacts instead of true AI-vs-real cues
* **data leakage**: near-duplicate or related images may contaminate train/test splits
* **poor generalization**: strong performance on known generators may fail on unseen ones
* **misleading accuracy**: random splits may overestimate real-world performance

These risks will be treated as core parts of the project, not side issues.

## Success Criteria for v1

A successful v1 should:

* use a clean, well-structured dataset with metadata
* train a baseline image classifier successfully
* report metrics beyond accuracy
* evaluate on at least one more realistic split, such as held-out generators
* include failure analysis and basic interpretability
* provide a simple working demo

## Notes

This project is being built as a **portfolio-grade applied ML system**, not just a notebook experiment. The focus is on both model performance and the quality of the engineering and evaluation process.
