# RealVision

RealVision is a computer vision project that detects whether an image is **AI-generated** or a **real photograph**. The goal is to build a portfolio-grade, interview-ready system that goes beyond a simple binary classifier by focusing on **generalization, robustness, calibration, interpretability, and deployment**.

## Project Goal

The main objective of RealVision is to build a detector that **generalizes to any unseen or future generator**. That means performance is judged primarily by **cross-generator robustness** rather than in-distribution accuracy.

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

This repository is now in the **cross-generator robustness phase**. The project already includes a working dataset, metadata-driven LOGO training pipeline, a final inference-model workflow, report generation, failure analysis, CLI inference, and a Streamlit demo.

Completed so far:
- dataset assembled and processed with metadata and splits in `data/metadata/processed_metadata.csv`
- current dataset size: **6,993 images**
- class balance: **3,500 AI-generated** and **3,493 real**
- real-image sources: **Pexels** and **Wikimedia Commons**
- AI generators currently represented: **DALL·E 3**, **Midjourney**, **SD2.1**, **SD3**, **SDXL**, **Ideogram v2**, **Recraft v2**, **Imagen 4**, **OpenAI 4o**, and **Hidream I1**
- metadata-aware dataloading and split support implemented in `src/training/dataset.py`
- ConvNeXt-Tiny training pipeline implemented in `src/training/train.py`
- LOGO checkpoint evaluation utility implemented in `src/training/evaluate.py`
- single-image inference implemented in `src/inference/predict.py`
- interactive demo implemented in `app/streamlit_app.py`
- LOGO summary and failure reports generated in `reports/`
- final inference checkpoint workflow implemented in `src/training/train.py`

The main technical question is no longer whether the model can fit the task at all, but how well it generalizes to **unseen generator families**.

## Reproducible Pipeline

The repository now has one stage-driven entrypoint for the full workflow:

```bash
python3 -m src.run_pipeline --stage <data_prep|logo_eval|final_inference|reports|full>
```

Recommended reviewer flow:

1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run data prep

```bash
python3 -m src.run_pipeline --stage data_prep
```

Expected outputs:
- `data/metadata/master_metadata.csv`
- `data/metadata/processed_metadata.csv`
- `data/processed/images/`

If you want collection included in the same stage, use:

```bash
python3 -m src.run_pipeline --stage data_prep --include_collection
```

3. Train the LOGO checkpoint suite

```bash
python3 -m src.run_pipeline --stage logo_eval
```

Expected outputs:
- `reports/logo_summary.csv`
- `reports/details/logo_test_*_generator_metrics.csv`
- `reports/details/logo_test_*_predictions.csv`
- `checkpoints/convnext_tiny_logo_test_<generator>_best.pt`

4. Generate the report bundle

```bash
python3 -m src.run_pipeline --stage reports
```

Expected outputs:
- `reports/dataset_specs.csv`
- `reports/logo_summary_enriched.csv`
- `reports/failures/logo_failure_summary.csv`
- `reports/figures/logo_*.png`
- `reports/baseline_manifest.json`

5. Train the final deployment checkpoint

```bash
python3 -m src.run_pipeline --stage final_inference
```

Expected outputs:
- `checkpoints/convnext_tiny_final_inference_best.pt`
- `reports/final_inference_calibration_predictions.csv`
- `reports/final_inference_generator_metrics.csv`
- `reports/final_inference_threshold_sweep.csv`
- `reports/final_inference_chosen_threshold.txt`

One-command end-to-end run:

```bash
python3 -m src.run_pipeline --stage full
```

Helpful options:
- `--logo_test_generator <name>` runs a single LOGO checkpoint instead of the full generator sweep.
- `--logo_val_generator <name>` fixes the validation generator across all LOGO runs.
- `--final_val_fraction <value>` controls how much of the random train/val pool is reserved for final-model validation.
- `--target_recall <value>` controls threshold selection for the final inference checkpoint.
- `--backfill_logo_details --backfill_missing_only` repairs missing LOGO detail CSVs before the `reports` stage.

## Experiment Tracking

Training and evaluation runs now write lightweight experiment metadata to `runs/`:

- `runs/registry.jsonl`: append-only structured run log
- `runs/registry.csv`: spreadsheet-friendly registry
- `runs/<run_id>.json`: full per-run metadata snapshot

Each tracked run includes:
- config used
- checkpoint name and path
- dataset version from `configs/data/dataset_v1.yaml`
- metadata path and row count
- split type
- metrics
- timestamp
- key artifact paths such as predictions and figures

This keeps the workflow repo-native and reproducible now, while leaving a clean upgrade path to MLflow or Weights & Biases later if you want hosted dashboards.

## Containerization

The repo now includes a `Dockerfile` for a consistent inference and demo environment.

Build the image:

```bash
docker build -t realvision .
```

Run the Streamlit demo:

```bash
docker run --rm -p 8501:8501 realvision
```

Then open `http://localhost:8501`.

Run CLI inference against a local image:

```bash
docker run --rm \
  -v "$(pwd)/path/to/image.jpg:/tmp/input.jpg:ro" \
  realvision \
  python -m src.inference.predict /tmp/input.jpg
```

What the container includes:
- `app/` for the Streamlit demo
- `src/` for inference and project code
- `configs/` for dataset/runtime settings
- `checkpoints/` for LOGO and final inference model weights
- `reports/` for LOGO summaries, calibration artifacts, and report bundles

What it intentionally excludes from the build context:
- raw and processed dataset directories
- local virtualenvs
- run registries and other local-only outputs

This keeps the image focused on deployment-style inference instead of full training.

## Final Model

This repository uses **two distinct training views** on purpose:

- `LOGO` is the official benchmark used to prove cross-generator robustness.
- `final_inference` is the single deployment model used for actual predictions.

The final model is trained on all known generators, while preserving a held-out random calibration split for threshold selection and sanity-check metrics. The canonical deployment checkpoint is:

- `checkpoints/convnext_tiny_final_inference_best.pt`

The canonical saved threshold for inference is:

- `reports/final_inference_chosen_threshold.txt`

CLI and Streamlit inference now prefer this final checkpoint automatically when it exists, and fall back to LOGO checkpoints only when a final model has not been trained yet.

## Results

The most important result in this repository is the **leave-one-generator-out (LOGO)** evaluation from the expanded 10-generator dataset in `reports/logo_summary.csv`. In this setup, one generator is fully held out at test time while the model is trained on the others, which is a better proxy for real-world robustness than a standard random split.

LOGO test results:

| Held-out generator | Test accuracy | Test recall |
| --- | --- | --- |
| DALL·E 3 | 0.8957 | 0.8966 |
| Hidream I1 | 0.8568 | 0.8967 |
| Ideogram v2 | 0.8750 | 0.9167 |
| Imagen 4 | 0.9211 | 0.9200 |
| Midjourney | 0.8734 | 0.8101 |
| OpenAI 4o | 0.9090 | 0.8900 |
| Recraft v2 | 0.9150 | 0.9333 |
| SD2.1 | 0.9092 | 0.9152 |
| SD3 | 0.9205 | 0.9267 |
| SDXL | 0.8945 | 0.8937 |

Macro-average LOGO performance on the 10-generator run:
- **average test recall: 0.8999**
- **average test accuracy: 0.8970**

What these results suggest:
- the expanded generator set substantially improved unseen-generator robustness
- the largest gains happened on the previous weak points: **DALL·E 3** and **Midjourney**
- strong generalization on the SD-family generators was preserved after adding broader generator diversity
- the model now shows consistently high held-out recall across both older and newly added generators

Before-vs-after comparison on the overlapping held-out generators:

| Held-out generator | Recall Before | Recall After | Delta | Acc Before | Acc After | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DALL·E 3 | 0.3023 | 0.8966 | +0.5943 | 0.5750 | 0.8957 | +0.3208 |
| Midjourney | 0.5288 | 0.8101 | +0.2813 | 0.7025 | 0.8734 | +0.1709 |
| SDXL | 0.8309 | 0.8937 | +0.0628 | 0.8361 | 0.8945 | +0.0583 |
| SD2.1 | 0.8628 | 0.9152 | +0.0524 | 0.8716 | 0.9092 | +0.0376 |
| SD3 | 0.9136 | 0.9267 | +0.0131 | 0.9340 | 0.9205 | -0.0135 |

Across those overlapping five generators:
- **average recall improved from 0.6877 to 0.8885**
- **average accuracy improved from 0.7838 to 0.8987**

Key artifacts:
- `reports/logo_summary.csv`
- `reports/baseline_manifest.json`
- `reports/details/logo_test_*_generator_metrics.csv`
- `reports/details/logo_test_*_predictions.csv`
- `reports/logo_summary_baseline_5gen.csv`

## Planned Research Question

**Can a detector trained on a diverse set of known AI generators generalize to *any* unseen or future generator, and how can we measure and improve that cross-generator robustness?**

This project will also explore:
- performance on held-out generators
- robustness to resizing and compression
- calibration quality
- failure modes by generator and image type
- adaptation when new generator data is added

## Best Next Move

The clearest next step is to **formalize the expanded-data result as the new baseline** and then run one focused follow-up experiment.

Why this is the best next move:
- the core data-diversity experiment already produced a strong positive result
- the biggest improvements occurred on the exact generators that were previously weakest
- changing multiple things at once now would make the story less defensible
- the repository should first lock in the stronger baseline before exploring further changes

Recommended implementation strategy:
1. Treat the 10-generator LOGO run as the new baseline.
2. Keep the before-vs-after comparison visible in the README and reports.
3. Regenerate supporting summaries and failure-analysis artifacts for the new run.
4. Run one focused follow-up experiment, preferably calibration analysis or threshold analysis on the stronger model.
5. Only after that baseline is locked in, explore architecture or loss-function changes.

Success for the next iteration should be measured primarily by:
- maintaining the stronger **macro average LOGO recall**
- preserving gains on **DALL·E 3** and **Midjourney**
- improving calibration and decision quality without losing cross-generator robustness

## Project Roadmap

### 1. Dataset Design
- collect diverse real photographs
- collect diverse AI-generated images from modern generators
- build metadata-rich dataset
- remove duplicates and low-quality samples
- create robust train/validation/test partitions for real-image support within LOGO
- create leave-one-generator-out evaluation splits for generalization testing

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
- leave-one-generator-out (LOGO) evaluation

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

## Immediate Roadmap

The highest-priority sequence from the current project state is:

1. **Promote the expanded LOGO run to the main baseline**
   - treat the 10-generator result as the primary benchmark for the project

2. **Generate updated summary artifacts**
   - rebuild reports and failure-analysis outputs so they reflect the new run consistently

3. **Add calibration analysis**
   - quantify under-confidence vs over-confidence with reliability plots or ECE on the stronger baseline

4. **Run one focused model ablation**
   - compare the current ConvNeXt-Tiny baseline against one alternative or one loss-function change

5. **Add interpretability**
   - generate Grad-CAM examples for true positives, false negatives, and hard real-image false positives

6. **Polish the demo and project narrative**
   - make the README, reports, and app all tell the same updated cross-generator robustness story

## Repository Structure

```text
realvision/
├── Dockerfile
├── README.md
├── requirements.txt
├── .gitignore
├── .dockerignore
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── metadata/
├── src/
│   ├── data/
│   ├── inference/
│   ├── training/
│   └── utils/
├── app/
├── reports/
├── runs/
├── checkpoints/
└── outputs/
````

## Tech Stack

Current tools and libraries:

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
* train the LOGO checkpoint suite successfully
* report metrics beyond accuracy
* evaluate on leave-one-generator-out splits and report cross-generator performance
* include failure analysis and basic interpretability
* provide a simple working demo

## Notes

This project is being built as a **portfolio-grade applied ML system**, not just a notebook experiment. The focus is on both model performance and the quality of the engineering and evaluation process.
