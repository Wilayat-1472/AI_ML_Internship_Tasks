# Task 3: Multimodal ML - Housing Price Prediction Using Images + Tabular Data

## Objective and scope
Demonstrate image/tabular feature fusion using **generated illustrations and
synthetic prices**, not real property photographs or market observations. This is
a learning prototype; no visual-feature benefit has been established by an ablation.

## Methodology
1. Generated synthetic housing dataset (2,000 samples with tabular features + house images)
2. Use a frozen pretrained ResNet-18 to extract image features (or random weights for an offline software check)
3. Built a tabular neural network for structured features (6 features)
4. Fused image and tabular features in a shared representation
5. Fit feature and log-price normalization on training records only
6. Use 70/15/15 training/validation/test splits; schedule learning and select the best checkpoint using validation only
7. Evaluate the selected checkpoint once on test data alongside training-mean and tabular Ridge baselines

## Architecture
```
Image Input (128x128) --> ResNet-18 (512-d) --|
                                               +--> Fusion Network --> Price Prediction
Tabular Input (6 features) --> MLP (32-d) ----|
```

## Historical results (old pipeline)
| Metric | Value |
|--------|-------|
| Final MAE | $343,358 |
| Final RMSE | $384,713 |
| R-squared | **-2.0692** |
| Training Loss | 167.5 → 5.1 |
| Model Parameters | 11.2M (9M trainable) |

These values belong to the original script, which fitted the scaler before
splitting and used test loss for scheduling. Negative R-squared means its squared
error exceeded the evaluation-set mean baseline. They are preserved in
`multimodal_model/metrics.json`, not presented as results of the corrected pipeline.
The current frozen-encoder architecture and target normalization require new
weights; the completed corrected fit is reported below.

## Corrected-pipeline result

A completed CPU run on 2,000 generated records used 1,400 training, 300 validation
and 300 test samples (seed 42, 15 epochs, frozen ImageNet encoder). Epoch 15 was
selected using validation loss, before test evaluation.

| Model | Test MAE | Test RMSE | Test R-squared |
|---|---:|---:|---:|
| Multimodal ResNet-18 + MLP | 43,853.61 | 55,264.27 | 0.9362 |
| Training-mean baseline | 183,171.19 | 218,841.56 | -0.0003 |
| Tabular Ridge baseline | **17,164.46** | **21,217.11** | **0.9906** |

The simple tabular baseline remains better on this generated task. The result
does not establish that images improve predictions or generalize to real homes.
It uses a new generator/split and must not be presented as a controlled
before/after improvement over the historical run.

[Metrics and validation history](reports/synthetic_v2/metrics.json),
[exact split row IDs](reports/synthetic_v2/splits.json), and
[command, versions and source/checkpoint hashes](reports/synthetic_v2/provenance.json)
are committed. Model weights remain in the ignored local run directory.

## Files
- `train.py` - Full pipeline: data generation, model building, training, evaluation
- `multimodal_model/metrics.json` - Historical metric artifact; not overwritten by new runs
- `runs/housing_multimodal/` - New checkpoint, exact split indices and versioned metrics (ignored by Git)
- `test_train.py` - Split/scaler leakage, best-checkpoint and singleton-batch regression checks
- `reports/synthetic_v2/` - Corrected 2,000-sample run metrics, splits and provenance

## How to Run
```bash
conda activate AI_ML2
cd task3_housing_multimodal
python train.py --device cpu

# Small offline execution check: not a benchmark result
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --no-pretrained --samples 40 --epochs 1 --device cpu --output-dir /tmp/housing-smoke

python -m unittest -v test_train.py
```

Install `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn` and `Pillow`
in the selected environment. Default training downloads ImageNet weights if they
are not cached. `--no-pretrained` avoids that download. Imports perform no training
or file writes. Illustrations are generated deterministically in memory.

New `metrics.json` includes the dataset/protocol, split sizes, seed, encoder mode,
selected epoch, validation history, multimodal results and same-split baselines.
`splits.json` records exact row IDs; `model.pt` includes weights and training-only
feature/log-target scaling values. Preprocessing changes make old checkpoints
incompatible. Do not compare metrics across different splits as an improvement.

## Skills Demonstrated
- Multimodal machine learning (image + tabular)
- Convolutional Neural Networks (ResNet-18)
- Feature fusion techniques
- Transfer learning with pretrained CNNs
- Regression modeling and evaluation (MAE, RMSE)
