# Task 3: Multimodal ML - Housing Price Prediction Using Images + Tabular Data

## Objective
Predict housing prices using both structured tabular data (area, bedrooms, location, etc.) and house images, demonstrating multimodal feature fusion.

## Methodology
1. Generated synthetic housing dataset (2,000 samples with tabular features + house images)
2. Used ResNet-18 CNN to extract visual features from house images
3. Built a tabular neural network for structured features (6 features)
4. Fused image and tabular features in a shared representation
5. Trained end-to-end with MSE loss on log-transformed prices

## Architecture
```
Image Input (128x128) --> ResNet-18 (512-d) --|
                                               +--> Fusion Network --> Price Prediction
Tabular Input (6 features) --> MLP (32-d) ----|
```

## Key Results
| Metric | Value |
|--------|-------|
| Final MAE | $343,358 |
| Final RMSE | $384,713 |
| Training Loss | 167.5 → 5.1 |
| Model Parameters | 11.2M (9M trainable) |

## Files
- `train.py` - Full pipeline: data generation, model building, training, evaluation
- `multimodal_model/` - Saved model weights, scaler, and metrics
- `housing_data/images/` - Generated house images
- `results.png` - Training curves and actual vs predicted plot

## How to Run
```bash
conda activate AI_ML2
cd task3_housing_multimodal
python train.py    # Generate data, train model, evaluate
```

## Skills Demonstrated
- Multimodal machine learning (image + tabular)
- Convolutional Neural Networks (ResNet-18)
- Feature fusion techniques
- Transfer learning with pretrained CNNs
- Regression modeling and evaluation (MAE, RMSE)
