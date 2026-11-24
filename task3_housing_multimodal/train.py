"""Synthetic multimodal regression with train-only preprocessing and a locked test set.

Importing this module performs no downloads, training or file writes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

FEATURES = ('area', 'bedrooms', 'bathrooms', 'age', 'garage', 'location_score')


def generate_data(samples=2000, seed=42):
    """Generate toy records, not real property observations."""
    if samples < 20:
        raise ValueError('Use at least 20 synthetic records for three nonempty splits.')
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        'area': rng.uniform(500, 5000, samples),
        'bedrooms': rng.integers(1, 6, samples),
        'bathrooms': rng.integers(1, 4, samples),
        'age': rng.uniform(0, 50, samples),
        'garage': rng.integers(0, 3, samples),
        'location_score': rng.uniform(1, 10, samples),
    })
    frame['price'] = (frame.area * 150 + frame.bedrooms * 25000
                      + frame.bathrooms * 15000 - frame.age * 1000
                      + frame.garage * 20000 + frame.location_score * 30000
                      + rng.normal(0, 20000, samples))
    return frame


def split_indices(count, seed=42):
    train, held_out = train_test_split(np.arange(count), test_size=0.3, random_state=seed)
    val, test = train_test_split(held_out, test_size=0.5, random_state=seed)
    return {'train': train, 'validation': val, 'test': test}


def fit_preprocessors(frame, training_indices):
    training = frame.iloc[training_indices]
    features = StandardScaler().fit(training.loc[:, FEATURES])
    targets = StandardScaler().fit(np.log1p(training.price.to_numpy())[:, None])
    return features, targets


def house_image(row, seed):
    """Toy illustration encodes bedroom count; it is not a photograph."""
    rng = np.random.default_rng(seed)
    image = rng.integers(180, 255, (128, 128, 3), dtype=np.uint8)
    image[40:90, 20:110] = rng.integers([150, 120, 80], [230, 200, 160])
    yy, xx = np.mgrid[20:45, 15:115]
    image[20:45, 15:115][np.abs(yy - 42) > np.abs(xx - 65) * 0.3] = rng.integers([80, 40, 30], [140, 80, 60])
    image[60:88, 55:70] = [101, 67, 33]
    for i in range(min(int(row.bedrooms), 3)):
        image[50:65, 25 + i * 30:40 + i * 30] = [135, 206, 235]
    return Image.fromarray(image)


class HousingDataset(Dataset):
    def __init__(self, frame, indices, feature_scaler, target_scaler, training=False, seed=42):
        self.frame, self.indices, self.seed = frame, np.asarray(indices), seed
        rows = frame.iloc[self.indices]
        self.features = feature_scaler.transform(rows.loc[:, FEATURES]).astype(np.float32)
        self.targets = target_scaler.transform(np.log1p(rows.price.to_numpy())[:, None]).ravel().astype(np.float32)
        steps = [transforms.RandomHorizontalFlip(0.3), transforms.ColorJitter(brightness=0.2, contrast=0.2)] if training else []
        self.transform = transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_id = int(self.indices[index])
        image = house_image(self.frame.iloc[row_id], self.seed + row_id)
        return self.transform(image), torch.from_numpy(self.features[index]), torch.tensor(self.targets[index])


class MultimodalHousingModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn.fc = nn.Identity()
        if pretrained:
            for parameter in self.cnn.parameters():
                parameter.requires_grad = False
        self.frozen_encoder = pretrained
        # LayerNorm permits a final minibatch containing a single example.
        self.tabular_net = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.LayerNorm(64),
                                         nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(544, 128), nn.ReLU(), nn.LayerNorm(128),
                                    nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_encoder:
            self.cnn.eval()  # Frozen BatchNorm statistics must stay frozen too.
        return self

    def forward(self, image, tabular):
        return self.fusion(torch.cat([self.cnn(image), self.tabular_net(tabular)], 1)).squeeze(-1)


def validation_loss(model, loader, device):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for images, features, targets in loader:
            prediction = model(images.to(device), features.to(device))
            total += nn.functional.mse_loss(prediction, targets.to(device), reduction='sum').item()
            count += len(targets)
    if not count:
        raise ValueError('Validation split is empty.')
    return total / count


def fit_model(model, train_loader, val_loader, device, epochs=15, learning_rate=1e-4):
    """No test loader is accepted; scheduling and selection use validation only."""
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_loss, best_state, best_epoch = float('inf'), None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for images, features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(images.to(device), features.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
        val_loss = validation_loss(model, val_loader, device)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({'epoch': epoch, 'train_mse': total / len(train_loader.dataset), 'validation_mse': val_loss})
        print(f'Epoch {epoch}/{epochs}: train={history[-1]["train_mse"]:.4f}; validation={val_loss:.4f}', flush=True)
    if best_state is None:
        raise ValueError('No finite validation checkpoint was produced.')
    model.load_state_dict(best_state)
    return best_epoch, history


def regression_metrics(targets, predictions):
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite predictions; inspect target scaling and training.')
    return {'mae': float(mean_absolute_error(targets, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
            'r2': float(r2_score(targets, predictions))}


def tabular_baselines(frame, splits, feature_scaler):
    train, test = splits['train'], splits['test']
    x_train = feature_scaler.transform(frame.iloc[train].loc[:, FEATURES])
    x_test = feature_scaler.transform(frame.iloc[test].loc[:, FEATURES])
    y_train, y_test = frame.iloc[train].price, frame.iloc[test].price
    return {name: regression_metrics(y_test, estimator.fit(x_train, y_train).predict(x_test))
            for name, estimator in [('mean', DummyRegressor(strategy='mean')), ('tabular_ridge', Ridge(alpha=1.0))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--no-pretrained', action='store_true', help='Random encoder weights, without downloads; for offline software checks.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/housing_multimodal'))
    args = parser.parse_args(argv)
    if args.samples < 20 or args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('samples >= 20, epochs/batch-size >= 1 and learning-rate > 0 are required.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = generate_data(args.samples, args.seed)
    splits = split_indices(len(frame), args.seed)
    feature_scaler, target_scaler = fit_preprocessors(frame, splits['train'])
    datasets = {name: HousingDataset(frame, ids, feature_scaler, target_scaler, training=name == 'train', seed=args.seed)
                for name, ids in splits.items()}
    loaders = {name: DataLoader(data, batch_size=args.batch_size, shuffle=name == 'train') for name, data in datasets.items()}
    model = MultimodalHousingModel(pretrained=not args.no_pretrained).to(device)
    best_epoch, history = fit_model(model, loaders['train'], loaders['validation'], device, args.epochs, args.learning_rate)
    model.eval()
    predicted = []
    # First test pass occurs after all model selection has finished.
    with torch.no_grad():
        for images, features, _ in loaders['test']:
            predicted.extend(model(images.to(device), features.to(device)).cpu().tolist())
    log_prices = target_scaler.inverse_transform(np.asarray(predicted)[:, None]).ravel()
    metrics = {'schema_version': 2, 'dataset': 'synthetic_housing_illustrations',
               'protocol': 'train_validation_test_70_15_15',
               'samples': len(frame), 'split_sizes': {key: len(ids) for key, ids in splits.items()},
               'seed': args.seed, 'device': str(device), 'pretrained_encoder': not args.no_pretrained,
               'training_config': {'epochs': args.epochs, 'batch_size': args.batch_size,
                                   'learning_rate': args.learning_rate},
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'best_epoch': best_epoch,
               'multimodal': regression_metrics(frame.iloc[splits['test']].price, np.expm1(log_prices)),
               'baselines': tabular_baselines(frame, splits, feature_scaler), 'history': history}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'splits.json').write_text(json.dumps({key: ids.tolist() for key, ids in splits.items()}, indent=2) + '\n')
    torch.save({'model_state_dict': model.cpu().state_dict(), 'feature_names': list(FEATURES),
                'feature_mean': feature_scaler.mean_.tolist(), 'feature_scale': feature_scaler.scale_.tolist(),
                'target_log_mean': float(target_scaler.mean_[0]), 'target_log_scale': float(target_scaler.scale_[0]),
                'metrics': metrics, 'seed': args.seed}, args.output_dir / 'model.pt')
    print(json.dumps({key: value for key, value in metrics.items() if key != 'history'}, indent=2))
    return metrics


if __name__ == '__main__':
    main()
