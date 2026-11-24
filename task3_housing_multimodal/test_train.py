"""Regression checks for leakage, checkpoint selection and singleton batches."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import train


class HousingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disjoint_reproducible_splits_and_training_only_scalers(self):
        frame = train.generate_data(100)
        parts = train.split_indices(len(frame))
        self.assertEqual({k: len(v) for k, v in parts.items()}, {'train': 70, 'validation': 15, 'test': 15})
        self.assertEqual(len(set(np.concatenate(list(parts.values())))), 100)
        for key in parts:
            np.testing.assert_array_equal(parts[key], train.split_indices(100)[key])
        features, targets = train.fit_preprocessors(frame, parts['train'])
        altered = frame.copy()
        altered.loc[np.concatenate([parts['validation'], parts['test']]), :] = 1e9
        other_features, other_targets = train.fit_preprocessors(altered, parts['train'])
        np.testing.assert_array_equal(features.mean_, other_features.mean_)
        np.testing.assert_array_equal(features.scale_, other_features.scale_)
        np.testing.assert_array_equal(targets.mean_, other_targets.mean_)
        np.testing.assert_array_equal(targets.scale_, other_targets.scale_)

    def test_validation_selects_best_checkpoint_not_final_epoch(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(6, 1)

            def forward(self, image, features):
                return self.head(features).squeeze(-1)

        torch.manual_seed(3)
        data = TensorDataset(torch.zeros(4, 3, 4, 4), torch.randn(4, 6), torch.randn(4))
        loader = DataLoader(data, batch_size=2)
        model = TinyModel()
        snapshots = []

        def score(current, validation_loader, device):
            self.assertIs(validation_loader, loader)
            snapshots.append({key: value.clone() for key, value in current.state_dict().items()})
            return [1.0, 0.5, 2.0][len(snapshots) - 1]

        with patch.object(train, 'validation_loss', side_effect=score):
            epoch, history = train.fit_model(model, loader, loader, 'cpu', epochs=3)
        self.assertEqual(epoch, 2)
        self.assertEqual(len(history), 3)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshots[1][key])

    def test_synthetic_baselines_and_single_example_backward(self):
        frame = train.generate_data(2000)
        splits = train.split_indices(len(frame))
        features, targets = train.fit_preprocessors(frame, splits['train'])
        result = train.tabular_baselines(frame, splits, features)
        self.assertGreater(result['tabular_ridge']['r2'], 0.95)
        self.assertLess(result['tabular_ridge']['rmse'], result['mean']['rmse'])
        data = train.HousingDataset(frame, splits['validation'][:1], features, targets)
        image, tabular, target = data[0]
        model = train.MultimodalHousingModel(pretrained=False).train()
        prediction = model(image.unsqueeze(0), tabular.unsqueeze(0))
        loss = (prediction - target).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
