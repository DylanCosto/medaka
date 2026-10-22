"""MPS batching safety checks that also run on machines without MPS."""

import os
import unittest
from unittest import mock

import torch
from tensordict import TensorDict

from medaka.models import TorchModel


class RecordingModel(TorchModel):
    """Use CPU tensors to exercise the MPS batching control flow."""

    def __init__(self):
        super().__init__()
        self.parts = []

    def get_model_input_features(self, batch):
        return batch

    def forward(self, features):
        self.parts.append(features.shape[0])
        if isinstance(features, TensorDict):
            features = features['features']
        return features + 1


class TestMPSBatching(unittest.TestCase):
    """Reject unsafe individual samples before a device transfer or forward."""

    def test_oversized_single_sample_is_rejected_before_transfer(self):
        model = RecordingModel()
        features = mock.Mock(spec=torch.Tensor)
        features.shape = (1,)
        features.numel.return_value = 6_000_000
        with mock.patch.dict(os.environ, MEDAKA_MPS_MAX_CELLS='4000000'):
            with self.assertRaisesRegex(RuntimeError, 'single sample.*6000000'):
                model._predict_on_batch_mps(features)
        features.to.assert_not_called()
        self.assertEqual(model.parts, [])

    @mock.patch('torch.mps.empty_cache')
    def test_oversized_tensordict_sample_is_rejected(self, empty_cache):
        features = TensorDict({'features': torch.zeros(1, 6, 2)}, [1])
        model = RecordingModel()
        with mock.patch.dict(os.environ, MEDAKA_MPS_MAX_CELLS='11'):
            with self.assertRaisesRegex(RuntimeError, 'Use --cpu'):
                model._predict_on_batch_mps(features)
        self.assertEqual(model.parts, [])

    def test_invalid_cell_limits_are_rejected(self):
        for limit in ('0', '-1'):
            with self.subTest(limit=limit):
                with mock.patch.dict(os.environ, MEDAKA_MPS_MAX_CELLS=limit):
                    with self.assertRaisesRegex(ValueError, 'must be positive'):
                        RecordingModel()._predict_on_batch_mps(torch.zeros(1, 1))

    @mock.patch('torch.mps.empty_cache')
    def test_exact_boundary_and_final_partial_batch(self, empty_cache):
        model = RecordingModel()
        features = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        with mock.patch.dict(os.environ, MEDAKA_MPS_MAX_CELLS='6'):
            result = model._predict_on_batch_mps(features)
        self.assertEqual(model.parts, [2, 2, 1])
        torch.testing.assert_close(result, features + 1)
        empty_cache.assert_called_once()

    @mock.patch('torch.mps.empty_cache')
    def test_tensordict_batching_preserves_order(self, empty_cache):
        features = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        model = RecordingModel()
        batch = TensorDict({'features': features, 'other': torch.zeros(4, 1)}, [4])
        with mock.patch.dict(os.environ, MEDAKA_MPS_MAX_CELLS='6'):
            result = model._predict_on_batch_mps(batch)
        self.assertEqual(model.parts, [2, 2])
        torch.testing.assert_close(result, features + 1)


if __name__ == '__main__':
    unittest.main()
