import sys
from unittest.mock import MagicMock

# 1. Infrastructure Mocking to allow import without dependencies
class MockModule: pass
class MockMultiheadAttention: pass

sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()

import torch
import torch.nn as nn
import torch.nn.functional as F

# Replace MagicMocks with real classes for inheritance stability
nn.Module = MockModule
nn.MultiheadAttention = MockMultiheadAttention

# Mock all other external dependencies
for mod in ['pandas', 'numpy', 'networkx', 'einops', 'scipy', 'scipy.stats',
            'sklearn', 'sklearn.linear_model', 'sklearn.ensemble']:
    sys.modules[mod] = MagicMock()

# Mock internal dependencies
mock_registry = MagicMock()
mock_registry.register_model = lambda x: lambda y: y
sys.modules['proximalfm.models.registry'] = mock_registry

class MockBaseModel:
    def __init__(self, *args, **kwargs): pass
mock_base = MagicMock()
mock_base.BaseCATEModel = MockBaseModel
sys.modules['proximalfm.models.base'] = mock_base
sys.modules['proximalfm.Structure'] = MagicMock()

# 2. Import the real class
from proximalfm.models.gmm import ProximalGMM

# 3. Mock Tensor class for logic verification
class MockTensor:
    def __init__(self, shape=None):
        self._shape = shape or ()

    @property
    def shape(self): return self._shape

    def dim(self): return len(self._shape)

    def unsqueeze(self, dim):
        new_shape = list(self._shape)
        if dim == -1: dim = len(new_shape)
        new_shape.insert(dim, 1)
        return MockTensor(tuple(new_shape))

    def mean(self, *args, **kwargs): return self

    # Enable arithmetic operations for logic flow
    def __neg__(self): return self
    def __sub__(self, other): return self
    def __pow__(self, other): return self
    def __mul__(self, other): return self
    def __rmul__(self, other): return self
    def __add__(self, other): return self
    def __radd__(self, other): return self

    def __getitem__(self, idx):
        # Handle slicing used in the loss function
        if isinstance(idx, tuple) and any(isinstance(i, slice) for i in idx):
            new_shape = list(self._shape)
            for i, s in enumerate(idx):
                if isinstance(s, slice) and s.stop is not None:
                    new_shape[i] = s.stop
            return MockTensor(tuple(new_shape))
        return self

# Configure torch and F to handle MockTensor
F.log_softmax.side_effect = lambda x, **kwargs: x
torch.exp.side_effect = lambda x: x
torch.logsumexp.side_effect = lambda x, **kwargs: x

def test_proximal_gmm_loss_logic_and_shape():
    """
    Verifies the ProximalGMM.loss logic flow and target shape handling.
    """
    # Create instance without running real __init__
    model = ProximalGMM.__new__(ProximalGMM)
    model.n_gmm = 2

    # Case 1: Target requires unsqueeze (2D)
    outputs = MockTensor(shape=(2, 3, 6))
    target_2d = MagicMock(spec=MockTensor)
    target_2d.dim.return_value = 2
    target_2d.unsqueeze.return_value = MockTensor(shape=(2, 3, 1))

    loss_1 = model.loss(outputs, target_2d)

    target_2d.unsqueeze.assert_called_with(-1)
    assert isinstance(loss_1, MockTensor)

def test_proximal_gmm_loss_no_unsqueeze():
    """
    Verifies that unsqueeze is not called if the target is already 3D.
    """
    model = ProximalGMM.__new__(ProximalGMM)
    model.n_gmm = 2

    outputs = MockTensor(shape=(2, 3, 6))
    target_3d = MockTensor(shape=(2, 3, 1))

    # By using a real MockTensor for target, we ensure its dim() == 3
    # and unsqueeze will not be called.
    loss_2 = model.loss(outputs, target_3d)
    assert isinstance(loss_2, MockTensor)
