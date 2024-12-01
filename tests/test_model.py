import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from S6_Model3 import Net
#from Newmodel import Net

@pytest.fixture
def model():
    return Net()

def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_batch_normalization(model):
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"

def test_dropout(model):
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_or_fc_layer(model):
    has_gap = any(isinstance(m, (torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)) for m in model.modules())
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected Layer"
    if has_gap:
        print("Model is using Global Average Pooling")
    if has_fc:
        print("Model is using Fully Connected Layer")

""" def test_forward_pass(model):
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape (1, 10), got {output.shape}"  """

