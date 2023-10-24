import numpy as np
import pytest
import torch

from pfmatch.algorithm.match_model import GradientModel
from pfmatch.algorithm.match_model import PoissonMatchLoss
from pfmatch.algorithm.match_model import EarlyStopping
from pfmatch.backend import device
# flashalgo_matrix reqs:
from tests.fixtures import flashalgo_config_dict, detector_specs, fake_photon_library, config_sirenpath
from tests.algorithm.test_flashalgo import flashalgo_matrix
# rest of fixtures:
from tests.fixtures import rng, torch_rng, num_pmt

@pytest.fixture
def randn(torch_rng):
    return lambda size, **kwargs: torch.randn(size, generator=torch_rng, **kwargs)

def test_PoissonMatchLoss(num_pmt, randn):
    loss_fn = PoissonMatchLoss()
    qpt_v = randn(size=(2,4), device=device)
    qpt_v[1, :3] = 2*qpt_v[0, :3]
    qpt_v[:, -1] = 20_000*qpt_v[:,-1].abs()

    pred = torch.pow(-8*randn(size=(num_pmt,), device=device), 10).requires_grad_(True)
    target = torch.pow(-8*randn(size=(num_pmt,), device=device), 10)
    loss = loss_fn(pred, target)
    assert isinstance(loss, torch.Tensor), 'expected tensor'
    assert loss.shape == (), 'expected scalar'

    loss.backward()

def test_GradientModel(num_pmt, flashalgo_matrix, randn):    
    for flash_algo, __ in flashalgo_matrix:
        # send to device
        model = GradientModel(flash_algo, dx0=0.05, dx_min=-10, dx_max=10)
        model.to(device)
        
        qpt_v = randn(size=(2,4), device=device)
        qpt_v[1, :3] = 2*qpt_v[0, :3]
        qpt_v[:, -1] = 20_000*qpt_v[:,-1].abs()
        
        pred = model(qpt_v)
        assert isinstance(pred, torch.Tensor), f"prediction should be a tensor, got {type(pred)}"
        assert pred.shape == torch.Size([num_pmt]), "prediction should be of shape (#pmts, )"
        assert (pred >= 0).all(), "probabilities should be positive"
        
        # test that the model is differentiable
        pred.sum().backward()
        
        # test for len(flashalgo.qe_v)==0
        flash_algo.qe_v = torch.tensor([])
        pred = model(qpt_v)
        assert isinstance(pred, torch.Tensor), f"prediction should be a tensor, got {type(pred)}"
        assert pred.shape == torch.Size([num_pmt]), "prediction should be of shape (#pmts, )"
        assert (pred >= 0).all(), "probabilities should be positive"
        
        
def test_EarlyStopping():
    early_stopping = EarlyStopping(patience=2, min_delta=3)
    losses = [10, 11, 5, 4, 3, 2, 1, 0]

    for i, loss in enumerate(losses):
        early_stopping(loss)
        if early_stopping.early_stop:
            break
        
    assert i == 4, "early stopping failed to trigger"
    
