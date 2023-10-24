from numpy import size
import torch
import pytest
from pfmatch.algorithm.match_model import PoissonMatchLoss
from pfmatch.algorithm.match_modules import SirenFlash, XShift, GenFlash
from pfmatch.backend import device
# flashalgo_matrix reqs:
from tests.fixtures import flashalgo_config_dict, detector_specs, fake_photon_library, config_sirenpath
from tests.algorithm.test_flashalgo import flashalgo_matrix
# other fixtures:
from tests.fixtures import rng, torch_rng, num_pmt

@pytest.fixture
def randn(torch_rng):
    return lambda size, **kwargs: torch.randn(size, generator=torch_rng, **kwargs)

""" -------------------------------- xshift -------------------------------- """
def test_XShift_forward(randn):
    # Test XShift forward pass
    dx0 = 0.5
    dx_min = -1.0
    dx_max = 1.0
    xshift = XShift(dx0, dx_min, dx_max)
    input = 10*randn(size=(2,4), device=device)
    output = xshift(input)
    
    expected_output = torch.clone(input)
    expected_output[:,0] += dx0
    assert torch.allclose(output, expected_output), "XShift forward pass failed"
    
    # test differentiability
    output[:,0].sum().backward()
    
def test_XShift_clamp(randn):
    dx0 = 0.5
    dx_min = -1.0
    dx_max = 1.0
    xshift = XShift(dx0, dx_min, dx_max)
    
    xshift.dx.data = torch.tensor([dx_min-0.5], device=device)
    
    input = 10*randn(size=(2,4), device=device)
    output = xshift(input)
    
    expected_output = torch.clone(input)
    expected_output[:,0] += dx_min
    assert torch.allclose(output, expected_output), "XShift forward pass with clamp failed"
    
    
    xshift.dx.data = torch.tensor([dx_max+0.5], device=device)
    output = xshift(input)
    expected_output = torch.clone(input)
    expected_output[:,0] += dx_max
    assert torch.allclose(output, expected_output), "XShift forward pass with clamp failed"

    
""" ------------------------------- genflash ------------------------------- """
def test_GenFlash(flashalgo_matrix, num_pmt):
    for flashalgo,__ in flashalgo_matrix:    
        if not flashalgo.plib:
            continue

        # Test GenFlash forward pass
        input = torch.tensor([[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 10.0], [7.0, 8.0, 9.0, 10.0]],
                             device=device, requires_grad=True)
        gen_flash = GenFlash.apply
        output = gen_flash(input, flashalgo)
        assert output.shape == torch.Size([num_pmt]), "GenFlash forward pass shape mismatch"
        assert (output>=0).all(), 'negative probabilities'
        
        # Test GenFlash backward pass        
        grad_output = torch.ones(num_pmt, device=device)
        output.backward(grad_output)
        assert input.grad is not None, "GenFlash backward pass not computed"
        assert input.grad.shape == torch.Size([3, 4]), "GenFlash backward pass shape mismatch"

""" ------------------------------ sirenflash ------------------------------ """
def test_SirenFlash(flashalgo_matrix, num_pmt, randn):
    for flashalgo,__ in flashalgo_matrix:    
        if not flashalgo.slib:
            continue
        
        siren_flash = SirenFlash(flashalgo)
        # Test SirenFlash forward pass
        qpt_v = randn(size=(2,4), device=device)
        qpt_v[1, :3] = 2*qpt_v[0, :3]
        qpt_v[:, -1] = 20_000*qpt_v[:,-1].abs()
        qpt_v = qpt_v.requires_grad_(True)
        
        pred = siren_flash(qpt_v)
        assert pred.shape == torch.Size([num_pmt]), 'SirenFlash forward pass shape mismatch'
        assert (pred>=0).all(), 'SirenFlash forward pass failed'
        
        pred.sum().backward()