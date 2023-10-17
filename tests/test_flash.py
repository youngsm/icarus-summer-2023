import numpy as np
import torch

from pfmatch.backend import device
from pfmatch.flashmatch_types import Flash

# import pytest fixtures; do not remove
from tests.fixtures import rng

def test_flash_fill(rng):
    
    # numpy array
    pe_v = rng.random(size=(180,))
    flash = Flash()
    flash.fill(pe_v)
    assert np.allclose(pe_v, flash.pe_v.cpu().numpy())
    
    # list
    pe_v = pe_v.tolist()
    flash = Flash()
    flash.fill(pe_v)
    assert np.allclose(pe_v, flash.pe_v.cpu().numpy())
    
    # Tensor
    pe_v = torch.tensor(pe_v, device=device)
    flash = Flash()
    flash.fill(pe_v)
    assert np.allclose(pe_v, flash.pe_v.cpu().numpy())
    

def test_flash_len(rng):
    flash = Flash()
    assert len(flash) == 0
    
    flash = Flash()
    pe_v = rng.random(size=(180,))
    flash.fill(pe_v)
    assert len(flash) == 180
    
def test_flash_sum(rng):
    flash = Flash()
    assert flash.sum() == 0
    
    pe_v = rng.random(size=(180,))
    flash.fill(pe_v)
    assert np.allclose(flash.sum(), np.sum(pe_v))