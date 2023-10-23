import pytest
import numpy as np

from pfmatch.flashmatch_types import Flash, QCluster
import torch

GLOBAL_SEED = 123

@pytest.fixture
def rng():
    return  np.random.default_rng(GLOBAL_SEED)

@pytest.fixture
def torch_rng():
    return torch.Generator().manual_seed(GLOBAL_SEED)


@pytest.fixture
def fake_flashmatch_data(rng):
    nevents = int(rng.random()*100)
    out = []
    for _ in range(nevents):
        ntracks = int(rng.random()*8)+2 # at least 2 tracks
        qcluster_v = []

        for _ in range(ntracks):
            qcluster_v.append(QCluster())
            
            qpt_v = rng.random(size=(int(rng.random()*100)+1,4))
            qcluster_v[-1].fill(qpt_v)
        nflash = int(rng.random()*10)+1 
        flash_v = []
        for _ in range(nflash):
            flash_v.append(Flash())
            
            pe_v = rng.random(180)
            flash_v[-1].fill(pe_v)
            
        out.append((qcluster_v,flash_v))
    return out
