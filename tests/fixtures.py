import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest
import torch

from pfmatch.flashmatch_types import Flash, QCluster

GLOBAL_SEED = 123

@pytest.fixture
def rng():
    return  np.random.default_rng(GLOBAL_SEED)

@pytest.fixture
def torch_rng():
    return torch.Generator().manual_seed(GLOBAL_SEED)

def writable_temp_file(suffix=None):
    return NamedTemporaryFile('w', suffix=suffix, delete=False).name

@pytest.fixture
def config_sirenpath():
    return os.path.dirname(__file__) + '/data/siren.ckpt'

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

@pytest.fixture
def num_pmt():
    """can't change because of hardcoded values in the SIREN model"""
    return 180

@pytest.fixture
def fake_photon_library(rng, num_pmt):
    """
    h5 file has the following structure:
       - numvox: number of voxels in each dimension with shape (3,)
       - vis: 3D array of visibility values with shape (numvox, Npmt)
       - min: minimum coordinate of the active volume with shape (3,)
       - max: maximum coordinate of the active volume with shape (3,)
    """
    fake_h5 = writable_temp_file(suffix='.h5')
    with h5py.File(fake_h5, 'w') as f:
        f.create_dataset('numvox', shape=(3,), data=[10, 10, 10])
        total_numvox = np.prod(f['numvox'][:])

        # fake vis data -- random numbers uniformly distributed from 10^-7 to 10^-3
        vis = 10**rng.uniform(low=-7, high=-3, size=(total_numvox, num_pmt))
        f.create_dataset('vis', shape=(total_numvox, num_pmt), data=vis)
        
        # fake min/max data
        f.create_dataset('min', shape=(3,), data=[-400, -200, -1000])
        f.create_dataset('max', shape=(3,), data=[-35, 170, 1000])
    yield fake_h5
    os.remove(fake_h5)


@pytest.fixture
def flashalgo_config_dict(rng, num_pmt):
    return {'PhotonLibHypothesis':
                {
                    'GlobalQE': rng.random()*1.e-4 + 1.e-5,
                    'RecoPECalibFactor': rng.random()*2 + 1,
                    'CCVCorrection': rng.random(num_pmt).tolist(),
                    'SirenPath': None
                }
            } 

@pytest.fixture
def detector_specs():
    return {"ActiveVolumeMax": [-50.,150.,900.],
            "ActiveVolumeMin": [-360.,-180.,-900.]}