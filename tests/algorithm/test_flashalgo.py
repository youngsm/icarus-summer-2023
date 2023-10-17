from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest
import torch
import os
import hashlib

import yaml

from pfmatch.backend import device
from pfmatch.algorithm.flashalgo import FlashAlgo
from pfmatch.photonlib import PhotonLibrary
from tests.fixtures import rng


def writable_temp_file(suffix=None):
    return NamedTemporaryFile('w', suffix=suffix, delete=False).name

""" ------------------------ test-specific fixtures ------------------------ """
@pytest.fixture
def detector_specs():
    return {"ActiveVolumeMax": [-50.,150.,900.],
            "ActiveVolumeMin": [-360.,-180.,-900.]}
    
@pytest.fixture
def track():
    return torch.tensor([[1, 100, 300, 23000],
                         [-100, 0, 500, 18000],
                         [-300, -100, -500, 19000]], device=device, dtype=torch.float32)
    
@pytest.fixture
def num_pmt():
    """can't change because of hardcoded values in the SIREN model"""
    return 180

@pytest.fixture
def config_dict(rng, num_pmt):
    return {'PhotonLibHypothesis':
                {
                    'GlobalQE': rng.random()*1.e-4 + 1.e-5,
                    'RecoPECalibFactor': rng.random()*2 + 1,
                    'CCVCorrection': rng.random(num_pmt).tolist(),
                    'SirenPath': os.path.dirname(os.path.dirname(__file__)) + '/data/siren.ckpt'
                }
            } 

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
def flashalgo_noconfig(fake_photon_library, detector_specs):
    plib = PhotonLibrary(fake_photon_library)
    return FlashAlgo(detector_specs, photon_library=plib, cfg_file=None)

@pytest.fixture
def flashalgo_withconfig(fake_photon_library, detector_specs, config_dict):
    plib = PhotonLibrary(fake_photon_library)
    yield FlashAlgo(detector_specs, photon_library=plib, cfg_file=config_dict)
    
@pytest.fixture
def flashalgo_matrix(fake_photon_library, detector_specs, config_dict):
    out = []
    
    # with yaml config
    config_yaml = writable_temp_file(suffix='.yaml')
    yaml.dump(config_dict, open(config_yaml, 'w'))
    for cfg_file in [None, config_dict, config_yaml]:
        for plib in [None, PhotonLibrary(fake_photon_library)]:
            
            # edge case
            if cfg_file is None and plib is None:
                with pytest.raises(RuntimeError):
                    FlashAlgo(detector_specs, photon_library=plib, cfg_file=cfg_file)
                continue
            
            # all other cases
            try:
                out.append(FlashAlgo(detector_specs, photon_library=plib, cfg_file=cfg_file))
            except Exception as e:
                raise RuntimeError(f'Failed to initialize FlashAlgo with config {type(cfg_file)} and plib {type(plib)}') from e
    yield out
    os.remove(config_yaml)
        
""" --------------------------------- tests -------------------------------- """
def test_flashalgo_config_input(fake_photon_library, detector_specs, config_dict):
    # tests if configuration is properly loaded from dict or file
    plib = PhotonLibrary(fake_photon_library)
    config_yaml = writable_temp_file(suffix='.yaml')
    yaml.dump(config_dict, open(config_yaml, 'w'))
    
    dict_init = FlashAlgo(detector_specs, photon_library=plib, cfg_file=config_dict)
    file_init = FlashAlgo(detector_specs, photon_library=plib, cfg_file=config_yaml)
    for attr in ['global_qe', 'reco_pe_calib', 'qe_v', 'siren_path']:
        compare = lambda a, b: a == b
        if isinstance(getattr(dict_init, attr), torch.Tensor):
            compare = torch.allclose
        elif isinstance(getattr(dict_init, attr), (list, np.ndarray)):
            compare = np.allclose

        assert compare(getattr(dict_init, attr), getattr(file_init, attr)), \
            f'Attribute {attr} does not match between configuration dict and file initialization'

    os.remove(config_yaml)

def test_fill_estimate_shape(track, flashalgo_matrix, num_pmt):
    for flashalgo in flashalgo_matrix:
        estimate = flashalgo.fill_estimate(track)
        assert estimate.shape == (num_pmt,)

def test_backward_gradient_shape(track, num_pmt, flashalgo_matrix):
    for flashalgo in flashalgo_matrix:
        flashalgo.fill_estimate(track)
        if flashalgo.plib is None:
            with pytest.raises(RuntimeError):
                flashalgo.backward_gradient(track)
            continue
        
        gradient = flashalgo.backward_gradient(track)
        assert gradient.shape == (3, num_pmt)

def test_NormalizePosition_shape(track, flashalgo_matrix):
    for flashalgo in flashalgo_matrix:
        normalized = flashalgo.NormalizePosition(track[:,:3])
        assert normalized.shape == track[:,:3].shape