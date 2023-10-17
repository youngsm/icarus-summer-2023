from tests.fixtures import rng

import numpy as np
import torch
import yaml
import pytest

from pfmatch.flashmatch_types import QCluster
from pfmatch.algorithm.lightpath import LightPath
from tests.fixtures import rng
from tempfile import NamedTemporaryFile

def writable_temp_file(suffix=None):
    return NamedTemporaryFile('w', suffix=suffix, delete=False).name

def test_lightpath_configure_from_yaml(rng):
    detector_specs = {'MIPdEdx': 2.2, 'LightYield': 24000}
    lightpath = LightPath(detector_specs)
    assert lightpath.dEdxMIP == detector_specs['MIPdEdx']
    assert lightpath.light_yield == detector_specs['LightYield']

    # without SegmentSize
    wrong_yaml_dict = {}
    tmp = writable_temp_file(suffix='.yaml')
    yaml.dump(wrong_yaml_dict, open(tmp, 'w'))
    with pytest.raises(KeyError):
        lightpath.configure_from_yaml(tmp)

    # file that doesn't exist
    fake_tmp = 'aiwjdlwadl' + tmp
    with pytest.raises(FileNotFoundError):
        lightpath.configure_from_yaml(fake_tmp)
        
    # with SegmentSize
    yaml_dict = {'LightPath': {'SegmentSize': rng.random()}}
    yaml.dump(yaml_dict, open(tmp, 'w'))
    
    
    lightpath.configure_from_yaml(tmp)
    assert lightpath.gap == yaml_dict['LightPath']['SegmentSize']

def test_lightpath_fill_qcluster(rng):
    # create a LightPath instance
    detector_specs = {'MIPdEdx': 2.2, 'LightYield': 24000} 
    lightpath = LightPath(detector_specs)

    # create a track with two points
    randx = rng.random()*99 + 1
    randy = rng.random()*99 + 1
    randz = rng.random()*99 + 1
    
    track_length1 = [[0, 0, 0], [randx, randy, randz]]
    length = int(rng.random()*98)+2
    track_lengthN = [[0, 0, 0]] + [[i*randx, i*randy, i*randz] for i in range(1, length)]

    for track in [track_length1, track_lengthN]:
        # create a qcluster from the track
        qcluster = QCluster()
        for i in range(len(track)-1):
            lightpath.fill_qcluster(track[i], track[i+1], qcluster)
            
        assert all(qcluster.qpt_v[:, 3] >= 0), "Charge must be positive"

        directions = [arr/np.linalg.norm(arr) for arr in qcluster.qpt_v[:, :3]]
        cumdot = [torch.dot(directions[i], directions[i-1]).item() for i in range(len(directions)-1)]
        assert np.allclose(cumdot, np.ones_like(cumdot)), "Segments not in uniform direction"