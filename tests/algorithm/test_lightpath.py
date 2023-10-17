from tests.fixtures import rng

import numpy as np
import torch
import yaml
import pytest

from pfmatch.backend import device
from pfmatch.flashmatch_types import QCluster
from pfmatch.algorithm.lightpath import LightPath
from tempfile import NamedTemporaryFile
from tests.fixtures import rng

def writable_temp_file(suffix=None):
    return NamedTemporaryFile('w', suffix=suffix, delete=False).name


@pytest.fixture
def lightpath():
    detector_specs = {'MIPdEdx': 2.2, 'LightYield': 24000} 
    return LightPath(detector_specs)

def test_lightpath_configure_from_yaml(rng, lightpath):
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

def test_lightpath_fill_qcluster(rng, lightpath):
    # test for list input
    random = rng.random()*99 + 1
    
    pt1 = [0,0,0]
    pt2 = [0,random,0]
    qcluster = QCluster()
    lightpath.fill_qcluster(pt1, pt2, qcluster)
    
    # test for numpy array input
    pt1 = np.array([0,0,0])
    pt2 = np.array([0,random,0])
    lightpath.fill_qcluster(pt1, pt2, qcluster)
    
    # test for torch tensor input
    pt1 = torch.tensor([random,0,random], device=device, dtype=torch.float32)
    pt2 = torch.tensor([0,random,0], device=device, dtype=torch.float32)
    lightpath.fill_qcluster(pt1, pt2, qcluster)

    # test output:
    #   - charge must be positive
    #   - inputs that are all in the same direction should result in outputs
    #     in the same direction
    randx = rng.random()*99 + 1
    randy = rng.random()*99 + 1
    randz = rng.random()*99 + 1
    track_length1 = np.array([[0, 0, 0], [randx, randy, randz]])
    
    # create a track with more than two points
    length = int(rng.random()*98)+2
    track_lengthN = np.array([[0, 0, 0]] + [[i*randx, i*randy, i*randz] for i in range(1, length)])

    for track in [track_length1, track_lengthN]:
        # create a qcluster from the track
        qcluster = QCluster()
        for i in range(len(track)-1):
            lightpath.fill_qcluster(track[i], track[i+1], qcluster)
            
        assert all(qcluster.qpt_v[:, 3] >= 0), "Charge must be positive"

        directions = [arr/np.linalg.norm(arr) for arr in qcluster.qpt_v[:, :3]]
        cumdot = [torch.dot(directions[i], directions[i-1]).item() for i in range(len(directions)-1)]
        init_direction = [(track[1]-track[0])/np.linalg.norm(track[1]-track[0])]
        cumdot = [np.dot(init_direction, directions[0].cpu().numpy()).item()] + cumdot
        assert np.allclose(cumdot, np.ones_like(cumdot)), "Segments not in uniform direction"
        
    # test for zero length track
    qcluster = QCluster()
    lightpath.fill_qcluster(np.array([0,0,0]), np.array([0,0,0]), qcluster)
    assert qcluster.qpt_v.shape == (0,)

def test_lightpath_fill_qcluster_from_track(rng, lightpath):
    randx = rng.random()*99 + 1
    randy = rng.random()*99 + 1
    randz = rng.random()*99 + 1
    track_length1 = np.array([[0, 0, 0], [randx, randy, randz]])
    
    # create a track with more than two points
    length = int(rng.random()*98)+2
    track_lengthN = np.array([[0, 0, 0]] + [[i*randx, i*randy, i*randz] for i in range(1, length)])
    for track in [track_length1, track_lengthN]:
        qcluster = lightpath.make_qcluster_from_track(track)
        
        assert qcluster.qpt_v[0, 3] == qcluster.qpt_v[-1, 3] == 0.0, "expected zero charge at endpoints"
        assert all(qcluster.qpt_v[:, 3] >= 0), "charger must be positive"
        assert np.allclose(qcluster.qpt_v[0, :3], track[0]), "expected first point to be the first point of the track"
        assert np.allclose(qcluster.qpt_v[-1, :3], track[-1]), "expected last point to be the last point of the track"
        
    # test track of 0 length
    with pytest.raises(ValueError):
        lightpath.make_qcluster_from_track([[0,0,0]])
