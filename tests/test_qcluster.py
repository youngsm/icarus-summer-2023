import numpy as np
import pytest
import torch

from pfmatch.backend import device
from pfmatch.flashmatch_types import QCluster

# pytest fixtures; do not remove
from tests.fixtures import rng, fake_flashmatch_data

def test_qcluster_copy(fake_flashmatch_data):
    (qcluster_v,_) = fake_flashmatch_data[0]
    qcluster_v_copy = [qcluster.copy() for qcluster in qcluster_v]
    for qcluster,qcluster_copy in zip(qcluster_v,qcluster_v_copy):
        assert qcluster is not qcluster_copy
        assert qcluster.qpt_v is not qcluster_copy.qpt_v
        assert np.allclose(qcluster.qpt_v.cpu().numpy(),qcluster_copy.qpt_v.cpu().numpy())
        
def test_qcluster_length(rng):
    # test for length = 0
    qcluster = QCluster()
    assert qcluster.length() == 0

    # test for length = 1*input
    qpt_v = rng.random(size=(2,4))
    qpt_v[1, :3] = 2*qpt_v[0, :3]
    qcluster.fill(qpt_v)
    
    assert qcluster.length() == np.linalg.norm(qpt_v[0, :3])
    
def test_qcluster_fill(rng):
    qcluster = QCluster()
    
    # nominal fill with numpy array
    qpt_v = rng.random(size=(int(rng.random()*10)+1,4))
    qcluster.fill(qpt_v)
    assert np.allclose(qpt_v,qcluster.qpt_v.cpu().numpy())
    
    # nominal fill with list
    qcluster.fill(qpt_v.tolist())
    assert np.allclose(qpt_v,qcluster.qpt_v.cpu().numpy())
    
    # nominal fill with torch tensor
    qcluster.fill(torch.tensor(qpt_v, device=device))
    assert np.allclose(qpt_v,qcluster.qpt_v.cpu().numpy())

    # incorrect shapes
    with pytest.raises(ValueError):
        qcluster.fill([1, 1, 1, 1])
    with pytest.raises(ValueError):
        qcluster.fill([[1, 1], [1, 1], [1, 1], [1, 1]])

def test_qcluster_qsum(rng):
    qcluster = QCluster()
    assert qcluster.sum() == 0
    
    qpt_v = rng.random(size=(int(rng.random()*100)+1,4))
    qcluster.fill(qpt_v)
    assert np.allclose(qcluster.sum(), np.sum(qpt_v[:,-1]))
    
def test_qcluster_xsum(rng):
    qcluster = QCluster()
    assert qcluster.xsum() == 0
    
    qpt_v = rng.random(size=(int(rng.random()*100)+1,4))
    qcluster.fill(qpt_v)
    assert np.allclose(qcluster.xsum(), np.sum(qpt_v[:,0]))
    
def test_qcluster_shift(rng):
    qcluster = QCluster()
    qpt_v = rng.random(size=(int(rng.random()*100)+1,4))
    qcluster.fill(qpt_v)
    
    dx = rng.random()
    qcluster_shift = qcluster.shift(dx)
    assert np.allclose(qcluster_shift.qpt_v[:,0], qcluster.qpt_v[:,0]+dx)
    assert np.allclose(qcluster_shift.qpt_v[:,1:], qcluster.qpt_v[:,1:])
    
def test_qcluster_drop(rng):
    """  x  """
    qcluster = QCluster()
    qpt_v = np.zeros((50,4))
    qpt_v[:, 0] = np.arange(1, 51)
    qpt_v[:, 1:] = rng.random(size=(50,3))
    qcluster.fill(qpt_v)
    
    # drop nothing
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(1,50)
    assert np.allclose(qcluster_drop.qpt_v.cpu().numpy(), qpt_v)
    
    # drop everything
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(0,0)
    assert len(qcluster_drop) == 0
    
    # drop half
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(1,25)
    assert len(qcluster_drop) == 25
    assert np.allclose(qcluster_drop.qpt_v.cpu().numpy(), qpt_v[:25])
    
    """  y  """
    qcluster = QCluster()
    qpt_v = np.zeros((50,4))
    qpt_v[:, 1] = np.arange(1, 51)
    qpt_v[:, [0, 2, 3]] = rng.random(size=(50,3))
    qcluster.fill(qpt_v)
    
    # drop nothing
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(-np.inf, np.inf, y_min=1, y_max=50)
    assert np.allclose(qcluster_drop.qpt_v.cpu().numpy(), qpt_v)
    
    # drop everything
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(-np.inf, np.inf, y_min=0, y_max=0)
    assert len(qcluster_drop) == 0
    
    # drop half
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(-np.inf, np.inf, y_min=1, y_max=25)
    assert len(qcluster_drop) == 25
    assert np.allclose(qcluster_drop.qpt_v.cpu().numpy(), qpt_v[:25])
    
    """  z  """
    qcluster = QCluster()
    qpt_v = np.zeros((50,4))
    qpt_v[:, 2] = np.arange(1, 51)
    qpt_v[:, [0, 1, 3]] = rng.random(size=(50,3))
    qcluster.fill(qpt_v)
    
    # drop nothing
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(-np.inf, np.inf, z_min=1, z_max=50)
    assert np.allclose(qcluster_drop.qpt_v.cpu().numpy(), qpt_v)
    
    # drop everything
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(-np.inf, np.inf, z_min=0, z_max=0)
    assert len(qcluster_drop) == 0
    
    # drop half
    qcluster_drop = qcluster.copy()
    qcluster_drop.drop(-np.inf, np.inf, z_min=1, z_max=25)
    assert len(qcluster_drop) == 25
    assert np.allclose(qcluster_drop.qpt_v.cpu().numpy(), qpt_v[:25])

def test_qcluster_iadd(fake_flashmatch_data):
    (qcluster_v,_) = fake_flashmatch_data[0]
    if len(qcluster_v) < 2:
        raise ValueError("test isn't being setup correctly; need at least 2 qcluster tracks")
    
    qcluster = QCluster()
    for qcluster_add in qcluster_v:
        qcluster += qcluster_add
    assert np.allclose(qcluster.qpt_v.cpu().numpy(), \
        np.concatenate([qcluster_add.qpt_v.cpu().numpy() for qcluster_add in qcluster_v]))
    
