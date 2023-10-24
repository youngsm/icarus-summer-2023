import pytest
import torch
import numpy as np

from pfmatch.backend import device
from pfmatch.photonlib.photon_library import PhotonLibrary
from tests.fixtures import rng, torch_rng, num_pmt, fake_photon_library

@pytest.fixture
def plib(fake_photon_library):
    return PhotonLibrary(fake_photon_library)

def test_LoadData(plib):
    
    # load data w/o transform
    data = plib.LoadData(transform=False)
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    assert torch.allclose(data, plib._vis), "data should be the same as _vis"
    
    # load data with transform (default)
    data = plib.LoadData(transform=True)
    assert data.shape == plib._vis.shape, "data should be the same shape as _vis"
    inv_data = plib.DataTransformInv(data)
    assert torch.allclose(inv_data, plib._vis), "data transforms not working properly"

def test_DataTransforms(plib):
    data_tranformed = plib.DataTransform(plib._vis)
    data_inv = plib.DataTransformInv(data_tranformed)
    assert torch.allclose(data_inv, plib._vis), "transform + inverse transform != original data"


""" ---------------------- axis (idx, idy, idz) inputs --------------------- """
def test_AxisID2VoxID(plib):
    # [idx, idy, idz] -> [voxid]
    voxaxes = [0, 0, 0]
    tensor_vox = torch.tensor(voxaxes, device=device)
    array_vox = np.array(voxaxes)

    for vox in [voxaxes, tensor_vox, array_vox]:
        voxid = plib.AxisID2VoxID(vox)
        assert isinstance(voxid, torch.Tensor), "voxid should be a torch.Tensor"
        assert torch.allclose(voxid, torch.tensor([0])), "voxid should be [0]"
        
    voxaxes = [[0, 0, 0], [1, 1, 1]]
    tensor_vox = torch.tensor(voxaxes, device=device)
    array_vox = np.array(voxaxes)
    
    for vox in [voxaxes, tensor_vox, array_vox]:
        second_answer = (1 + plib.shape[0] + plib.shape[0]*plib.shape[1]).long()
        voxid = plib.AxisID2VoxID(vox)
        assert isinstance(voxid, torch.Tensor), "voxid should be a torch.Tensor"
        assert torch.allclose(voxid, torch.tensor([0, second_answer])), f"voxid should be [0, {second_answer}]"

def test_AxisID2Position(plib):
    """not tested for accuracy of position, just that it returns a tensor of the right shape"""
    
    # [idx, idy, idz] -> [x, y, z]
    voxaxes = [0, 0, 0]
    tensor_vox = torch.tensor(voxaxes, device=device)
    array_vox = np.array(voxaxes)
    
    for vox in [voxaxes, tensor_vox, array_vox]:
        pos = plib.AxisID2Position(vox)
        assert isinstance(pos, torch.Tensor), "pos should be a torch.Tensor"
        assert pos.shape == (1, 3), "pos should have 3 dimensions" 
        assert all(pos[:,i].min() >= plib._min[i] for i in range(3)), "pos out of min range"
        assert all(pos[:,i].max() <= plib._max[i] for i in range(3)), "pos out of max range"

    voxaxes = [[0, 0, 0], [1, 1, 1]]
    tensor_vox = torch.tensor(voxaxes, device=device)
    array_vox = np.array(voxaxes)
    for vox in [voxaxes, tensor_vox, array_vox]:
        pos = plib.AxisID2Position(vox)
        assert isinstance(pos, torch.Tensor), "pos should be a torch.Tensor"
        assert pos.shape == (len(voxaxes), 3), "pos should have 3 dimensions" 
        assert all(pos[:,i].min() >= plib._min[i] for i in range(3)), "pos out of min range"
        assert all(pos[:,i].max() <= plib._max[i] for i in range(3)), "pos out of max range"

""" ------------------------ position (x,y,z) inputs ----------------------- """
def test_Position2AxisID(plib, torch_rng):
    # [x, y, z] -> [voxid]
    mins = plib._min
    maxs = plib._max
    
    # 1D tensor
    rand_pos = torch.rand(size=(3,), generator=torch_rng)*(maxs-mins)+mins
    axisid = plib.Position2AxisID(rand_pos)
    assert isinstance(axisid, torch.Tensor), "rand_pos should be a torch.Tensor"
    assert axisid.shape == (1, 3), "axisid should have 3 dimensions"

    # 1D list
    axisid = plib.Position2AxisID(list(rand_pos))
    assert isinstance(axisid, torch.Tensor), "rand_pos should be a torch.Tensor"
    assert axisid.shape == (1, 3), "axisid should have 3 dimensions"

    # 2D tensor
    rand_pos = torch.rand(size=(100,3), generator=torch_rng)*(maxs-mins)+mins
    axisid = plib.Position2AxisID(rand_pos)
    assert isinstance(axisid, torch.Tensor), "rand_pos should be a torch.Tensor"
    assert axisid.shape == (100, 3), "axisid should have 3 dimensions"
    
    # check that axisid is in the right range
    axisid = axisid.squeeze()
    assert torch.all(axisid >= 0)
    assert all(torch.all(axisid[:, i] < plib.shape[i]) for i in range(3)), \
            f"axisid should be in the right range ({plib.shape}), {axisid}"

def test_Position2VoxID(plib, torch_rng):
    """not tested for accuracy of voxid just that the returned voxids are in range"""
    # [x, y, z] -> [voxid]
    mins = plib._min
    maxs = plib._max
    
    rand_pos = torch.rand(size=(100, 3), generator=torch_rng)*(maxs-mins)+mins
    voxids = plib.Position2VoxID(rand_pos)    
    
    maxvox = plib.shape[0]*plib.shape[1]*plib.shape[2] - 1
    assert torch.all(voxids>=0), "voxids should be positive"
    assert torch.all(voxids<=maxvox), f"voxids should be less than {maxvox}"

""" ---------------------------- voxel id inputs --------------------------- """
def test_VoxID2AxisID(plib, torch_rng):
    # [voxid] -> [idx, idy, idz]
    maxvox = plib.shape[0]*plib.shape[1]*plib.shape[2] - 1
    voxids = torch.randint(0, int(maxvox), size=(100,), generator=torch_rng)
    
    axisid = plib.VoxID2AxisID(voxids)
    assert all(torch.all(axisid[:, i] < plib.shape[i]) for i in range(3)), \
            f"axisid should be in the right range ({plib.shape}), {axisid}"

def test_VoxID2Coord(plib, torch_rng):
    # [voxid] -> [x, y, z], ...]
    maxvox = plib.shape[0]*plib.shape[1]*plib.shape[2] - 1
    voxids = torch.randint(0, int(maxvox), size=(100,), generator=torch_rng)

    # note: normalized!
    pos = plib.VoxID2Coord(voxids)

    assert isinstance(pos, torch.Tensor), "pos should be a torch.Tensor"
    assert pos.shape == (100, 3), "pos should have 3 dimensions" 
    assert pos.min() >= 0, "pos out of min range"
    assert pos.max() <= 1, "pos out of max range"

""" ------------------------- ring around the rosie ------------------------ """
def test_axis2pos2axis(plib, torch_rng):
    input = (torch.rand(size=(100, 3), generator=torch_rng)*plib.shape).int()
    output = plib.Position2AxisID(plib.AxisID2Position(input))
    assert torch.allclose(input, output), '(ix,iy,iz) -> (x,y,z) -> (ix,iy,iz) failed'

def test_vox2axis2vox(plib, torch_rng):
    maxvox = int(plib.shape[0]*plib.shape[1]*plib.shape[2]) - 1
    input = torch.randint(0, maxvox, size=(100,), generator=torch_rng)
    output = plib.AxisID2VoxID(plib.VoxID2AxisID(input))
    assert torch.allclose(input, output), '(voxid) -> (ix, iy, iz) -> (voxid) failed'

""" ------------------------------ visibility ------------------------------ """
def test_Visibility(plib, torch_rng, num_pmt):
    kwargs = [dict(vids=0, ch=None),
              dict(vids=[0,1,2], ch=None),
              dict(vids=[0,1,2], ch=torch.randint(0, num_pmt, size=(1,), generator=torch_rng)),
              dict(vids=torch.tensor([0,1,2]), ch=None),
              dict(vids=torch.tensor([0,1,2]), ch=torch.randint(0, num_pmt, size=(1,), generator=torch_rng)),
              dict(vids=[0,1,2], ch=[1,2,3])
              ]
    shapes = [(num_pmt,), (3, num_pmt), (3, 1), (3, num_pmt), (3, 1), (3, 3)]
    
    # all good
    for kwarg,shape in zip(kwargs, shapes):
        assert plib.Visibility(**kwarg).shape == shape
    
    # all bad
    with pytest.raises(IndexError):
        plib.Visibility([2013, 30129, 218])
        plib.Visibility([0, 1, 2], num_pmt + 10)
        
def test_VisibilityFromAxisID(plib, num_pmt):
    voxaxes = [[0, 0, 0], [1, 1, 1]]
    tensor_vox = torch.tensor(voxaxes, device=device)
    array_vox = np.array(voxaxes)
    
    for vox in [voxaxes, tensor_vox, array_vox]:
        vis = plib.VisibilityFromAxisID(vox)
        assert isinstance(vis, torch.Tensor), "voxid should be a torch.Tensor"
        assert vis.shape == (len(vox), num_pmt)
        
def test_VisibilityFromXYZ(plib, torch_rng, num_pmt):
    mins = plib._min
    maxs = plib._max
    
    rand_pos = torch.rand(size=(100, 3), generator=torch_rng)*(maxs-mins)+mins
    assert plib.VisibilityFromXYZ(rand_pos).shape == (100, num_pmt)
    
""" --------------------------------- coord -------------------------------- """
def test_CoordFromVoxID(plib, torch_rng):
    maxvox = int(plib.shape[0]*plib.shape[1]*plib.shape[2]) - 1
    input = torch.randint(0, maxvox, size=(100,), generator=torch_rng)

    # ensure normalization
    normed_pos = plib.CoordFromVoxID(input, normalize=True)
    assert normed_pos.min() >= -1 and normed_pos.max() <= 1, "coord out of range [-1,1]"
    unnormed_pos = plib.CoordFromVoxID(input, normalize=False)
    assert unnormed_pos.min() >= 0 and unnormed_pos.max() <= 1, "coord out of range [0, 1]"
    
    # test for scalar input
    assert plib.CoordFromVoxID(0).shape == (3,)
    # test for list input
    assert plib.CoordFromVoxID([1,2,3]).shape == (3,3)
    
def test_LoadCoord(plib):
    
    unnormed_pos = plib.LoadCoord(normalize=False)
    assert unnormed_pos.min() >= 0 and unnormed_pos.max() <= 1, "coord out of range [0, 1]"

    normed_pos = plib.LoadCoord(normalize=True)
    assert normed_pos.min() >= -1 and normed_pos.max() <= 1, "coord out of range [-1,1]"

    maxvox = int(plib.shape[0]*plib.shape[1]*plib.shape[2])
    assert normed_pos.shape == (maxvox, 3)
    
""" --------------------------------- other -------------------------------- """
def test_WeightFromPos(plib, torch_rng, num_pmt):
    mins = plib._min
    maxs = plib._max
    
    # 1D tensor
    rand_pos = torch.rand(size=(3,), generator=torch_rng)*(maxs-mins)+mins
    weight = plib.WeightFromPos(rand_pos)
    assert torch.all(weight>0), "weights must be > 0"
    assert weight.shape == (num_pmt,), "must return weight per pmt"
    
    # 2D tensor
    rand_pos = torch.rand(size=(100, 3), generator=torch_rng)*(maxs-mins)+mins
    weight = plib.WeightFromPos(rand_pos)
    assert torch.all(weight>0), "weights must be > 0"
    assert weight.shape == (num_pmt,), "must return weight per pmt"
