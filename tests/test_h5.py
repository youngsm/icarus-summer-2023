from tempfile import NamedTemporaryFile
from typing import List
import numpy as np
import pytest

from pfmatch.data.h5file import H5File
from pfmatch.flashmatch_types import Flash, QCluster

""" -------------------------------- helpers ------------------------------- """

@pytest.fixture
def rng():
    return  np.random.default_rng(123)

@pytest.fixture
def fake_data(rng):
    nevents = int(rng.random()*100)
    out = []
    for _ in range(nevents):
        ntracks = int(rng.random()*10)+1
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

def writable_temp_file():
    return NamedTemporaryFile('w', delete=False).name

def compare_len_sum(write: List[QCluster | Flash], read: List[QCluster | Flash]):
    clstype = write[0].__class__.__name__
    assert len(write) == len(read), f'{clstype} written vs read length mismatch ({len(write)} != {len(read)})'
    attributes = {'QCluster': 'qpt_v', 'Flash': 'pe_v'}
    
    if isinstance(write[0], (QCluster, Flash)):
        sumfunc = lambda x: getattr(x, attributes[clstype]).cpu().numpy().sum()
    else: # numpy array
        sumfunc = lambda x: x.sum()
    
    w_sum  = np.sum([sumfunc(f) for f in write])
    r_sum = np.sum([sumfunc(f) for f in read])

    assert abs(w_sum-r_sum) < max(abs(w_sum/1.e5),abs(r_sum/1.e5)), \
            f'{clstype} value sum mismatch ({w_sum} != {r_sum})'

""" --------------------------------- tests -------------------------------- """

def test_h5_write_read_one(fake_data):
    (qcluster_v,flash_v) = fake_data[0]

    f = H5File(writable_temp_file(),'w')
    f.write_one(qcluster_v,flash_v)
    
    qcluster_vout, flash_vout = f.read_one(0)
    compare_len_sum(qcluster_v,qcluster_vout)
    compare_len_sum(flash_v,flash_vout)

    f.close()

def test_h5_write_mode(fake_data):
    (qcluster_v,flash_v) = fake_data[0]

    tmp_name = writable_temp_file()
    f = H5File(tmp_name,'w')
    f.close()
    f = H5File(tmp_name,'r')
    with pytest.raises(ValueError):
        f.write_one(qcluster_v,flash_v)

    f.close()
    
def test_h5_write_and_read(fake_data):
    temp_file = writable_temp_file()

    f = H5File(temp_file,'w')
    qcluster_vv = []
    pe_vv = []
    for (qcluster_v,flash_v) in fake_data:
        qcluster_vv.append(qcluster_v)
        pe_vv.append(flash_v)
    f.write_many(qcluster_vv,pe_vv)
    f.close()
    
    f = H5File(temp_file,'r')
    qcluster_vv_read, pe_vv_read = f.read_many(range(len(f)))
    for i in range(len(qcluster_vv)):
        qcluster_v = qcluster_vv[i]
        qcluster_v_read = qcluster_vv_read[i]
        pe_v = pe_vv[i]
        pe_v_read = pe_vv_read[i]
        compare_len_sum(qcluster_v,qcluster_v_read)
        compare_len_sum(pe_v,pe_v_read)