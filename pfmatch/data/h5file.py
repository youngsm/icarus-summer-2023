from typing import List
import h5py as h5
import numpy as np

from pfmatch.flashmatch_types import Flash, QCluster


class H5File(object):
    '''
    Class to interface with the HDF5 dataset for pfmatch. 
    It can store arbitrary number of events.
    Each event contains arbitrary number of QCluster and Flash using a "ragged array".
    '''
    
    def __init__(self,file_name,mode):
        '''
        Constructor
            file_name: string path to the data file to be read/written
            mode: h5py.File constructor "mode" (w=write, a=append, r=read)
        '''
        dt_float = h5.vlen_dtype(np.dtype('float32'))
        dt_int   = h5.vlen_dtype(np.dtype('int32'))
        self._mode = mode
        self._f = h5.File(file_name,self._mode)
        if self._mode in ['w','a']:
            self._wh_point = self._f.create_dataset('point', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_group = self._f.create_dataset('group', shape=(0,), maxshape=(None,), dtype=dt_int  )
            self._wh_flash = self._f.create_dataset('flash', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_point.attrs['note'] = '"point": 3D points with photon count. Reshape to (N,4) for total of N points. See "group" attributes to split into cluster.'
            self._wh_group.attrs['note'] = '"group": An array of integers = number of points per cluster. The sum of the array == N points of "point" data.'
            self._wh_flash.attrs['note'] = '"flash": Flashes (photo-electrons-per-pmt). Reshape to (K,180) for K flashes' 
            
            # Attributes below are not implemented
            self._wh_qtime = self._f.create_dataset('qtime', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_qtime_true = self._f.create_dataset('qtime_true', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime = self._f.create_dataset('ftime', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime_true = self._f.create_dataset('ftime_true', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime_width = self._f.create_dataset('ftime_width', shape=(0,), maxshape=(None,), dtype=dt_float)

    def close(self):
        self._f.close()
        
    def __del__(self):
        try:
            self._f.close()
        except AttributeError:
            pass
        
    def __len__(self):
        return len(self._f['point'])
    
    def __getitem__(self,idx):
        return self.read_one(idx)
    
    def __str__(self):
        msg=f'{len(self)} entries in this file. Raw hdf5 attribute descriptions below.\n'
        for k in self._f.keys():
            try:
                msg += ' '*2 + self._f[k].attrs['note'] + '\n'
            except KeyError:
                pass
        return msg
        
    def read_one(self,idx):
        '''
        Read one event specified by the integer index
        '''
        qcluster_vv,flash_vv = self.read_many([idx])
        if len(qcluster_vv)<1:
            return (None, None)
        return (qcluster_vv[0],flash_vv[0])
            
    def read_many(self,idx_v):
        '''
        Read many event specified by an array of integer indexes
        '''
        flash_vv = []
        qcluster_vv = []
        
        for idx in idx_v:
            if idx >= len(self):
                raise IndexError(f'index {idx} out of range (max={len(self)-1})')     
        event_point_v = [np.array(data).reshape(-1,4) for data in self._f['point'][idx_v]]
        event_group_v = self._f['group'][idx_v]
        event_flash_v = [np.array(data) for data in self._f['flash'][idx_v]]
        
        for i in range(len(idx_v)):
            
            event_point = event_point_v[i]
            event_group = event_group_v[i]
            event_flash = event_flash_v[i]
            
            flash_v = []
            qcluster_v = []
            
            event_flash = event_flash.reshape(-1,180)
            for f in event_flash:
                flash_v.append(Flash())
                flash_v[-1].fill(f)

            start = 0
            for ctr in event_group:
                qcluster_v.append(QCluster())
                qcluster_v[-1].fill(event_point[start:start+ctr])
                start = start + ctr
            
            qcluster_vv.append(qcluster_v)
            flash_vv.append(flash_v)
            
        return (qcluster_vv, flash_vv)
        
    
    def write_one(self, qcluster_v: List[QCluster], flash_v: List[Flash]):
        '''
        Write many event to a file with the provided list of QCluster and Flash
        '''
        self.write_many([qcluster_v],[flash_v])
        
    def write_many(self,qcluster_vv: List[List[QCluster]], flash_vv: List[List[QCluster]]):
        '''
        Write many event to a file with the provided list of QCluster and Flash
        '''
        if self._mode not in ['w','a']:
            raise ValueError('the dataset is not created in the w (write) nor a (append) mode')
        if len(qcluster_vv) != len(flash_vv):
            raise ValueError(f'len(qcluster_vv) ({len(qcluster_vv)}) != len(flash_vv) ({len(flash_vv)}')
        
        # expand the output count by one for the new entry
        data_index = self._wh_point.shape[0]
        data_count = data_index+len(qcluster_vv)
        self._wh_point.resize(data_count,axis=0)
        self._wh_group.resize(data_count,axis=0)
        self._wh_flash.resize(data_count,axis=0)

        for i in range(len(qcluster_vv)):
        
            qcluster_v = qcluster_vv[i]
            flash_v = flash_vv[i]
            
            ntracks = len(qcluster_v)
            nflash  = len(flash_v)

            point_v  = []
            for j in range(ntracks):
                point_v.append(qcluster_v[j].qpt_v.cpu().numpy())
            point_group = np.array([pts.shape[0] for pts in point_v])
            point_v = np.concatenate(point_v)

            photon_v = []
            photon_err_v = []
            for j in range(nflash):
                photon_v.append(flash_v[j].pe_v.cpu().numpy())
                photon_err_v.append(flash_v[j].pe_err_v.cpu().numpy())
            photon_v = np.concatenate(photon_v)
            photon_err_v = np.concatenate(photon_err_v)
            
            self._wh_point[data_index] = point_v.flatten()
            self._wh_group[data_index] = point_group
            self._wh_flash[data_index] = photon_v.flatten()
            
            data_index += 1