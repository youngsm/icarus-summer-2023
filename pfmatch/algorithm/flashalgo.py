import torch
import yaml

from pfmatch.backend import device
from pfmatch.photonlib import SirenLibrary, PhotonLibrary

#TODO: MODIFY TO BE SIREN-COMPATIBLE

class FlashAlgo():
    def __init__(self, detector_specs: dict, photon_library: PhotonLibrary | SirenLibrary, cfg_file: dict = None):
        self.plib = photon_library #same photon library that was passed in DataGen, will be None unless user changes

        if not self.plib:
          if cfg_file:
            self.slib = SirenLibrary(cfg_file)
          else:
            raise RuntimeError("Either config file or photon library must be specified.")

        self.global_qe = 0.0093
        self.reco_pe_calib = 1
        self.qe_v = []  # CCVCorrection factor array
        self.vol_min = torch.tensor(detector_specs["ActiveVolumeMin"], device=device)
        self.vol_max = torch.tensor(detector_specs["ActiveVolumeMax"], device=device)
        if isinstance(cfg_file, dict):
          self.configure(cfg_file)
        elif isinstance(cfg_file, str):
          self.configure_from_yaml(cfg_file)

    def configure_from_yaml(self, cfg_file: str):
        self.configure(yaml.load(open(cfg_file), Loader=yaml.Loader))
        
    def configure(self, fmatch_config):
        config = fmatch_config['PhotonLibHypothesis']
        self.global_qe = config["GlobalQE"]
        self.reco_pe_calib = config["RecoPECalibFactor"]
        self.qe_v = torch.tensor(config["CCVCorrection"], device=device)
        self.siren_path = config["SirenPath"]
        if not self.siren_path and not self.plib:
          raise RuntimeError("PhotonLibrary path not specified in config file")

    def NormalizePosition(self, pos):
        '''
        Convert position in world coordinate to normalized coordinate      
        '''
        #TODO
        if not self.plib:
           return pos / (self.vol_max - self.vol_min)
        
        return ((self.plib.Position2AxisID(pos) + 0.5) / self.plib.shape - 0.5) * 2

    def fill_estimate(self, track):
        """
        fill flash hypothsis based on given qcluster track
        ---------
        Arguments
          track: qcluster track of 3D position + charge
        -------
        Returns hypothesized number of p.e. to be detected in each PMT
          
        """
        if not torch.is_tensor(track):
          track = torch.tensor(track, device=device)

        #fill estimate
        if not self.plib:
          local_pe_v = torch.sum(self.slib.VisibilityFromXYZ(track[:, :3])*(track[:, 3].unsqueeze(-1)), axis = 0)
        else:
          local_pe_v = torch.sum(self.plib.VisibilityFromXYZ(track[:, :3])*(track[:, 3].unsqueeze(-1)), axis = 0)

        if len(self.qe_v) == 0:
          self.qe_v = torch.ones(local_pe_v.shape, device=device)
        return local_pe_v * self.global_qe * self.reco_pe_calib / self.qe_v

    def backward_gradient(self, track):
        """
        Compue the gradient of the fill_estimate step for given track
        ---------
        Arguments
          track: qcluster track of 3D position + charge
        -------
        Returns
          gradient value of the fill_estimate step for track
        """
        
        if not self.plib:
          #neighboring voxel vis values - track voxel vis values / distance btwn voxel pairs
          # neighbor_track = track[:, :3]
          # neighbor_track[:, 0] += self.slib.voxel_width

          # grad = (self.slib.Visibility(neighbor_track) - self.slib.Visibility(track[:, :3])) / self.slib.voxel_width
          raise NotImplementedError

        else:
          vids = self.plib.Position2VoxID(track[:, :3])
          grad = (self.plib.Visibility(vids+1) - self.plib.Visibility(vids)) / self.plib.gap

        return grad * (track[:, 3].unsqueeze(-1)) * self.global_qe * self.reco_pe_calib / self.qe_v