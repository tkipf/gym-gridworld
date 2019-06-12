"""Utility functions."""

import h5py
import numpy as np

from torch.utils import data


def save_dict_h5py(data, fname):
    """Save dictionary containing numpy arrays to h5py file."""
    with h5py.File(fname, 'w') as hf:
        for key in data.keys():
            hf.create_dataset(key, data=data[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    data = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][:]
    return data


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


class TrajectoryDataset(data.Dataset):
    """Create dataset of (o_t, a_t) trajectories from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_dict_h5py(hdf5_file)

    def __len__(self):
        return len(self.experience_buffer['actions'])

    def __getitem__(self, idx):
        sample = {
            'obs': to_float(self.experience_buffer['observations'][idx]),
            'action': self.experience_buffer['actions'][idx],
        }
        return sample
