import copy

import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from PIL import Image
from robomimic.utils.dataset import SequenceDataset
from torch.utils.data import Dataset

"""
    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
    we should in principle use seq_len, but the paddings of the two are different.
    So that's why we currently use frame_stack instead of seq_len.
"""


def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    *args,
    **kwargs
):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
    )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        # Do NOT store the original sequence_dataset to avoid h5py pickling issues
        # Instead, store only the metadata and parameters needed to recreate it
        self.task_emb = task_emb
        self.n_demos = sequence_dataset.n_demos
        self.total_num_sequences = sequence_dataset.total_num_sequences
        self._dataset_length = len(sequence_dataset)
        
        # Store dataset creation parameters for worker processes
        self.dataset_path = sequence_dataset.hdf5_path
        self.obs_keys = sequence_dataset.obs_keys
        self.dataset_keys = sequence_dataset.dataset_keys
        
        if hasattr(sequence_dataset, 'frame_stack'):
            self.frame_stack = sequence_dataset.frame_stack
        elif hasattr(sequence_dataset, 'n_frame_stack'):
            self.frame_stack = sequence_dataset.n_frame_stack
        else:
            raise AttributeError("SequenceDataset object has no frame_stack or n_frame_stack attribute")
        self.seq_length = sequence_dataset.seq_length
        self.pad_frame_stack = sequence_dataset.pad_frame_stack
        self.pad_seq_length = sequence_dataset.pad_seq_length
        self.get_pad_mask = sequence_dataset.get_pad_mask
        self.goal_mode = sequence_dataset.goal_mode
        self.hdf5_cache_mode = sequence_dataset.hdf5_cache_mode
        self.hdf5_use_swmr = sequence_dataset.hdf5_use_swmr
        self.hdf5_normalize_obs = sequence_dataset.hdf5_normalize_obs
        self.filter_by_attribute = getattr(sequence_dataset, 'filter_by_attribute', None)
        
        # Worker-specific dataset instance (not pickled)
        self._worker_dataset = None
        self._worker_id = None

    def __getstate__(self):
        """Custom pickling to exclude non-picklable objects"""
        state = self.__dict__.copy()
        # Remove the unpicklable worker dataset
        state['_worker_dataset'] = None
        state['_worker_id'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore state"""
        self.__dict__.update(state)
        # Initialize worker-specific attributes
        self._worker_dataset = None
        self._worker_id = None

    def _get_worker_dataset(self):
        """Get or create dataset instance for current worker process"""
        import torch
        current_worker_id = torch.utils.data.get_worker_info()
        current_worker_id = current_worker_id.id if current_worker_id is not None else None
        
        # Create new dataset instance for each worker to avoid h5py pickling issues
        if self._worker_dataset is None or self._worker_id != current_worker_id:            
            # Initialize observation utilities for worker process
            import robomimic.utils.obs_utils as ObsUtils
            obs_modality = {
                'rgb': [key for key in self.obs_keys if 'rgb' in key or 'image' in key],
                'depth': [key for key in self.obs_keys if 'depth' in key],
                'low_dim': [key for key in self.obs_keys if key not in
                           [k for k in self.obs_keys if 'rgb' in k or 'image' in k or 'depth' in k]]
            }
            ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
            
            from robomimic.utils.dataset import SequenceDataset
            self._worker_dataset = SequenceDataset(
                hdf5_path=self.dataset_path,
                obs_keys=self.obs_keys,
                dataset_keys=self.dataset_keys,
                load_next_obs=False,
                frame_stack=self.frame_stack,
                seq_length=self.seq_length,
                pad_frame_stack=self.pad_frame_stack,
                pad_seq_length=self.pad_seq_length,
                get_pad_mask=self.get_pad_mask,
                goal_mode=self.goal_mode,
                hdf5_cache_mode=self.hdf5_cache_mode,
                hdf5_use_swmr=self.hdf5_use_swmr,
                hdf5_normalize_obs=self.hdf5_normalize_obs,
                filter_by_attribute=self.filter_by_attribute,
            )
            self._worker_id = current_worker_id
        return self._worker_dataset

    def __len__(self):
        return self._dataset_length

    def __getitem__(self, idx):
        # Use worker-specific dataset instance
        worker_dataset = self._get_worker_dataset()
        return_dict = worker_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # Store dataset creation parameters for worker processes
        self.dataset_params = []
        for ds in sequence_datasets:
            if hasattr(ds, 'frame_stack'):
                frame_stack_value = ds.frame_stack
            elif hasattr(ds, 'n_frame_stack'):
                frame_stack_value = ds.n_frame_stack
            else:
                raise AttributeError("SequenceDataset object has no frame_stack or n_frame_stack attribute")
            
            params = {
                'dataset_path': ds.hdf5_path,
                'obs_keys': ds.obs_keys,
                'dataset_keys': ds.dataset_keys,
                'frame_stack': frame_stack_value,
                'seq_length': ds.seq_length,
                'pad_frame_stack': ds.pad_frame_stack,
                'pad_seq_length': ds.pad_seq_length,
                'get_pad_mask': ds.get_pad_mask,
                'goal_mode': ds.goal_mode,
                'hdf5_cache_mode': ds.hdf5_cache_mode,
                'hdf5_use_swmr': ds.hdf5_use_swmr,
                'hdf5_normalize_obs': ds.hdf5_normalize_obs,
                'filter_by_attribute': getattr(ds, 'filter_by_attribute', None),
            }
            self.dataset_params.append(params)
        
        # Worker-specific dataset instances (not pickled)
        self._worker_datasets = None
        self._worker_id = None

    def __getstate__(self):
        """Custom pickling to exclude non-picklable objects"""
        state = self.__dict__.copy()
        # Remove the unpicklable worker datasets
        state['_worker_datasets'] = None
        state['_worker_id'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore state"""
        self.__dict__.update(state)
        # Initialize worker-specific attributes
        self._worker_datasets = None
        self._worker_id = None

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def _get_worker_datasets(self):
        """Get or create dataset instances for current worker process"""
        import torch
        current_worker_id = torch.utils.data.get_worker_info()
        current_worker_id = current_worker_id.id if current_worker_id is not None else None
        
        # Create new dataset instances for each worker to avoid h5py pickling issues
        if self._worker_datasets is None or self._worker_id != current_worker_id:
            from robomimic.utils.dataset import SequenceDataset
            self._worker_datasets = []
            for params in self.dataset_params:
                worker_dataset = SequenceDataset(
                    hdf5_path=params['dataset_path'],
                    obs_keys=params['obs_keys'],
                    dataset_keys=params['dataset_keys'],
                    load_next_obs=False,
                    frame_stack=params['frame_stack'],
                    seq_length=params['seq_length'],
                    pad_frame_stack=params['pad_frame_stack'],
                    pad_seq_length=params['pad_seq_length'],
                    get_pad_mask=params['get_pad_mask'],
                    goal_mode=params['goal_mode'],
                    hdf5_cache_mode=params['hdf5_cache_mode'],
                    hdf5_use_swmr=params['hdf5_use_swmr'],
                    hdf5_normalize_obs=params['hdf5_normalize_obs'],
                    filter_by_attribute=params['filter_by_attribute'],
                )
                self._worker_datasets.append(worker_dataset)
            self._worker_id = current_worker_id
        return self._worker_datasets

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        # Use worker-specific dataset instances
        worker_datasets = self._get_worker_datasets()
        return_dict = worker_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict


class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        # Do NOT store the original sequence_dataset to avoid h5py pickling issues
        self.buffer_size = buffer_size
        
        # Store dataset creation parameters for worker processes
        if hasattr(sequence_dataset, 'dataset_path'):
            # If it's already a SequenceVLDataset or GroupedTaskDataset, use its parameters
            self.dataset_path = sequence_dataset.dataset_path
            self.obs_keys = sequence_dataset.obs_keys
            self.dataset_keys = sequence_dataset.dataset_keys
            self.frame_stack = sequence_dataset.frame_stack
            self.seq_length = sequence_dataset.seq_length
            self.pad_frame_stack = sequence_dataset.pad_frame_stack
            self.pad_seq_length = sequence_dataset.pad_seq_length
            self.get_pad_mask = sequence_dataset.get_pad_mask
            self.goal_mode = sequence_dataset.goal_mode
            self.hdf5_cache_mode = sequence_dataset.hdf5_cache_mode
            self.hdf5_use_swmr = sequence_dataset.hdf5_use_swmr
            self.hdf5_normalize_obs = sequence_dataset.hdf5_normalize_obs
            self.filter_by_attribute = sequence_dataset.filter_by_attribute
        else:
            # If it's a raw SequenceDataset, extract parameters directly
            self.dataset_path = sequence_dataset.hdf5_path
            self.obs_keys = sequence_dataset.obs_keys
            self.dataset_keys = sequence_dataset.dataset_keys
            
            if hasattr(sequence_dataset, 'frame_stack'):
                self.frame_stack = sequence_dataset.frame_stack
            elif hasattr(sequence_dataset, 'n_frame_stack'):
                self.frame_stack = sequence_dataset.n_frame_stack
            else:
                raise AttributeError("SequenceDataset object has no frame_stack or n_frame_stack attribute")
                
            self.seq_length = sequence_dataset.seq_length
            self.pad_frame_stack = sequence_dataset.pad_frame_stack
            self.pad_seq_length = sequence_dataset.pad_seq_length
            self.get_pad_mask = sequence_dataset.get_pad_mask
            self.goal_mode = sequence_dataset.goal_mode
            self.hdf5_cache_mode = sequence_dataset.hdf5_cache_mode
            self.hdf5_use_swmr = sequence_dataset.hdf5_use_swmr
            self.hdf5_normalize_obs = sequence_dataset.hdf5_normalize_obs
            self.filter_by_attribute = getattr(sequence_dataset, 'filter_by_attribute', None)
        
        # Worker-specific dataset instance (not pickled)
        self._worker_dataset = None
        self._worker_id = None

    def __getstate__(self):
        """Custom pickling to exclude non-picklable objects"""
        state = self.__dict__.copy()
        # Remove the unpicklable worker dataset
        state['_worker_dataset'] = None
        state['_worker_id'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore state"""
        self.__dict__.update(state)
        # Initialize worker-specific attributes
        self._worker_dataset = None
        self._worker_id = None

    def _get_worker_dataset(self):
        """Get or create dataset instance for current worker process"""
        import torch
        current_worker_id = torch.utils.data.get_worker_info()
        current_worker_id = current_worker_id.id if current_worker_id is not None else None
        
        # Create new dataset instance for each worker to avoid h5py pickling issues
        if self._worker_dataset is None or self._worker_id != current_worker_id:
            from robomimic.utils.dataset import SequenceDataset
            self._worker_dataset = SequenceDataset(
                hdf5_path=self.dataset_path,
                obs_keys=self.obs_keys,
                dataset_keys=self.dataset_keys,
                load_next_obs=False,
                frame_stack=self.frame_stack,
                seq_length=self.seq_length,
                pad_frame_stack=self.pad_frame_stack,
                pad_seq_length=self.pad_seq_length,
                get_pad_mask=self.get_pad_mask,
                goal_mode=self.goal_mode,
                hdf5_cache_mode=self.hdf5_cache_mode,
                hdf5_use_swmr=self.hdf5_use_swmr,
                hdf5_normalize_obs=self.hdf5_normalize_obs,
                filter_by_attribute=self.filter_by_attribute,
            )
            self._worker_id = current_worker_id
        return self._worker_dataset

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        # Use worker-specific dataset instance
        worker_dataset = self._get_worker_dataset()
        return worker_dataset.__getitem__(idx)