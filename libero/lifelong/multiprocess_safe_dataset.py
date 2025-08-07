"""
多进程安全的数据集包装器，解决 h5py 对象无法被 pickle 的问题。

核心思路：
1. 在主进程中只保存数据集的配置信息，不持有 h5py 句柄
2. 在每个 worker 进程中独立创建数据集实例
3. 使用延迟初始化确保 h5py 文件在正确的进程中打开
"""

import os
import pickle
from typing import Any, Dict, Optional
from torch.utils.data import Dataset


class MultiprocessSafeDatasetWrapper(Dataset):
    """
    多进程安全的数据集包装器。
    
    该包装器通过延迟初始化解决 h5py 对象无法跨进程传递的问题。
    在主进程中只保存创建数据集所需的参数，在 worker 进程中重新创建数据集实例。
    """
    
    def __init__(self, dataset_config: Dict[str, Any], task_emb: Optional[Any] = None):
        """
        初始化多进程安全的数据集包装器。
        
        Args:
            dataset_config: 创建数据集所需的配置参数
            task_emb: 任务嵌入向量（可选）
        """
        self.dataset_config = dataset_config
        self.task_emb = task_emb
        self._dataset = None
        self._worker_id = None
        
        # 在主进程中创建一个临时数据集来获取长度信息
        self._init_dataset()
        self._length = len(self._dataset)
        
        # 清理主进程中的数据集，避免 h5py 句柄被继承
        self._cleanup_dataset()
    
    def _init_dataset(self):
        """初始化底层数据集实例"""
        if self._dataset is None:
            # 重新创建数据集
            from libero.lifelong.datasets import get_dataset, SequenceVLDataset
            sequence_dataset, _ = get_dataset(**self.dataset_config)
            
            if self.task_emb is not None:
                self._dataset = SequenceVLDataset(sequence_dataset, self.task_emb)
            else:
                self._dataset = sequence_dataset
    
    def _cleanup_dataset(self):
        """清理数据集实例，释放 h5py 句柄"""
        if self._dataset is not None:
            # 如果数据集有 close 方法，调用它
            if hasattr(self._dataset, 'close'):
                self._dataset.close()
            elif hasattr(self._dataset, 'sequence_dataset') and hasattr(self._dataset.sequence_dataset, 'close'):
                self._dataset.sequence_dataset.close()
            
            self._dataset = None
    
    def _ensure_dataset_initialized(self):
        """确保数据集在当前进程中已初始化"""
        import os
        current_worker_id = os.getpid()
        
        # 如果是新的 worker 进程，重新初始化数据集
        if self._worker_id != current_worker_id or self._dataset is None:
            self._worker_id = current_worker_id
            self._init_dataset()
    
    def __len__(self):
        """返回数据集长度"""
        return self._length
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        self._ensure_dataset_initialized()
        sample = self._dataset[idx]
        
        # 如果有 task_emb 但样本中没有，添加它
        if self.task_emb is not None and 'task_emb' not in sample:
            sample['task_emb'] = self.task_emb
            
        return sample
    
    def __getstate__(self):
        """自定义 pickle 序列化，排除不可序列化的对象"""
        state = self.__dict__.copy()
        # 移除不可序列化的数据集实例
        state['_dataset'] = None
        return state
    
    def __setstate__(self, state):
        """自定义 pickle 反序列化"""
        self.__dict__.update(state)
        # 数据集将在需要时延迟初始化


class MultiprocessSafeGroupedTaskDataset(Dataset):
    """
    多进程安全的分组任务数据集包装器。
    """
    
    def __init__(self, dataset_configs: list, task_embs: list):
        """
        初始化多进程安全的分组任务数据集。
        
        Args:
            dataset_configs: 数据集配置列表
            task_embs: 任务嵌入向量列表
        """
        self.dataset_configs = dataset_configs
        self.task_embs = task_embs
        self._datasets = None
        self._grouped_dataset = None
        self._worker_id = None
        
        # 在主进程中创建临时数据集来获取元信息
        self._init_datasets()
        self._length = len(self._grouped_dataset)
        self.task_group_size = self._grouped_dataset.task_group_size
        self.n_demos = self._grouped_dataset.n_demos
        self.total_num_sequences = self._grouped_dataset.total_num_sequences
        
        # 清理主进程中的数据集
        self._cleanup_datasets()
    
    def _init_datasets(self):
        """初始化底层数据集实例"""
        if self._datasets is None:
            from libero.lifelong.datasets import get_dataset, GroupedTaskDataset
            self._datasets = []
            for config in self.dataset_configs:
                sequence_dataset, _ = get_dataset(**config)
                self._datasets.append(sequence_dataset)
            
            self._grouped_dataset = GroupedTaskDataset(self._datasets, self.task_embs)
    
    def _cleanup_datasets(self):
        """清理数据集实例"""
        if self._datasets is not None:
            for dataset in self._datasets:
                if hasattr(dataset, 'close'):
                    dataset.close()
            self._datasets = None
            self._grouped_dataset = None
    
    def _ensure_datasets_initialized(self):
        """确保数据集在当前进程中已初始化"""
        import os
        current_worker_id = os.getpid()
        
        if self._worker_id != current_worker_id or self._datasets is None:
            self._worker_id = current_worker_id
            self._init_datasets()
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        self._ensure_datasets_initialized()
        return self._grouped_dataset[idx]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_datasets'] = None
        state['_grouped_dataset'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)


def create_multiprocess_safe_dataset(dataset_path: str, obs_modality: dict, 
                                    task_emb: Optional[Any] = None, **kwargs) -> MultiprocessSafeDatasetWrapper:
    """
    创建多进程安全的数据集。
    
    Args:
        dataset_path: 数据集文件路径
        obs_modality: 观察模态配置
        task_emb: 任务嵌入向量
        **kwargs: 其他数据集参数
    
    Returns:
        MultiprocessSafeDatasetWrapper: 多进程安全的数据集包装器
    """
    dataset_config = {
        'dataset_path': dataset_path,
        'obs_modality': obs_modality,
        **kwargs
    }
    
    return MultiprocessSafeDatasetWrapper(dataset_config, task_emb)


def create_multiprocess_safe_grouped_dataset(dataset_paths: list, obs_modality: dict,
                                            task_embs: list, **kwargs) -> MultiprocessSafeGroupedTaskDataset:
    """
    创建多进程安全的分组任务数据集。
    
    Args:
        dataset_paths: 数据集文件路径列表
        obs_modality: 观察模态配置
        task_embs: 任务嵌入向量列表
        **kwargs: 其他数据集参数
    
    Returns:
        MultiprocessSafeGroupedTaskDataset: 多进程安全的分组任务数据集
    """
    dataset_configs = []
    for path in dataset_paths:
        config = {
            'dataset_path': path,
            'obs_modality': obs_modality,
            **kwargs
        }
        dataset_configs.append(config)
    
    return MultiprocessSafeGroupedTaskDataset(dataset_configs, task_embs)