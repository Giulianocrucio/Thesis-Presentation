# src/data/data_loaders.py

import os
import torch
from torch_geometric.datasets import ZINC, QM9, TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from omegaconf import DictConfig
from .transformation import L2Transform, SLG2Data
torch.serialization.add_safe_globals([SLG2Data])

def get_optimal_workers():
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        return min(4, os.cpu_count() or 1)

def load_zinc_benchmark(root_dir=None, batch_size=32, use_l2=True, num_workers=None):
    """
    Loads the ZINC dataset. If use_l2 is True, applies L2 Transformations.
    """
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), 'data', 'ZINC_SLG2') if use_l2 else os.path.join(os.getcwd(), 'data', 'ZINC')

    os.makedirs(root_dir, exist_ok=True)
    l2_transform = L2Transform() if use_l2 else None
    if num_workers is None:
        num_workers = get_optimal_workers()
    version_name = "SLG2" if use_l2 else "RAW"
    print(f"Loading ZINC {version_name} from {root_dir} with {num_workers} CPU workers.")
    
    # Because root_dir is unique, PyG will look for processed files here.
    # If you ran preprocessing.py first, this will load instantly.
    train_dataset = ZINC(root=root_dir, subset=True, split='train', pre_transform=l2_transform)
    val_dataset   = ZINC(root=root_dir, subset=True, split='val', pre_transform=l2_transform)
    test_dataset  = ZINC(root=root_dir, subset=True, split='test', pre_transform=l2_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def load_qm9_benchmark(root_dir=None, batch_size=32, seed=42, use_l2=True, 
                       num_train=10000, num_val=1000, num_test=1000):
    """
    Loads a subset of the QM9 dataset. 
    Applies L2 Transformations in-memory only to the sampled subset, exactly once.
    """
    # Always load from the RAW directory to avoid triggering a full-dataset pre_transform
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), 'data', 'QM9')

    os.makedirs(root_dir, exist_ok=True)
    num_workers = get_optimal_workers() # Assuming this is defined elsewhere in your code
    
    print(f"Loading raw QM9 from {root_dir}...")
    
    # Load the raw dataset without pre_transform
    dataset = QM9(root=root_dir)
    
    total_subset_size = num_train + num_val + num_test
    if total_subset_size > len(dataset):
        raise ValueError(f"Requested subset size ({total_subset_size}) exceeds dataset size ({len(dataset)}).")

    print(f"Extracting a subset of {total_subset_size} molecules from QM9")
    if use_l2 == True:
        print(f"Applying L2 transformation to the subset (happens only once)")
        
    # Generate deterministic random indices based on the seed
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:total_subset_size].tolist()

    l2_transform = L2Transform() if use_l2 else None
    
    # Process only the subset in memory
    subset_data = []
    for idx in indices:
        data = dataset[idx]
        if l2_transform is not None:
            data = l2_transform(data)
        subset_data.append(data)

    # Split the transformed in-memory list
    train_data = subset_data[:num_train]
    val_data   = subset_data[num_train : num_train + num_val]
    test_data  = subset_data[num_train + num_val :]
    
    # Create DataLoaders directly from the lists
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def load_qm9_benchmark_complete(root_dir=None, batch_size=32, seed=42, use_l2=True):
    """
    Loads the QM9 dataset. If use_l2 is True, applies L2 Transformations.
    """
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), 'data', 'QM9_SLG2') if use_l2 else os.path.join(os.getcwd(), 'data', 'QM9')

    os.makedirs(root_dir, exist_ok=True)
    l2_transform = L2Transform() if use_l2 else None
    num_workers = get_optimal_workers()
    version_name = "SLG2" if use_l2 else "RAW"
    print(f"Loading QM9 {version_name} from {root_dir} with {num_workers} CPU workers.")
    
    dataset = QM9(root=root_dir, pre_transform=l2_transform)
    
    num_train = 10000
    num_val   = 1000
    num_test  = len(dataset) - num_train - num_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [num_train, num_val, num_test], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def load_mutag_benchmark(root_dir=None, batch_size=32, use_l2=True, num_workers=None, seed=42):
    """
    Loads the MUTAG dataset. If use_l2 is True, applies L2 Transformations.
    Splits into 80% train, 10% val, 10% test.
    """
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), 'data', 'mutag_SLG2') if use_l2 else os.path.join(os.getcwd(), 'data', 'mutag')

    os.makedirs(root_dir, exist_ok=True)
    l2_transform = L2Transform() if use_l2 else None
    if num_workers is None:
        num_workers = get_optimal_workers()
        
    version_name = "SLG2" if use_l2 else "RAW"
    print(f"Loading MUTAG {version_name} from {root_dir} with {num_workers} CPU workers.")
    
    dataset = TUDataset(root=root_dir, name='MUTAG', pre_transform=l2_transform)
    
    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_train - num_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [num_train, num_val, num_test], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def load_NCI1_benchmark(root_dir=None, batch_size=32, use_l2=True, num_workers=None, seed=42):
    """
    Loads the NCI1 dataset. If use_l2 is True, applies L2 Transformations.
    Splits into 80% train, 10% val, 10% test.
    """
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), 'data', 'NCI1_SLG2') if use_l2 else os.path.join(os.getcwd(), 'data', 'NCI1')

    os.makedirs(root_dir, exist_ok=True)
    l2_transform = L2Transform() if use_l2 else None
    if num_workers is None:
        num_workers = get_optimal_workers()
        
    version_name = "SLG2" if use_l2 else "RAW"
    print(f"Loading NCI1 {version_name} from {root_dir} with {num_workers} CPU workers.")
    
    dataset = TUDataset(root=root_dir, name='NCI1', pre_transform=l2_transform)
    
    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_train - num_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [num_train, num_val, num_test], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def load_ENZYMES_benchmark(root_dir=None, batch_size=32, use_l2=True, num_workers=None, seed=42):
    """
    Loads the ENZYMES dataset. If use_l2 is True, applies L2 Transformations.
    Splits into 80% train, 10% val, 10% test.
    """
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), 'data', 'ENZYMES_SLG2') if use_l2 else os.path.join(os.getcwd(), 'data', 'ENZYMES')

    os.makedirs(root_dir, exist_ok=True)
    l2_transform = L2Transform() if use_l2 else None
    if num_workers is None:
        num_workers = get_optimal_workers()
        
    version_name = "SLG2" if use_l2 else "RAW"
    print(f"Loading ENZYMES {version_name} from {root_dir} with {num_workers} CPU workers.")
    
    dataset = TUDataset(root=root_dir, name='ENZYMES', pre_transform=l2_transform)
    
    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_train - num_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [num_train, num_val, num_test], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader