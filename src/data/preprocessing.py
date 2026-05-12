# src/data/preprocessing.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from torch_geometric.datasets import TUDataset, ZINC, QM9
from data import L2Transform
import torch_geometric.io.fs as fs
import torch

def custom_torch_save(data, path):
    # Bypass fsspec which might buffer entirely in memory, and use old serialization
    # which sometimes avoids 2GB MemoryError on Windows
    torch.save(data, path, _use_new_zipfile_serialization=False)

fs.torch_save = custom_torch_save

def main():
    print("=== Starting Offline Data Preprocessing ===")
    start_time = time.time()

    # Define folders for both raw datasets and transformed datasets
    data_dir = os.path.join(os.getcwd(), 'data')
    #zinc_path_slg2 = os.path.join(data_dir, 'ZINC_SLG2')
    #qm9_path_slg2 = os.path.join(data_dir, 'QM9_SLG2')
    #zinc_path_raw = os.path.join(data_dir, 'ZINC')
    #qm9_path_raw = os.path.join(data_dir, 'QM9')
    nci_path_slg2 = os.path.join(data_dir, 'NCI1_SLG2')
    nci_path_raw = os.path.join(data_dir, 'NCI1')
    enzymes_path_slg2 = os.path.join(data_dir, 'ENZYMES_SLG2')
    enzymes_path_raw = os.path.join(data_dir, 'ENZYMES')

    #mutag_path_slg2 = os.path.join(data_dir, 'mutag_SLG2')

    #os.makedirs(zinc_path_slg2, exist_ok=True)
    #os.makedirs(qm9_path_slg2, exist_ok=True)
    #os.makedirs(mutag_path_slg2, exist_ok=True)
    os.makedirs(nci_path_slg2, exist_ok=True)
    os.makedirs(nci_path_raw, exist_ok=True)
    os.makedirs(enzymes_path_slg2, exist_ok=True)
    os.makedirs(enzymes_path_raw, exist_ok=True)
    #os.makedirs(zinc_path_raw, exist_ok=True)
    #os.makedirs(qm9_path_raw, exist_ok=True)

    l2_transform = L2Transform()

    #print(f"\nProcessing STANDARD ZINC dataset into: {zinc_path_raw}")
    #ZINC(root=zinc_path_raw, subset=True, split='train')
    #ZINC(root=zinc_path_raw, subset=True, split='val')
    #ZINC(root=zinc_path_raw, subset=True, split='test')
    #print("STANDARD ZINC preprocessing complete!")

    #print(f"\nProcessing SLG2 ZINC dataset into: {zinc_path_slg2}")
    #ZINC(root=zinc_path_slg2, subset=True, split='train', pre_transform=l2_transform)
    #ZINC(root=zinc_path_slg2, subset=True, split='val', pre_transform=l2_transform)
    #ZINC(root=zinc_path_slg2, subset=True, split='test', pre_transform=l2_transform)
    #print("SLG2 ZINC preprocessing complete!")

    #print(f"\nProcessing STANDARD QM9 dataset into: {qm9_path_raw}")
    #QM9(root=qm9_path_raw)
    #print("STANDARD QM9 preprocessing complete!")

    #print(f"\nProcessing SLG2 QM9 dataset into: {qm9_path_slg2}")
    #QM9(root=qm9_path_slg2, pre_transform=l2_transform)
    #print("SLG2 QM9 preprocessing complete!")

    print(f"\nProcessing STANDARD NCI1 dataset into: {nci_path_raw}")
    #TUDataset(root=nci_path_raw, name='NCI1')
    print("STANDARD NCI1 preprocessing complete!")
    
    print(f"\nProcessing SLG2 NCI1 dataset into: {nci_path_slg2}")
    #TUDataset(root=nci_path_slg2, name='NCI1', pre_transform=l2_transform)
    print("SLG2 NCI1 preprocessing complete!")
    
    print(f"\nProcessing STANDARD ENZYMES dataset into: {enzymes_path_raw}")
    TUDataset(root=enzymes_path_raw, name='ENZYMES')
    print("STANDARD ENZYMES preprocessing complete!")
    
    print(f"\nProcessing SLG2 ENZYMES dataset into: {enzymes_path_slg2}")
    TUDataset(root=enzymes_path_slg2, name='ENZYMES', pre_transform=l2_transform)
    print("SLG2 ENZYMES preprocessing complete!")
    
    #print(f"\nProcessing SLG2 MUTAG dataset into: {mutag_path_slg2}")
    #TUDataset(root=mutag_path_slg2, name='MUTAG', pre_transform=l2_transform)
    #print("SLG2 MUTAG preprocessing complete!")

    elapsed_time = (time.time() - start_time) / 60
    print(f"\n=== All Preprocessing Finished in {elapsed_time:.2f} minutes ===")

if __name__ == "__main__":
    main()