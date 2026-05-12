import sys
import os

# Add the project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import ZINC, TUDataset, QM9
from torch_geometric.utils import to_networkx
import pandas as pd
import matplotlib.pyplot as plt

# Import transformations
from src.data.transformation import L2Transform, SLG2Data

# Alias 'data.transformation' to 'src.data.transformation' for backward compatibility with existing pickles
import src.data.transformation
sys.modules['data.transformation'] = src.data.transformation

# Register the custom class as safe for PyTorch to load from disk (Torch 2.6+)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([SLG2Data])


data_stat = pd.DataFrame()
# store data EL2: 
data_EL2 = pd.DataFrame()

def get_mean_variance(g):
    # get the mean and variance of the degree of the nodes
    # Convert PyG data object to NetworkX graph to use .degree()
    nx_graph = to_networkx(g, to_undirected=True)
    degrees = [dict(nx_graph.degree())[n] for n in nx_graph.nodes]
    return np.mean(degrees), np.var(degrees)

def compute_formula(g):
    # compute the formula for the graph
    # M^2(M^2 + \sigma^2)
    mean, var = get_mean_variance(g)
    return (mean**2) * ((mean**2) + var)

def compute_EL2_stima(g):
    # compute the EL2_stima for the graph
    # M = mean, m = number of edges, sigma^2 = variance
    # m^3 M + \frac{m^3 \sigma^2}{M}
    mean, var = get_mean_variance(g)
    m = g.num_edges
    if mean == 0:
        return 0  # Avoid division by zero if graph has no edges
    return ((m**3) * mean) + ((m**3 * var) / mean)

def elaborate_dataset(name_dataset, dataset, has_l2=False):
    print(f"Processing {name_dataset} dataset...")
    formulas, nnodes_list, is_better_list, EL2_list, n_4_list, real_EL2_list = get_dataset_stats(name_dataset, dataset, has_l2)
    add_dataset_stat(name_dataset, formulas, nnodes_list, is_better_list, EL2_list, n_4_list, real_EL2_list)
    
def get_dataset_stats(name_dataset, dataset, has_l2=False):
    formulas = []
    nnodes_list = []
    is_better_list = []
    EL2_list = []
    n_4_list = []
    real_EL2_list = []
    
    global data_EL2
    dataset_el2_rows = []

    for i, g in enumerate(dataset):
        if i % 1000 == 0:
            print(f"{dataset.__class__.__name__} progress: {100 * i / len(dataset):.1f}%")
        
        # get number of nodes
        Nnodes = g.num_nodes

        # get the formula value
        Formula = compute_formula(g)

        # get mean and variance of the degree of the nodes
        mean, var = get_mean_variance(g)
         
        is_better = False

        if Formula <= Nnodes:
            is_better = True
        
        # number of nodes**4
        n_4 = Nnodes**4
        n_4_list.append(n_4)

        nnodes_list.append(Nnodes)
        formulas.append(Formula)
        is_better_list.append(is_better)
        
        el2_stima = compute_EL2_stima(g)
        EL2_list.append(el2_stima)
        
        if has_l2:
            num_l2_edges = g.l2_edge_index.size(1) // 2 if hasattr(g, 'l2_edge_index') else getattr(g, 'num_l2_edges', 0)
            real_EL2_list.append(num_l2_edges)
            
            # dataset, id_graph, |E_L_2|, EL2_stima, n**4
            dataset_el2_rows.append({
                'dataset': name_dataset,
                'id_graph': i,
                'm': g.num_edges,
                '|E_L_2|': round(num_l2_edges, 3),
                'EL2_stima': round(el2_stima, 3),
                'n**4': round(n_4, 3)
            })
            
    if has_l2:
        new_df = pd.DataFrame(dataset_el2_rows)
        data_EL2 = pd.concat([data_EL2, new_df], ignore_index=True)

    return formulas, nnodes_list, is_better_list, EL2_list, n_4_list, real_EL2_list if has_l2 else None

# add the row of data:stat
def add_dataset_stat(name_dataset, formulas, nnodes_list, is_better_list, EL2_list, n_4_list, real_EL2_list=None):
    global data_stat
    # Create a new dictionary for the single row
    row_dict = {
        'dataset': name_dataset,
        'mean_formula': round(np.mean(formulas), 3),
        'var_formula': round(np.var(formulas), 3),
        'mean_nodes': round(np.mean(nnodes_list), 3),
        'var_nodes': round(np.var(nnodes_list), 3),
        'mean_is_better': round(np.mean(is_better_list), 3),
        'var_is_better': round(np.var(is_better_list), 3),
        'mean_EL2_stima': round(np.mean(EL2_list), 3),
        'var_EL2_stima': round(np.var(EL2_list), 3),
        'mean_n**4': round(np.mean(n_4_list), 3),
        'var_n**4': round(np.var(n_4_list), 3)
    }
    
    # In the case of mutag, nci1 and zinc also add these lines
    if real_EL2_list is not None:
        row_dict['mean_real_EL2'] = round(np.mean(real_EL2_list), 3)
        row_dict['var_real_EL2'] = round(np.var(real_EL2_list), 3)
        
    new_row = pd.DataFrame([row_dict])
    
    # Concatenate it with the existing global DataFrame
    data_stat = pd.concat([data_stat, new_row], ignore_index=True)

def check_zinc():
    # --- ZINC Section ---
    print("Loading ZINC dataset...")
    root_dir = os.path.join(os.getcwd(), 'data', 'ZINC_SLG2')
    zinc_dataset = ZINC(root=root_dir, subset=True, split='train', pre_transform=L2Transform())
    elaborate_dataset("ZINC", zinc_dataset, has_l2=True)
    
def check_nci1():
    # ---  nci1 Section ---
    print("Loading nci1 dataset...")
    root_dir = os.path.join(os.getcwd(), 'data', 'NCI1_SLG2')
    nci1_dataset = TUDataset(root=root_dir, name='NCI1', pre_transform=L2Transform())
    elaborate_dataset("nci1", nci1_dataset, has_l2=True)

def check_qm9():
    # ---  qm9 Section ---
    print("Loading qm9 dataset...")
    root_dir = os.path.join(os.getcwd(), 'data', 'QM9')
    qm9_dataset = QM9(root=root_dir)
    elaborate_dataset("qm9", qm9_dataset, has_l2=False)

def check_enzymes():
    # ---  enzymes Section ---
    print("Loading enzymes dataset...")
    root_dir = os.path.join(os.getcwd(), 'data', 'ENZYMES_SLG2')
    enzymes_dataset = TUDataset(root=root_dir, name='ENZYMES',pre_transform=L2Transform())
    elaborate_dataset("enzymes", enzymes_dataset, has_l2=True)

def check_mutag():
    # ---  mutag Section ---
    print("Loading mutag dataset...")
    root_dir = os.path.join(os.getcwd(), 'data', 'mutag_SLG2')
    mutag_dataset = TUDataset(root=root_dir, name='MUTAG', pre_transform=L2Transform())
    elaborate_dataset("mutag", mutag_dataset, has_l2=True)

def create_plots():
    if data_EL2.empty:
        return
        
    datasets = data_EL2['dataset'].unique()
    for ds in datasets:
        df_ds = data_EL2[data_EL2['dataset'] == ds].copy()
        
        # Order by number of edges of G
        df_ds = df_ds.sort_values(by='m')
        
        plt.figure(figsize=(10, 6))
        
        # Plotting only |E_L_2| and n**4
        plt.plot(df_ds['m'].values, df_ds['|E_L_2|'].values, label='|E_L_2| (Real EL2)', marker='o', linestyle='-', markersize=3, alpha=0.7)
        plt.plot(df_ds['m'].values, df_ds['n**4'].values, label='n**4', marker='s', linestyle='-.', markersize=3, alpha=0.7)
        
        # Updated title as requested
        plt.title(f"{ds} - |E_L2| vs n**4")
        plt.xlabel("Number of edges (m) in G")
        plt.ylabel("Value")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        output_path = os.path.join(os.path.dirname(__file__), f"plot_{ds}.png")
        plt.savefig(output_path)
        plt.close()


if __name__ == '__main__':
    check_qm9()
    check_enzymes()

    check_mutag()
    check_zinc()
    check_nci1()
    
    print(data_stat)
    
    # Save the two csv in the same directory of the file check_formula
    output_dir = os.path.dirname(__file__)
    data_stat.to_csv(os.path.join(output_dir, 'data_stat.csv'), index=False)
    
    # Select only requested columns for data_EL2 and output
    if not data_EL2.empty:
        data_EL2_out = data_EL2[['dataset', 'id_graph', '|E_L_2|', 'EL2_stima', 'n**4']].copy()
        data_EL2_out.to_csv(os.path.join(output_dir, 'data_EL2.csv'), index=False)
    
    create_plots()