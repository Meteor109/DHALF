import warnings
import torch
import scipy.sparse as sp
import numpy as np
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.utils import remove_self_loops
import glob
# Added below the existing imports
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp


warnings.simplefilter("ignore")


def construct_spatial_graph_edges(
    spatial_coords,
    k_neighbors=20,
    graph_mode="baseline_spatial_knn",
    expression_pca=None,
    expression_keep_fraction=0.5,
    min_degree=5,
    precluster_n_clusters=12,
    seed=0,
):
    """Construct an unsupervised spatial graph variant as directed symmetric edges.

    This helper is backward-compatible with the existing dataset loaders and does
    not use annotations, marker genes, or downstream cluster labels.
    """
    from sklearn.cluster import KMeans

    spatial_coords = np.asarray(spatial_coords, dtype=float)
    n_nodes = spatial_coords.shape[0]
    k_neighbors = min(max(1, int(k_neighbors)), n_nodes - 1)
    min_degree = min(max(1, int(min_degree)), k_neighbors)
    distances, indices = NearestNeighbors(
        n_neighbors=k_neighbors + 1, algorithm="ball_tree"
    ).fit(spatial_coords).kneighbors(spatial_coords)
    neighbor_indices = indices[:, 1:]

    if graph_mode == "baseline_spatial_knn":
        retained = [list(row) for row in neighbor_indices]
    elif graph_mode == "mutual_spatial_knn":
        neighbor_sets = [set(row.tolist()) for row in neighbor_indices]
        retained = []
        for node, row in enumerate(neighbor_indices):
            keep = [int(other) for other in row if node in neighbor_sets[int(other)]]
            retained.append(keep)
    elif graph_mode in {
        "expr_pruned_spatial_knn",
        "precluster_pruned_spatial_knn",
    }:
        if expression_pca is None:
            raise ValueError(f"{graph_mode} requires expression_pca")
        expression_pca = np.asarray(expression_pca, dtype=float)
        if expression_pca.shape[0] != n_nodes:
            raise ValueError("expression_pca rows must match spatial coordinates")
        norms = np.linalg.norm(expression_pca, axis=1, keepdims=True)
        normalized = expression_pca / np.maximum(norms, 1e-12)
        similarities = np.sum(
            normalized[:, None, :] * normalized[neighbor_indices], axis=2
        )
        preclusters = None
        if graph_mode == "precluster_pruned_spatial_knn":
            preclusters = KMeans(
                n_clusters=min(int(precluster_n_clusters), n_nodes),
                n_init=20,
                random_state=int(seed),
            ).fit_predict(expression_pca)
        retained = []
        keep_count = max(
            min_degree,
            int(np.ceil(k_neighbors * float(expression_keep_fraction))),
        )
        for node, row in enumerate(neighbor_indices):
            node_similarity = similarities[node]
            if graph_mode == "expr_pruned_spatial_knn":
                selected_positions = np.argsort(-node_similarity)[:keep_count]
            else:
                median_similarity = float(np.median(node_similarity))
                selected_positions = np.where(
                    (preclusters[row] == preclusters[node])
                    | (node_similarity >= median_similarity)
                )[0]
                selected_positions = selected_positions[
                    np.argsort(-node_similarity[selected_positions])
                ]
            keep = [int(row[position]) for position in selected_positions]
            if len(keep) < min_degree:
                for other in row:
                    other = int(other)
                    if other not in keep:
                        keep.append(other)
                    if len(keep) >= min_degree:
                        break
            retained.append(keep)
    else:
        raise ValueError(f"Unsupported graph_mode: {graph_mode}")

    directed = np.asarray(
        [(node, other) for node, row in enumerate(retained) for other in row],
        dtype=np.int64,
    )
    reverse = directed[:, [1, 0]]
    symmetric = np.unique(np.vstack((directed, reverse)), axis=0)
    symmetric = symmetric[symmetric[:, 0] != symmetric[:, 1]]
    return symmetric


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, num_splits: int = 10):

    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True

        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True

        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True

        trains.append(train_mask.unsqueeze(1))
        vals.append(val_mask.unsqueeze(1))
        tests.append(test_mask.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    val_mask_all = torch.cat(vals, 1)
    test_mask_all = torch.cat(tests, 1)

    return train_mask_all, val_mask_all, test_mask_all


def get_structural_encoding(edges, nnodes, str_enc_dim=16):

    row = edges[0, :].numpy()
    col = edges[1, :].numpy()
    data = np.ones_like(row)

    A = sp.csr_matrix((data, (row, col)), shape=(nnodes, nnodes))
    D = (np.array(A.sum(1)).squeeze()) ** -1.0

    Dinv = sp.diags(D)
    RW = A * Dinv
    M = RW

    SE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(str_enc_dim - 1):
        M_power = M_power * M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
    SE = torch.stack(SE, dim=-1)
    return SE


def load_mouse_embryo(path):
    print(f"Loading Mouse Embryo data from {path} (Label: Joint_clusters)...")
    import glob
    import pandas as pd
    import re

    # 1) Load the h5ad file
    # -------------------------------------------
    h5ad_files = glob.glob(os.path.join(path, '*.h5ad'))
    if not h5ad_files: 
        raise FileNotFoundError("No .h5ad file found.")
    
    adata = sc.read_h5ad(h5ad_files[0])
    adata.var_names_make_unique()
    
    # 2) Load the metadata table
    # -------------------------------------------
    meta_path = os.path.join(path, 'meta.tsv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.tsv not found.")

    df_meta = pd.read_csv(meta_path, sep='\t')
    if 'barcode' in df_meta.columns:
        df_meta.set_index('barcode', inplace=True)
    
    # 3) Barcode matching (keeps the mult- prefix fix)
    # -------------------------------------------
    adata.obs.index = adata.obs.index.astype(str)
    df_meta.index = df_meta.index.astype(str)
    
    # Attempt 1: direct match
    common_cells = adata.obs.index.intersection(df_meta.index)
    
    # Attempt 2: strip the 'mult-' prefix
    if len(common_cells) == 0:
        print("  -> Direct match failed. Removing 'mult-' prefix from h5ad...")
        stripped_index = adata.obs.index.str.replace(r'^mult-', '', regex=True)
        common_cells = stripped_index.intersection(df_meta.index)
        
        if len(common_cells) > 0:
            print(f"  -> Prefix removal successful! Matched {len(common_cells)} cells.")
            adata.obs.index = stripped_index
    
    # Attempt 3: strip the '-1' suffix
    if len(common_cells) == 0:
        stripped_index = adata.obs.index.str.replace(r'-1$', '', regex=True)
        common_cells = stripped_index.intersection(df_meta.index)
        if len(common_cells) > 0:
             adata.obs.index = stripped_index

    if len(common_cells) == 0:
        raise ValueError("Error: Barcodes still do not match after fixing prefix!")
        

    adata = adata[common_cells]
    
    # Prefer Joint_clusters when present
    if 'Joint_clusters' not in df_meta.columns:
        print(f"!! Warning: 'Joint_clusters' not found. Available: {df_meta.columns.tolist()}")
        # Otherwise fall back to the first column
        target_col = df_meta.columns[0]
    else:
        target_col = 'Joint_clusters'
        
    print(f"  -> Using Label Column: '{target_col}'")
    adata.obs['label'] = df_meta.loc[common_cells, target_col]
    # ==================================
    
    # 4) Extract spatial coordinates
    # -------------------------------------------
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'x' in adata.obs.columns:
        spatial_coords = adata.obs[['x', 'y']].values
    else:
        if 'x' in df_meta.columns and 'y' in df_meta.columns:
             spatial_coords = df_meta.loc[common_cells, ['x', 'y']].values
        else:
             print("!! Warning: Spatial coordinates not found. Using random coords.")
             spatial_coords = np.random.rand(len(common_cells), 2)
        
    adata.obsm['spatial'] = spatial_coords.astype(float)

    # 5) Standard preprocessing
    # -------------------------------------------
    print("Running preprocessing...")
    if 'count' not in adata.layers: 
        adata.layers['count'] = adata.X.copy() if not sp.issparse(adata.X) else adata.X.toarray()
        
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata)
    
    pca = PCA(n_components=60, random_state=2023)
    X_input = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    features = torch.FloatTensor(pca.fit_transform(X_input))

    # 6) Build the graph and package the outputs
    # -------------------------------------------
    nnodes = features.shape[0]
    nfeats = features.shape[1]
    
    k = 20
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    src, dst = [], []
    for i in range(nnodes):
        for j in range(1, k+1):
            src.append(i); dst.append(indices[i, j])
            src.append(indices[i, j]); dst.append(i)
    edge_index, _ = remove_self_loops(torch.tensor([src, dst], dtype=torch.long))

    le = LabelEncoder()
    labels = torch.LongTensor(le.fit_transform(adata.obs['label'].astype(str)))
    print(f"  -> Valid Classes ({len(le.classes_)}): {le.classes_}")

    idx = np.random.permutation(nnodes)
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)
    train_mask[idx[:int(0.6*nnodes)]] = True
    val_mask[idx[int(0.6*nnodes):int(0.8*nnodes)]] = True
    test_mask[idx[int(0.8*nnodes):]] = True
    
    se_path = '../data/se/mouse_embryo'
    if not os.path.exists(se_path): os.makedirs(se_path)
    se_file = os.path.join(se_path, 'mouse_embryo_16.pt')
    if os.path.exists(se_file): se = torch.load(se_file)
    else: se = get_structural_encoding(edge_index, nnodes); torch.save(se, se_file)

    return features, edge_index, se, train_mask.unsqueeze(1).repeat(1,10), val_mask.unsqueeze(1).repeat(1,10), test_mask.unsqueeze(1).repeat(1,10), labels, nnodes, nfeats


def load_candies_E18(path):
    print(f"Loading E18 data from {path}...")
    # 1) Read the data
    file_path = path if os.path.isfile(path) else os.path.join(path, 'E18_adata_rna.h5ad')
    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique()

    # 2) Set spatial coordinates (reference notebook cell 4)
    # Use array_col / array_row as spatial coordinates
    spatial_coords = np.array(adata.obs[['array_col', 'array_row']])
    adata.obsm['spatial'] = spatial_coords

    # 3) Preprocessing (reference notebook cell 8)
    # Gene filtering
    sc.pp.filter_genes(adata, min_cells=10)
    # Highly variable genes
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    # Library-size normalisation
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Log transform
    sc.pp.log1p(adata)
    # Scaling
    sc.pp.scale(adata)
    
    # PCA (node features)
    sc.tl.pca(adata, n_comps=60)
    features = torch.FloatTensor(adata.obsm['X_pca'].copy())

    # 4) Build the spatial neighbour graph
    # The reference notebook calls construct_neighbor_graph; ST data normally use a spatial kNN graph
    # Here the graph is built on the spatial coordinates
    k_neighbors = 30 
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    
    # Build edge_index (source, target)
    src = []
    dst = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]): # skip self (index 0)
            src.append(i)
            dst.append(indices[i, j])
            # Add both directions so the graph is undirected
            src.append(indices[i, j])
            dst.append(i)
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = remove_self_loops(edge_index)[0]

    # 5) Labels
    # 'Combined_Clusters' provides the ground truth
    if 'Combined_Clusters' in adata.obs:
        # Encode string labels as integer ids
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs['Combined_Clusters'])
        labels = torch.LongTensor(labels)
    else:
        print("Warning: 'Combined_Clusters' not found. Using dummy labels.")
        labels = torch.zeros(adata.n_obs, dtype=torch.long)

    # 6) Basic dimensions
    nnodes = features.shape[0]
    nfeats = features.shape[1]
    
    # 7) Train / validation / test masks
    # Use the repository split helper
    train_mask, val_mask, test_mask = get_split(nnodes)
    
    # 8) Structural encoding
    # The encoding is cached under ../data/se/E18_mouse_brain (created if missing)
    se_path = '../data/se/E18_mouse_brain'
    if not os.path.exists(se_path):
        os.makedirs(se_path)
    
    file_name = se_path + '/{}_{}.pt'.format('E18_mouse_brain', 16)
    if os.path.exists(file_name):
        se = torch.load(file_name)
        if se.shape[0] != nnodes:
            print(f"Cached structural encoding shape {tuple(se.shape)} does not match nnodes={nnodes}; recomputing.")
            se = None
    else:
        se = None
    if se is None:
        print('Computing structural encoding for E18...')
        se = get_structural_encoding(edge_index, nnodes)
        torch.save(se, file_name)
        print('Done. The structural encoding is saved as: {}.'.format(file_name))

    if 'layer_guess' in adata.obs and 'le' in locals():
        # Collect the class names when a LabelEncoder exists
        class_names = le.classes_
    else:
        # No labels, or all labels are -1
        class_names = None

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats


def load_151673(path):
    print(f"Loading 151673 data from {path} ")
    
    # --- 1. Read the data ---
    # Expected layout: ./data/151673/filtered_feature_bc_matrix.h5
    h5_path = os.path.join(path, 'filtered_feature_bc_matrix.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Could not find {h5_path}")
        
    adata = sc.read_10x_h5(h5_path)
    
    # Read spatial positions
    pos_path = os.path.join(path, 'tissue_positions_list.csv')
    positions = pd.read_csv(pos_path, header=None, names=['barcode', 'in_tissue', 'array_row', 'array_col', 'px', 'py'])
    positions.set_index('barcode', inplace=True)
    
    # Join positions and keep tissue spots
    adata.obs = adata.obs.join(positions, how='left')
    adata = adata[adata.obs['in_tissue'] == 1]
    adata.var_names_make_unique()
    
    # Physical coordinates used for graph construction
    adata.obsm['spatial'] = adata.obs[['px', 'py']].values

    # --- 2. Read the labels ---
    meta_path = os.path.join(path, 'metadata.tsv')
    if os.path.exists(meta_path):
        df_meta = pd.read_csv(meta_path, sep='\t', index_col=0)
        # Intersect barcodes
        common_index = adata.obs.index.intersection(df_meta.index)
        adata = adata[common_index]
        df_meta = df_meta.loc[common_index]
        
        if 'layer_guess' in df_meta.columns:
            adata.obs['layer_guess'] = df_meta['layer_guess']
        else:
            adata.obs['layer_guess'] = np.nan
    else:
        print("!! Warning: metadata.tsv not found. Labels will be NaN.")
        adata.obs['layer_guess'] = np.nan

    # --- 3. Preprocessing (SEDR pipeline) ---
    print("Running SEDR preprocessing...")
    # Keep raw counts for HVG selection
    adata.layers['count'] = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()
    
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    
    # PCA
    print("Running PCA (n_components=200)...")
    pca = PCA(n_components=200, random_state=2023)
    # FloatTensor expected by the model
    feat_pca = pca.fit_transform(adata.X if not sp.issparse(adata.X) else adata.X.toarray())
    features = torch.FloatTensor(feat_pca)

    k_neighbors = 40
    print(f"Constructing Spatial Graph (k={k_neighbors})...")
    cell_position = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    
    src = []
    dst = []
    for i in range(cell_position.shape[0]):
        for j in range(1, k_neighbors + 1):
            src.append(i)
            dst.append(indices[i, j])
            # Bidirectional edges
            src.append(indices[i, j])
            dst.append(i)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)

    # --- 5. Labels and masks ---
    # Handle NaN labels
    valid_mask = ~adata.obs['layer_guess'].isna()
    # Variant D addition: DLPFC unlabeled spots should not become an extra class.
    # Keep them as -1 so n_clusters/ARI use only valid manual labels.
    le = LabelEncoder()
    labels_np = np.full(adata.n_obs, -1, dtype=np.int64)
    if valid_mask.sum() > 0:
        labels_np[valid_mask.values] = le.fit_transform(adata.obs.loc[valid_mask, 'layer_guess'].astype(str))
    labels = torch.LongTensor(labels_np)
    
    nnodes = features.shape[0]
    nfeats = features.shape[1]

    # Train / validation / test masks over valid labels only
    # Evaluation only uses spots with ground-truth labels
    valid_indices = np.where(valid_mask)[0]
    np.random.shuffle(valid_indices)
    
    n_valid = len(valid_indices)
    train_end = int(0.6 * n_valid)
    val_end = int(0.8 * n_valid)
    
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)
    
    train_mask[valid_indices[:train_end]] = True
    val_mask[valid_indices[train_end:val_end]] = True
    test_mask[valid_indices[val_end:]] = True
    
    # Broadcast to the expected mask shape [n_nodes, 10]
    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    # --- 6. Structural encoding ---
    # Cache directory, e.g. ../data/se/151673
    se_path = '../data/se/151673'
    if not os.path.exists(se_path):
        os.makedirs(se_path)
    
    file_name = se_path + '/{}_{}.pt'.format('151673', 16) # structural encoding dimension 16
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding...')
        se = get_structural_encoding(edge_index, nnodes)
        torch.save(se, file_name)
        print('Done. Saved SE.')

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, nnodes, nfeats



def load_151674(path):
    print(f"Loading 151674 data from {path} ")

    # --- 1. Read the data ---
    h5_path = os.path.join(path, 'filtered_feature_bc_matrix.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Could not find {h5_path}")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    pos_path = os.path.join(path, "spatial", "tissue_positions_list.txt")
    if not os.path.exists(pos_path):
        pos_path = os.path.join(path, "tissue_positions_list.txt")
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"tissue_positions_list not found under {path}")

    positions = pd.read_csv(pos_path, sep=None, engine='python', header=None)
    positions = positions.iloc[:, :6].copy()

    # Drop a header row when present
    if str(positions.iloc[0, 0]).lower() == 'barcode':
        positions = positions.iloc[1:].copy()

    # py=row, px=col so that obsm['spatial'] stores (x, y)
    positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'py', 'px']

    positions['barcode'] = positions['barcode'].astype(str)
    positions['barcode'] = positions['barcode'].apply(lambda b: b if b.endswith('-1') else b + '-1')
    positions.set_index('barcode', inplace=True)

    # --- Barcode alignment (with or without the -1 suffix) ---
    adata.obs.index = adata.obs.index.astype(str)
    if len(adata.obs.index.intersection(positions.index)) == 0:
        # positions lack -1 but adata has it
        if positions.index.str.endswith('-1').sum() == 0 and adata.obs.index.str.endswith('-1').sum() > 0:
            positions.index = positions.index.astype(str) + '-1'
        # positions have -1 but adata lacks it
        elif positions.index.str.endswith('-1').sum() > 0 and adata.obs.index.str.endswith('-1').sum() == 0:
            positions.index = positions.index.astype(str).str.replace(r'-1$', '', regex=True)

    # Join positions and keep tissue spots
    adata.obs = adata.obs.join(positions, how='left')
    if adata.obs['in_tissue'].isna().all():
        raise ValueError("Failed to align tissue_positions_list.txt with 10x barcodes (in_tissue all NaN).")
    adata = adata[adata.obs['in_tissue'] == 1].copy()

    # Physical coordinates used for graph construction
    adata.obsm['spatial'] = adata.obs[['px', 'py']].astype(float).values

    # --- 3. Read the labels ---
    # 151674 labels come from cluster_labels_151674.csv (column ground_truth)
    label_path = os.path.join(path, 'cluster_labels_151674.csv')
    meta_path = os.path.join(path, 'metadata.tsv')

    if os.path.exists(label_path):
        df_lab = pd.read_csv(label_path)
        if 'key' not in df_lab.columns:
            raise ValueError("cluster_labels_151674.csv must contain column 'key'.")
        if 'ground_truth' not in df_lab.columns:
            raise ValueError("cluster_labels_151674.csv must contain column 'ground_truth'.")

        # key looks like 151674_<barcode>; strip the prefix to recover the barcode
        df_lab['barcode'] = df_lab['key'].astype(str).str.replace(r'^151674_', '', regex=True)
        df_lab.set_index('barcode', inplace=True)

        # Align with adata through the barcode intersection
        common_index = adata.obs.index.intersection(df_lab.index)
        if len(common_index) == 0:
            # Retry after normalising the -1 suffix
            if adata.obs.index.str.endswith('-1').sum() > 0 and df_lab.index.str.endswith('-1').sum() == 0:
                df_lab.index = df_lab.index.astype(str) + '-1'
            elif adata.obs.index.str.endswith('-1').sum() == 0 and df_lab.index.str.endswith('-1').sum() > 0:
                df_lab.index = df_lab.index.astype(str).str.replace(r'-1$', '', regex=True)
            common_index = adata.obs.index.intersection(df_lab.index)

        if len(common_index) == 0:
            raise ValueError("Failed to align cluster_labels_151674.csv with 10x barcodes.")

        adata = adata[common_index].copy()
        adata.obs['layer_guess'] = df_lab.loc[common_index, 'ground_truth'].values

    elif os.path.exists(meta_path):
        # Fallback: use metadata.tsv when the cluster file is absent
        df_meta = pd.read_csv(meta_path, sep='\t', index_col=0)
        df_meta.index = df_meta.index.astype(str)
        common_index = adata.obs.index.intersection(df_meta.index)
        adata = adata[common_index].copy()
        df_meta = df_meta.loc[common_index]
        if 'layer_guess' in df_meta.columns:
            adata.obs['layer_guess'] = df_meta['layer_guess'].values
        else:
            adata.obs['layer_guess'] = np.nan
    else:
        print("!! Warning: cluster_labels_151674.csv / metadata.tsv not found. Labels will be NaN.")
        adata.obs['layer_guess'] = np.nan

    # --- 4. Preprocessing (SEDR pipeline) ---
    print("Running SEDR preprocessing...")
    adata.layers['count'] = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()

    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True].copy()
    sc.pp.scale(adata)

    print("Running PCA (n_components=200)...")
    pca = PCA(n_components=200, random_state=2023)
    feat_pca = pca.fit_transform(adata.X if not sp.issparse(adata.X) else adata.X.toarray())
    features = torch.FloatTensor(feat_pca)

    # --- 5. Spatial kNN graph ---
    k_neighbors = 40
    print(f"Constructing Spatial Graph (k={k_neighbors})...")
    cell_position = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)

    src = []
    dst = []
    for i in range(cell_position.shape[0]):
        for j in range(1, k_neighbors + 1):
            src.append(i)
            dst.append(indices[i, j])
            src.append(indices[i, j])
            dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    # --- 6. Labels and masks ---
    valid_mask = ~adata.obs['layer_guess'].isna()
    # Variant D addition: DLPFC unlabeled spots should not become an extra class.
    # Keep them as -1 so n_clusters/ARI use only valid manual labels.
    le = LabelEncoder()
    labels_np = np.full(adata.n_obs, -1, dtype=np.int64)
    if valid_mask.sum() > 0:
        labels_np[valid_mask.values] = le.fit_transform(adata.obs.loc[valid_mask, 'layer_guess'].astype(str))
    labels = torch.LongTensor(labels_np)

    nnodes = features.shape[0]
    nfeats = features.shape[1]

    valid_indices = np.where(valid_mask)[0]
    np.random.shuffle(valid_indices)

    n_valid = len(valid_indices)
    train_end = int(0.6 * n_valid)
    val_end = int(0.8 * n_valid)

    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)

    train_mask[valid_indices[:train_end]] = True
    val_mask[valid_indices[train_end:val_end]] = True
    test_mask[valid_indices[val_end:]] = True

    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    # --- 7. Structural encoding ---
    se_path = '../data/se/151674'
    if not os.path.exists(se_path):
        os.makedirs(se_path)

    file_name = se_path + '/{}_{}.pt'.format('151674', 16)
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding...')
        se = get_structural_encoding(edge_index, nnodes)
        torch.save(se, file_name)
        print('Done. Saved SE.')

    if 'layer_guess' in adata.obs and 'le' in locals():
        # Collect the class names when a LabelEncoder exists
        class_names = le.classes_
    else:
        # No labels, or all labels are -1
        class_names = None

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats


def load_BRCA_visium(path, k_neighbors=40):
    """Load the annotated Breast_Caner Visium dataset used for BRCA evaluation."""
    print(f"Loading BRCA Visium data from {path}")
    h5_path = os.path.join(path, "filtered_feature_bc_matrix.h5")
    pos_path = os.path.join(path, "spatial", "tissue_positions_list.csv")
    ann_path = os.path.join(path, "spatial", "brca_ground_truth_annotation.csv")
    for required_path in (h5_path, pos_path, ann_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required BRCA Visium file not found: {required_path}")

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    positions = pd.read_csv(
        pos_path,
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "py", "px"],
    ).set_index("barcode")
    adata.obs = adata.obs.join(positions, how="left")
    if adata.obs["in_tissue"].isna().any():
        raise ValueError("Failed to align BRCA Visium positions with expression barcodes.")
    adata = adata[adata.obs["in_tissue"].astype(int) == 1].copy()
    adata.obsm["spatial"] = adata.obs[["px", "py"]].astype(float).to_numpy()

    annotations = pd.read_csv(ann_path).set_index("spot_id")
    common = adata.obs_names.intersection(annotations.index)
    if len(common) != adata.n_obs:
        raise ValueError(
            f"BRCA annotation alignment incomplete: {len(common)} of {adata.n_obs} tissue spots matched."
        )
    adata = adata[common].copy()
    adata.obs["ground_truth"] = annotations.loc[common, "ground_truth_annotation"].astype(str).to_numpy()

    print("Running SEDR preprocessing...")
    adata.layers["count"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count", n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata)

    print("Running PCA (n_components=200)...")
    x = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    features = torch.FloatTensor(PCA(n_components=200, random_state=2023).fit_transform(x))

    k_neighbors = min(max(1, int(k_neighbors)), adata.n_obs - 1)
    print(f"Constructing Spatial Graph (k={k_neighbors})...")
    coords = adata.obsm["spatial"]
    _, indices = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="ball_tree").fit(coords).kneighbors(coords)
    src, dst = [], []
    for node, neighbors in enumerate(indices[:, 1:]):
        for neighbor in neighbors:
            src.extend((node, int(neighbor)))
            dst.extend((int(neighbor), node))
    edge_index, _ = remove_self_loops(torch.tensor([src, dst], dtype=torch.long))

    encoder = LabelEncoder()
    labels = torch.LongTensor(encoder.fit_transform(adata.obs["ground_truth"].astype(str)))
    nnodes, nfeats = features.shape
    indices = np.random.permutation(nnodes)
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)
    train_mask[indices[:int(0.6 * nnodes)]] = True
    val_mask[indices[int(0.6 * nnodes):int(0.8 * nnodes)]] = True
    test_mask[indices[int(0.8 * nnodes):]] = True
    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    se_dir = "../data/se/BRCA"
    os.makedirs(se_dir, exist_ok=True)
    se_path = os.path.join(se_dir, f"BRCA_4992_k{k_neighbors}_16.pt")
    se = torch.load(se_path) if os.path.exists(se_path) else None
    if se is None or se.shape[0] != nnodes:
        print("Computing structural encoding for BRCA Visium...")
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=16)
        torch.save(se, se_path)

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, encoder.classes_, nnodes, nfeats


def _cns_subset_indices(adata, max_nodes=8000, seed=0):
    n_select = min(int(max_nodes), adata.n_obs)
    if n_select == adata.n_obs:
        return np.arange(adata.n_obs, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(adata.n_obs, size=n_select, replace=False))


def load_CNS(path, k_neighbors=10, max_nodes=8000, seed=0):
    label_col = "Main_molecular_tissue_region"
    print(f"Loading CNS data from {path} (max_nodes={max_nodes})...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CNS h5ad file not found: {path}")
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    if "spatial" not in adata.obsm:
        raise ValueError("CNS requires adata.obsm['spatial'] coordinates.")
    if label_col not in adata.obs:
        raise ValueError(f"CNS annotation column '{label_col}' is unavailable.")
    adata = adata[_cns_subset_indices(adata, max_nodes=max_nodes, seed=seed)].copy()
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)
    adata.obs["ground_truth"] = adata.obs[label_col].astype(str)

    print("Running CNS preprocessing...")
    adata.layers["count"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count", n_top_genes=min(1000, adata.n_vars))
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata)

    n_components = min(100, adata.n_obs - 1, adata.n_vars - 1)
    print(f"Running PCA (n_components={n_components})...")
    x = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    features = torch.FloatTensor(PCA(n_components=n_components, random_state=2023).fit_transform(x))

    k_neighbors = min(max(1, int(k_neighbors)), adata.n_obs - 1)
    print(f"Constructing CNS spatial graph (k={k_neighbors})...")
    _, indices = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="ball_tree").fit(adata.obsm["spatial"]).kneighbors(adata.obsm["spatial"])
    src, dst = [], []
    for node, neighbors in enumerate(indices[:, 1:]):
        for neighbor in neighbors:
            src.extend((node, int(neighbor)))
            dst.extend((int(neighbor), node))
    edge_index, _ = remove_self_loops(torch.tensor([src, dst], dtype=torch.long))

    encoder = LabelEncoder()
    labels = torch.LongTensor(encoder.fit_transform(adata.obs["ground_truth"].astype(str)))
    nnodes, nfeats = features.shape
    split = np.random.default_rng(seed).permutation(nnodes)
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)
    train_mask[split[:int(0.6 * nnodes)]] = True
    val_mask[split[int(0.6 * nnodes):int(0.8 * nnodes)]] = True
    test_mask[split[int(0.8 * nnodes):]] = True
    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    se_dir = "../data/se/CNS"
    os.makedirs(se_dir, exist_ok=True)
    se_file = os.path.join(se_dir, f"CNS3_subset{nnodes}_k{k_neighbors}_16.pt")
    se = torch.load(se_file) if os.path.exists(se_file) else None
    if se is None or se.shape[0] != nnodes:
        print("Computing structural encoding for CNS...")
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=16)
        torch.save(se, se_file)
    return features, edge_index, se, train_mask, val_mask, test_mask, labels, encoder.classes_, nnodes, nfeats


def load_Stereo_AB(path=None, annotation_col='Annotation', k_neighbors=20, pca_dim=80):

    if path is None:
        path = './data/Stereo/Stereo_AB.h5ad'

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stereo_AB.h5ad not found at: {path}\n"
            "Please place it at ./data/Stereo/Stereo_AB.h5ad (default) "
            "or pass a valid path to load_Stereo_AB(path=...)."
        )

    print(f"Loading Stereo_AB data from {path} (Label: {annotation_col}) ...")

    # 2) load h5ad
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    # 3) spatial coords
    if 'spatial' not in adata.obsm:
        # fallback: try obs columns
        if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['x', 'y']].values
        elif 'spatial_x' in adata.obs.columns and 'spatial_y' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['spatial_x', 'spatial_y']].values
        elif 'array_col' in adata.obs.columns and 'array_row' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['array_col', 'array_row']].values
        else:
            raise ValueError("Could not find spatial coordinates in adata.obsm['spatial'] or common obs columns.")

    spatial_coords = np.asarray(adata.obsm['spatial'], dtype=float)

    # 4) ground truth
    if annotation_col not in adata.obs.columns:
        raise ValueError(f"Annotation column '{annotation_col}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    adata.obs['ground_truth'] = adata.obs[annotation_col].astype(str)

    # 5) preprocessing (lightweight, close to your reference)
    # Use raw counts if provided as a layer; otherwise fall back to X.
    if 'count' not in adata.layers:
        if 'counts' in adata.layers:
            adata.layers['count'] = adata.layers['counts']
        else:
            adata.layers['count'] = adata.X.copy() if not sp.issparse(adata.X) else adata.X

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_genes(adata, min_counts=10)

    # HVG on counts (Seurat v3 expects counts-like)
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=3000)
        adata = adata[:, adata.var['highly_variable']].copy()
    except Exception as e:
        print(f"!! Warning: HVG selection failed ({e}). Proceeding without HVG filtering.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # PCA as node features
    sc.tl.pca(adata, n_comps=pca_dim, svd_solver='arpack')
    features = torch.FloatTensor(adata.obsm['X_pca'].copy())

    # 6) construct kNN graph on spatial coords (n=4 in your reference)
    nnodes = features.shape[0]
    nfeats = features.shape[1]

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    src, dst = [], []
    for i in range(nnodes):
        for j in range(1, k_neighbors + 1):
            src.append(i); dst.append(indices[i, j])
            src.append(indices[i, j]); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    # 7) labels + masks (60/20/20, repeated to 10 splits shape)
    # NOTE (minimal-reproduction fix): the original code guarded this block with an
    # `if section_filter is not None:` check inherited from load_MOB(), but
    # section_filter is not a parameter of this loader, so the branch always raised
    # NameError. Stereo_AB always provides the `Annotation` ground truth (validated
    # above), so the supervised label path is used unconditionally.
    le = LabelEncoder()
    labels = torch.LongTensor(le.fit_transform(adata.obs['ground_truth'].astype(str)))
    class_names = le.classes_

    idx = np.random.permutation(nnodes)
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)

    train_mask[idx[:int(0.6 * nnodes)]] = True
    val_mask[idx[int(0.6 * nnodes):int(0.8 * nnodes)]] = True
    test_mask[idx[int(0.8 * nnodes):]] = True

    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    # 8) structural encoding (cache)
    se_path = '../data/se/Stereo_AB'
    if not os.path.exists(se_path):
        os.makedirs(se_path)

    se_file = os.path.join(se_path, 'Stereo_AB_16.pt')
    if os.path.exists(se_file):
        se = torch.load(se_file)
        if se.shape[0] != nnodes:
            print(f"Cached structural encoding shape {tuple(se.shape)} does not match nnodes={nnodes}; recomputing.")
            se = None
    else:
        se = None
    if se is None:
        print("Computing structural encoding...")
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=16)
        torch.save(se, se_file)

    if 'layer_guess' in adata.obs and 'le' in locals():
        # Collect the class names when a LabelEncoder exists
        class_names = le.classes_
    else:
        # No labels, or all labels are -1
        class_names = None

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats


def _mob_subset_indices(adata, max_per_section=1000, seed=0):
    if 'section' not in adata.obs.columns:
        n = min(int(max_per_section) * 8, adata.n_obs)
        return np.sort(np.random.default_rng(seed).choice(adata.n_obs, size=n, replace=False))

    rng = np.random.default_rng(seed)
    sections = adata.obs['section'].astype(str).to_numpy()
    keep = []
    for section in sorted(np.unique(sections), key=lambda x: (str(x))):
        idx = np.where(sections == section)[0]
        take = min(int(max_per_section), len(idx))
        keep.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def load_MOB(path=None, k_neighbors=20, pca_dim=80, max_per_section=1000, section_filter=None):
    if path is None:
        path = './data/MOB/MOB_3D.h5ad'
    if not os.path.exists(path):
        raise FileNotFoundError(f"MOB_3D.h5ad not found at: {path}")

    section_msg = f"section={section_filter}" if section_filter is not None else f"stratified subset max_per_section={max_per_section}"
    print(f"Loading MOB data from {path} (Label: section; {section_msg}) ...")
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    if section_filter is not None:
        if 'section' not in adata.obs.columns:
            raise ValueError("MOB section_filter requires obs['section'].")
        section_mask = adata.obs['section'].astype(str).to_numpy() == str(section_filter)
        if section_mask.sum() == 0:
            raise ValueError(f"No MOB spots found for section={section_filter}.")
        adata = adata[section_mask].copy()
    else:
        subset_idx = _mob_subset_indices(adata, max_per_section=max_per_section, seed=0)
        adata = adata[subset_idx].copy()

    if 'spatial' not in adata.obsm:
        if {'xcoord', 'ycoord'}.issubset(adata.obs.columns):
            adata.obsm['spatial'] = adata.obs[['xcoord', 'ycoord']].astype(float).to_numpy()
        elif 'coord' in adata.obsm:
            adata.obsm['spatial'] = np.asarray(adata.obsm['coord'])[:, :2].astype(float)
        else:
            raise ValueError("MOB requires adata.obsm['spatial'], adata.obsm['coord'], or obs xcoord/ycoord.")
    spatial_coords = np.asarray(adata.obsm['spatial'], dtype=float)

    if 'section' not in adata.obs.columns:
        print("!! Warning: MOB obs['section'] not found. Labels will be all Unknown.")
        adata.obs['section'] = 'Unknown'
    adata.obs['ground_truth'] = adata.obs['section'].astype(str)

    if 'count' not in adata.layers:
        adata.layers['count'] = adata.X.copy() if not sp.issparse(adata.X) else adata.X

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_genes(adata, min_counts=10)
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=3000)
        adata = adata[:, adata.var['highly_variable']].copy()
    except Exception as e:
        print(f"!! Warning: MOB HVG selection failed ({e}). Proceeding without HVG filtering.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=pca_dim, svd_solver='arpack')
    features = torch.FloatTensor(adata.obsm['X_pca'].copy())

    nnodes = features.shape[0]
    nfeats = features.shape[1]
    k_eff = min(int(k_neighbors), nnodes - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='ball_tree').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    src, dst = [], []
    for i in range(nnodes):
        for j in range(1, k_eff + 1):
            src.append(i); dst.append(indices[i, j])
            src.append(indices[i, j]); dst.append(i)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    if section_filter is not None:
        labels = torch.full((nnodes,), -1, dtype=torch.long)
        class_names = None
    else:
        le = LabelEncoder()
        labels = torch.LongTensor(le.fit_transform(adata.obs['ground_truth'].astype(str)))
        class_names = le.classes_

    idx = np.random.permutation(nnodes)
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)
    train_mask[idx[:int(0.6 * nnodes)]] = True
    val_mask[idx[int(0.6 * nnodes):int(0.8 * nnodes)]] = True
    test_mask[idx[int(0.8 * nnodes):]] = True
    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    se_path = '../data/se/MOB'
    os.makedirs(se_path, exist_ok=True)
    se_file = os.path.join(se_path, f'MOB_subset{nnodes}_16.pt')
    if os.path.exists(se_file):
        se = torch.load(se_file)
        if se.shape[0] != nnodes:
            print(f"Cached MOB structural encoding shape {tuple(se.shape)} does not match nnodes={nnodes}; recomputing.")
            se = None
    else:
        se = None
    if se is None:
        print("Computing structural encoding for MOB subset...")
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=16)
        torch.save(se, se_file)

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats


def load_Stereo_mouse_brain(path=None, annotation_col=None, k_neighbors=40, pca_dim=200):

    if path is None:
        path = './data/Stereo/Stereo_mouse_brain.h5ad'

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stereo_mouse_brain.h5ad not found at: {path}\n"
            "Please place it at ./data/Stereo/Stereo_mouse_brain.h5ad (default) "
            "or pass a valid path to load_Stereo_AB(path=...)."
        )

    print(f"Loading Stereo_AB data from {path} (Label: {annotation_col}) ...")

    # 2) load h5ad
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    # 3) ensure spatial coordinates
    if 'spatial' not in adata.obsm:
        if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['x', 'y']].values
        elif 'x_array' in adata.obs.columns and 'y_array' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['x_array', 'y_array']].values
        else:
            raise ValueError("No spatial coordinates found in adata.obsm['spatial'] or obs[['x','y']] / obs[['x_array','y_array']].")
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)

    # 4) choose label column
    if annotation_col is None:
        for c in ['ground_truth', 'Annotation', 'cell_type', 'clusters', 'leiden']:
            if c in adata.obs.columns:
                annotation_col = c
                break

    if annotation_col is None or annotation_col not in adata.obs.columns:
        print("!! Warning: label column not found. Labels will be NaN (evaluation masks will be empty).")
        adata.obs['layer_guess'] = np.nan
    else:
        adata.obs['layer_guess'] = adata.obs[annotation_col]
        print(f"  -> Using labels from obs['{annotation_col}'] (n_classes={adata.obs['layer_guess'].nunique()})")

    # 5) preprocessing 
    print("Running SEDR preprocessing...")
    # Prefer raw counts layer if present
    if 'raw_count' in adata.layers:
        adata.X = adata.layers['raw_count'].copy()
    elif 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()

    adata.layers['count'] = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()

    # Stereo-seq sections have fewer spots, so a threshold of 10 is safer
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True].copy()
    sc.pp.scale(adata)

    print(f"Running PCA (n_components={pca_dim})...")
    pca = PCA(n_components=pca_dim, random_state=2023)
    X_mat = adata.X if not sp.issparse(adata.X) else adata.X.toarray()
    feat_pca = pca.fit_transform(X_mat)
    features = torch.FloatTensor(feat_pca)

    # 6) build spatial kNN graph
    print(f"Constructing spatial kNN graph (k={k_neighbors})...")
    cell_position = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)

    src, dst = [], []
    for i in range(cell_position.shape[0]):
        for j in range(1, k_neighbors + 1):
            src.append(i)
            dst.append(indices[i, j])
            src.append(indices[i, j])
            dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    # 7) labels + masks
    valid_mask = ~adata.obs['layer_guess'].isna()
    adata.obs['layer_guess'] = adata.obs['layer_guess'].astype('string').fillna('Unknown')


    le = LabelEncoder()
    labels_np = le.fit_transform(adata.obs['layer_guess'].astype(str))
    labels = torch.LongTensor(labels_np)

    nnodes = features.shape[0]
    nfeats = features.shape[1]

    valid_indices = np.where(valid_mask.values)[0]
    np.random.shuffle(valid_indices)

    n_valid = len(valid_indices)
    train_end = int(0.6 * n_valid)
    val_end = int(0.8 * n_valid)

    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)

    if n_valid > 0:
        train_mask[valid_indices[:train_end]] = True
        val_mask[valid_indices[train_end:val_end]] = True
        test_mask[valid_indices[val_end:]] = True

    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    # 8) structural encoding (SE)
    se_path = '../data/se/Stereo_mouse_brain'
    if not os.path.exists(se_path):
        os.makedirs(se_path)

    file_name = os.path.join(se_path, '{}_{}.pt'.format('Stereo_mouse_brain', 16))
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding...')
        se = get_structural_encoding(edge_index, nnodes)
        torch.save(se, file_name)
        print('Done. Saved SE.')

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, nnodes, nfeats


def load_Stereo_DPI(path=None, annotation_col=None, k_neighbors=20, pca_dim=60,
                    min_cells=10, min_counts=10, n_top_genes=2000, str_enc_dim=16):

    # ---- 1) resolve file path ----
    if path is None:
        path = './data/DPI/2DPI_1.h5ad'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stereo_DPI.h5ad not found at: {path}\n"
            "Please place it at ./data/Stereo/Stereo_DPI.h5ad (default) "
            "or pass a valid path to load_Stereo_AB(path=...)."
        )

    print(f"Loading Stereo_DPI data from {path} (Label: {annotation_col}) ...")

    # ---- 2) load AnnData ----
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    # ---- 3) spatial coordinates ----
    if 'spatial' in adata.obsm:
        spatial = adata.obsm['spatial']
    elif 'x_array' in adata.obs.columns and 'y_array' in adata.obs.columns:
        spatial = adata.obs[['x_array', 'y_array']].values
    elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        spatial = adata.obs[['x', 'y']].values
    else:
        raise ValueError("Could not find spatial coordinates in obsm['spatial'] or obs[['x_array','y_array']] / obs[['x','y']].")
    adata.obsm['spatial'] = spatial.astype(float)

    # ---- labels: choose a valid column, then map to 'layer_guess' ----
    if annotation_col is not None and annotation_col in adata.obs.columns:
        label_src = annotation_col
    elif 'ground_truth' in adata.obs.columns:
        label_src = 'ground_truth'
    elif 'Annotation' in adata.obs.columns:
        label_src = 'Annotation'
    else:
        raise ValueError(
            f"No label column found. Tried: {annotation_col}, 'ground_truth', 'Annotation'. "
            f"Available obs columns: {list(adata.obs.columns)}"
        )

    adata.obs['layer_guess'] = adata.obs[label_src]

    # Cast to string before filling NA to avoid categorical dtype errors
    adata.obs['layer_guess'] = adata.obs['layer_guess'].astype('string').fillna('Unknown').astype(str)

    print(f"  -> Using labels from obs['{label_src}'] (n_classes={pd.Series(adata.obs['layer_guess']).nunique()})")



    # ---- 5) preprocessing (Scanpy, SEDR-like but lighter defaults) ----
    print("Preprocessing (filter/normalize/HVG/scale/PCA)...")

    # keep a count layer for HVG
    if 'counts' in adata.layers:
        # keep sparse if present
        adata.layers['count'] = adata.layers['counts']
    else:
        adata.layers['count'] = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable'] == True].copy()
    sc.pp.scale(adata)

    pca = PCA(n_components=pca_dim, random_state=2023)
    X_mat = adata.X if not sp.issparse(adata.X) else adata.X.toarray()
    feat_pca = pca.fit_transform(X_mat)
    features = torch.FloatTensor(feat_pca)

    # ---- 6) graph construction: spatial kNN ----
    print(f"Constructing Spatial Graph (k={k_neighbors})...")
    cell_position = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)

    src, dst = [], []
    for i in range(cell_position.shape[0]):
        for j in range(1, k_neighbors + 1):
            nb = int(indices[i, j])
            src.append(i); dst.append(nb)
            src.append(nb); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    # ---- 7) label encoding + masks ----
    le = LabelEncoder()
    labels_np = le.fit_transform(adata.obs['layer_guess'].astype(str))
    labels = torch.LongTensor(labels_np)
    class_names = le.classes_

    nnodes = features.shape[0]
    nfeats = features.shape[1]

    valid_mask = (adata.obs['layer_guess'].astype(str) != 'Unknown')

    valid_indices = np.where(valid_mask.values)[0]
    np.random.shuffle(valid_indices)

    n_valid = len(valid_indices)
    train_end = int(0.6 * n_valid)
    val_end = int(0.8 * n_valid)

    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)

    train_mask[valid_indices[:train_end]] = True
    val_mask[valid_indices[train_end:val_end]] = True
    test_mask[valid_indices[val_end:]] = True

    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    # ---- 8) structural encoding (SE) ----
    se_path = '../data/se/DPI'
    if not os.path.exists(se_path):
        os.makedirs(se_path)

    file_name = os.path.join(se_path, f'DPI_{adata.n_obs}_k{k_neighbors}_{str_enc_dim}.pt')
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding...')
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=str_enc_dim)
        torch.save(se, file_name)
        print(f'Done. Saved SE to {file_name}')

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats


def _brca_xenium_subset_indices(adata, max_cells=20000, seed=0):
    """Deterministically sample Xenium cells for the full-batch DHALF model."""
    if adata.n_obs <= max_cells:
        return np.arange(adata.n_obs)
    return np.sort(np.random.default_rng(seed).choice(adata.n_obs, size=max_cells, replace=False))


def load_BRCA_Xexium(path=None, k_neighbors=20, pca_dim=60, max_cells=20000):
    if path is None:
        path = './data/BRCA/BRCA_Xexium.h5ad'
    if not os.path.exists(path):
        raise FileNotFoundError(f"BRCA Xenium file not found at: {path}")

    print(f"Loading BRCA Xenium data from {path} (max_cells={max_cells}) ...")
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    subset_idx = _brca_xenium_subset_indices(adata, max_cells=max_cells, seed=0)
    adata = adata[subset_idx].copy()

    if 'spatial' not in adata.obsm:
        if {'x_centroid', 'y_centroid'}.issubset(adata.obs.columns):
            adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].astype(float).to_numpy()
        else:
            raise ValueError("BRCA Xenium requires obsm['spatial'] or obs x_centroid/y_centroid.")
    spatial_coords = np.asarray(adata.obsm['spatial'], dtype=float)

    if 'Raw' in adata.layers:
        adata.layers['count'] = adata.layers['Raw'].copy()
        adata.X = adata.layers['Raw'].copy()
    elif 'counts' in adata.layers:
        adata.layers['count'] = adata.layers['counts'].copy()
        adata.X = adata.layers['counts'].copy()
    else:
        adata.layers['count'] = adata.X.copy()

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_genes(adata, min_counts=10)
    n_hvg = min(adata.n_vars, 300)
    try:
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', layer='count', n_top_genes=n_hvg)
        adata = adata[:, adata.var['highly_variable']].copy()
    except Exception as e:
        print(f"!! Warning: BRCA Xenium HVG selection failed ({e}). Proceeding with all genes.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    n_components = min(int(pca_dim), adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_components, svd_solver='arpack')
    features = torch.FloatTensor(adata.obsm['X_pca'].copy())

    nnodes, nfeats = features.shape
    k_eff = min(int(k_neighbors), nnodes - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='ball_tree').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    src, dst = [], []
    for i in range(nnodes):
        for j in range(1, k_eff + 1):
            src.append(i); dst.append(indices[i, j])
            src.append(indices[i, j]); dst.append(i)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    labels = torch.full((nnodes,), -1, dtype=torch.long)
    class_names = None
    idx = np.random.permutation(nnodes)
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)
    train_mask[idx[:int(0.6 * nnodes)]] = True
    val_mask[idx[int(0.6 * nnodes):int(0.8 * nnodes)]] = True
    test_mask[idx[int(0.8 * nnodes):]] = True
    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    se_path = '../data/se/BRCA_Xexium'
    os.makedirs(se_path, exist_ok=True)
    se_file = os.path.join(se_path, f'BRCA_Xexium_subset{nnodes}_k{k_eff}_16.pt')
    se = torch.load(se_file) if os.path.exists(se_file) else None
    if se is None or se.shape[0] != nnodes:
        print('Computing structural encoding for BRCA Xenium...')
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=16)
        torch.save(se, se_file)

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats


def load_data(dataset_name, data_path=None, spatial_k=None):

    if dataset_name == 'E18_mouse_brain':
        path = data_path or './data/E18.5_mouse_brain'
        return load_candies_E18(path)

    if dataset_name == '151673':
            path = './data/151673'
            return load_151673(path)
    
    if dataset_name == '151674':
            path = './data/151674'
            return load_151674(path)

    if dataset_name == 'mouse_embryo':
            path = './data/mouse_embryo'
            return load_mouse_embryo(path)

    if dataset_name == 'Stereo_AB':
        return load_Stereo_AB(data_path)

    if dataset_name == 'MOB':
        return load_MOB(data_path)

    if dataset_name == 'MOB_section3':
        return load_MOB(data_path, section_filter=3)
    
    if dataset_name == 'Stereo_mouse_brain':
            return load_Stereo_mouse_brain()

    if dataset_name in {'DPI', 'Stereo_DPI'}:
        path = data_path or './data/DPI/2DPI_1.h5ad'
        return load_Stereo_DPI(path)

    if dataset_name == 'BRCA_Xexium':
        path = data_path or './data/BRCA/BRCA_Xexium.h5ad'
        return load_BRCA_Xexium(path)

    if dataset_name in {'BRCA', 'Breast_Caner'}:
        path = data_path or './data/BRCA/Breast_Caner'
        return load_BRCA_visium(path, k_neighbors=spatial_k or 40)

    if dataset_name == 'CNS':
        path = data_path or './data/CNS/CNS3.h5ad'
        return load_CNS(path, k_neighbors=spatial_k or 10)


    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name)
    elif dataset_name in ['chameleon']:
        dataset = WikipediaNetwork(path, dataset_name)
    elif dataset_name in ['squirrel']:
        dataset = WikipediaNetwork(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['actor']:
        dataset = Actor(path)
    elif dataset_name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset_name)
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['cs', 'physics']:
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['wikics']:
        dataset = WikiCS(path)

    data = dataset[0]

    edges = remove_self_loops(data.edge_index)[0]

    features = data.x
    [nnodes, nfeats] = features.shape
    nclasses = torch.max(data.y).item() + 1

    if dataset_name in ['computers', 'photo', 'cs', 'physics', 'wikics']:
        train_mask, val_mask, test_mask = get_split(nnodes)
    else:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    if len(train_mask.shape) < 2:
        train_mask = train_mask.unsqueeze(1)
        val_mask = val_mask.unsqueeze(1)
        test_mask = test_mask.unsqueeze(1)

    labels = data.y

    path = '../data/se/{}'.format(dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + '/{}_{}.pt'.format(dataset_name, 16)
    if os.path.exists(file_name):
        se = torch.load(file_name)
        # print('Load exist structural encoding.')
    else:
        print('Computing structural encoding...')
        se = get_structural_encoding(edges, nnodes)
        torch.save(se, file_name)
        print('Done. The structural encoding is saved as: {}.'.format(file_name))

    return features, edges, se, train_mask, val_mask, test_mask, labels, nnodes, nfeats

import json
import matplotlib.image as mpimg
def load_visium_spatial_uns(data_dir: str, library_id: str = "visium"):
    """Build a minimal AnnData.uns['spatial']-style dict from a 10x Visium folder.

    Expected structure:
      data_dir/
        spatial/
          tissue_hires_image.png
          tissue_lowres_image.png
          scalefactors_json.json

    Returns:
      spatial_uns (dict): {library_id: {"images": {...}, "scalefactors": {...}, "metadata": {...}}}
    """
    spatial_dir = os.path.join(data_dir, "spatial")
    sf_path = os.path.join(spatial_dir, "scalefactors_json.json")
    hires_path = os.path.join(spatial_dir, "tissue_hires_image.png")
    lowres_path = os.path.join(spatial_dir, "tissue_lowres_image.png")

    if not os.path.exists(sf_path):
        raise FileNotFoundError(f"Missing scalefactors_json.json: {sf_path}")

    with open(sf_path, "r", encoding="utf-8") as f:
        scalefactors = json.load(f)

    images = {}
    if os.path.exists(hires_path):
        images["hires"] = mpimg.imread(hires_path)
    if os.path.exists(lowres_path):
        images["lowres"] = mpimg.imread(lowres_path)

    # Scanpy expects: uns['spatial'][library_id]['images'][key] as array
    spatial_uns = {
        library_id: {
            "images": images,
            "scalefactors": scalefactors,
            "metadata": {},
        }
    }
    return spatial_uns


warnings.simplefilter("ignore")
