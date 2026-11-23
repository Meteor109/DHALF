import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scanpy as sc
import pandas as pd
import anndata
# R is imported lazily by mclust_R(). KMeans-only downstream runs therefore do
# not require a local R installation merely to import DHALF utilities.
robjects = None
numpy2ri = None
_RPY2_IMPORT_ERROR = None
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from matplotlib.lines import Line2D
import os
import fcntl
EOS = 1e-10
import seaborn as sns


class _VariantDFileLock:
    # Variant D addition: guard summary/CSV appends when runs are launched in parallel.
    def __init__(self, target_path):
        self.lock_path = target_path + ".lock"
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self._fh = open(self.lock_path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def normalize_adj(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value
        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

def get_adj_from_edges(edges, weights, nnodes):
    adj = torch.zeros(nnodes, nnodes).cuda()
    adj[edges[0], edges[1]] = weights
    return adj

def generate_random_node_pairs(nnodes, nedges, backup=300):
    rand_edges = np.random.choice(nnodes, size=(nedges + backup) * 2, replace=True)
    rand_edges = rand_edges.reshape((2, nedges + backup))
    rand_edges = torch.from_numpy(rand_edges)
    rand_edges = rand_edges[:, rand_edges[0,:] != rand_edges[1,:]]
    rand_edges = rand_edges[:, 0: nedges]
    return rand_edges.cuda()

def eval_test_mode(embedding, labels, train_mask, val_mask, test_mask):
    X = embedding.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    X_train = X[train_mask.cpu()]
    X_val = X[val_mask.cpu()]
    X_test = X[test_mask.cpu()]
    y_train = Y[train_mask.cpu()]
    y_val = Y[val_mask.cpu()]
    y_test = Y[test_mask.cpu()]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)

    return acc_test * 100, acc_val * 100

# --- Added helpers ---

def mclust_R(embedding, num_cluster, modelNames='EEE', random_seed=2023):
    """
    Cluster with the R package mclust (PCA-accelerated).
    """
    global robjects, numpy2ri, _RPY2_IMPORT_ERROR
    if robjects is None or numpy2ri is None:
        try:
            import rpy2.robjects as _robjects
            from rpy2.robjects import numpy2ri as _numpy2ri
            robjects = _robjects
            numpy2ri = _numpy2ri
        except Exception as exc:
            _RPY2_IMPORT_ERROR = exc

    if robjects is None or numpy2ri is None:
        print(f"!! mclust is unavailable because rpy2/R could not be initialized: {_RPY2_IMPORT_ERROR}")
        return None

    numpy2ri.activate()
    robjects.r['set.seed'](random_seed)
    
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
        
    # Reduce to 30 dimensions when the embedding is wider
    if embedding.shape[1] > 30:
        pca = PCA(n_components=30, random_state=random_seed)
        embedding = pca.fit_transform(embedding)
    
    try:
        robjects.r.library("mclust")
    except Exception as e:
        print(f"!! Error loading 'mclust': {e}")
        return None

    robjects.r('''
    cluster_mclust <- function(data, n_clusters, modelNames) {
      dimnames(data) <- NULL
      if (is.null(modelNames)) {
          result <- mclust::Mclust(data, G = n_clusters)
      } else {
          result <- mclust::Mclust(data, G = n_clusters, modelNames = modelNames)
      }
      return(result)
    }
    ''')
    rmclust = robjects.r['cluster_mclust']

    data = np.ascontiguousarray(embedding, dtype=np.float64)
    r_modelNames = modelNames if modelNames is not None else robjects.r("NULL")

    try:
        res = rmclust(data, num_cluster, r_modelNames)
        if 'classification' in res.names:
            mclust_res = np.array(res.rx2('classification')).astype('int')
        else:
            mclust_res = np.array(res[-2]).astype('int')
        return mclust_res - 1
    except Exception as e:
        print(f"!! Mclust Execution Error: {e}")
        return None

def eval_clustering(embedding, labels, n_clusters):
    """ KMeans fallback. """
    from sklearn.cluster import KMeans
    X = embedding.detach().cpu().numpy() if torch.is_tensor(embedding) else embedding
    Y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    pred_labels = kmeans.fit_predict(X)
    valid_mask = Y >= 0
    ari = adjusted_rand_score(Y[valid_mask], pred_labels[valid_mask]) if np.sum(valid_mask) > 0 else 0.0
    return ari, pred_labels


def spatial_boundary_f1(pred_labels, true_labels, edges):
    """Boundary F1: F1 between predicted and ground-truth cross-domain spatial edges.

    Spots without ground truth (label < 0) are ignored, and an edge is only scored
    when both endpoints carry a label. Mirrors the B-F1 definition used for Table 1.
    """
    pred = pred_labels.detach().cpu().numpy() if torch.is_tensor(pred_labels) else np.asarray(pred_labels)
    true = true_labels.detach().cpu().numpy() if torch.is_tensor(true_labels) else np.asarray(true_labels)
    edge_np = edges.detach().cpu().numpy() if torch.is_tensor(edges) else np.asarray(edges)

    if edge_np.shape[0] != 2 and edge_np.shape[1] == 2:
        edge_np = edge_np.T
    keep = (true[edge_np[0]] >= 0) & (true[edge_np[1]] >= 0)
    edge_np = edge_np[:, keep]
    if edge_np.shape[1] == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "n_gt_boundary": 0, "n_pred_boundary": 0}

    gt = true[edge_np[0]] != true[edge_np[1]]
    pr = pred[edge_np[0]] != pred[edge_np[1]]
    tp = int(np.logical_and(gt, pr).sum())
    n_pred = int(pr.sum())
    n_gt = int(gt.sum())
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_gt_boundary": n_gt,
        "n_pred_boundary": n_pred,
    }


def build_variant_d_semantic_mask(features, k=10):
    # Variant D addition: cosine top-k semantic positives, computed once and reused.
    X = features.detach().cpu().numpy() if torch.is_tensor(features) else np.asarray(features)
    X = np.asarray(X, dtype=np.float32)
    n_nodes = X.shape[0]
    k_eff = max(1, min(int(k), n_nodes - 1))

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric='cosine').fit(X)
    _, indices = nbrs.kneighbors(X)

    mask = np.eye(n_nodes, dtype=np.float32)
    rows = np.repeat(np.arange(n_nodes), k_eff)
    cols = indices[:, 1:k_eff + 1].reshape(-1)
    mask[rows, cols] = 1.0
    mask[cols, rows] = 1.0
    return torch.from_numpy(mask)


def build_variant_d_spatial_mask(edges, nnodes, weights=None, threshold=0.5, polarity="all"):
    """Build a symmetric positive mask that only uses spatial edges.

    polarity:
      "all"          -- every spatial edge is a positive pair.
      "homophilic"   -- spatial edges whose low-pass weight exceeds `threshold`.
      "heterophilic" -- spatial edges whose high-pass weight (1 - low-pass weight)
                        exceeds `threshold`; used as the "high-pass spatial
                        positives" control that deliberately pulls potential
                        cross-domain neighbours together.
    """
    device = edges.device if torch.is_tensor(edges) else torch.device("cpu")
    mask = torch.eye(nnodes, device=device)

    if edges is None or edges.numel() == 0:
        return mask

    n_edges = edges.shape[1]
    if polarity == "all" or weights is None:
        keep = torch.ones(n_edges, dtype=torch.bool, device=device)
    else:
        weights = weights.detach() if torch.is_tensor(weights) else torch.as_tensor(weights, device=device)
        if weights.shape[0] != n_edges:
            keep = torch.ones(n_edges, dtype=torch.bool, device=device)
        elif polarity == "homophilic":
            keep = weights > threshold
        elif polarity == "heterophilic":
            keep = (1.0 - weights) > threshold
        else:
            raise ValueError(f"Unsupported spatial mask polarity: {polarity}")
        if keep.sum() == 0:
            keep = torch.ones(n_edges, dtype=torch.bool, device=device)

    src = edges[0, keep].long()
    dst = edges[1, keep].long()
    mask[src, dst] = 1.0
    mask[dst, src] = 1.0
    return mask


def build_variant_d_hom_pos_mask(edges, weights_lp, nnodes, threshold=0.5, use_hom_pos=True):
    # Variant D addition: homophilic positives from the current low-pass edge weights.
    return build_variant_d_spatial_mask(
        edges, nnodes, weights=weights_lp, threshold=threshold,
        polarity="homophilic" if use_hom_pos else "all",
    )


def append_variant_d_summary(summary_path, args, best_info):
    # Variant D addition: append a compact run summary for cross-dataset tracking.
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    params = vars(args).copy()
    changed_files = "main.py, model.py, utils.py"
    new_args = (
        "--lambda_fl, --lambda_fh, --lambda_base, --lambda_mix, --mix_detach_branches, --sem_k, "
        "--variant_d_warmup, --gate_hidden, --full_mlp_hidden, "
        "--full_mlp_out, --use_hom_pos, --variant_d_gate_mode"
    )
    command = best_info.get("command", "")
    with _VariantDFileLock(summary_path):
        f = open(summary_path, "a", encoding="utf-8")
        try:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Ablation: {getattr(args, 'ablation_name', '') or 'NA'}\n")
            f.write(f"Seed: {getattr(args, 'seed', 0)}\n")
            if best_info.get("has_ground_truth", True):
                f.write(f"Best ARI: {best_info.get('ari', -1.0):.4f}\n")
                f.write(f"Best raw ARI: {best_info.get('raw_ari', -1.0):.4f}\n")
                f.write(f"Best refined ARI: {best_info.get('refined_ari', best_info.get('ari', -1.0)):.4f}\n")
            else:
                f.write("Ground truth: unavailable; ARI skipped.\n")
            f.write(f"Best representation: {best_info.get('repr', 'NA')}\n")
            f.write(f"Best epoch: {best_info.get('epoch', 'NA')}\n")
            f.write(f"Best command: {command}\n")
            f.write("Best hyperparameters: ")
            f.write(
                f"lambda_fl={args.lambda_fl}, lambda_fh={args.lambda_fh}, "
                f"lambda_base={args.lambda_base}, lambda_mix={getattr(args, 'lambda_mix', 1.0)}, "
                f"mix_detach_branches={getattr(args, 'mix_detach_branches', 1)}, sem_k={args.sem_k}, "
                f"variant_d_warmup={args.variant_d_warmup}, "
                f"gate_mode={getattr(args, 'variant_d_gate_mode', 'node')}, "
                f"gate_hidden={args.gate_hidden}, full_mlp_hidden={args.full_mlp_hidden}, "
                f"full_mlp_out={args.full_mlp_out}, use_hom_pos={args.use_hom_pos}\n"
            )
            f.write(f"Image path: {best_info.get('plot_path', 'NA')}\n")
            f.write(f"Checkpoint path: {best_info.get('checkpoint_path', 'NA')}\n")
            f.write(
                f"Best diagnostics: gate_mean={best_info.get('gate_mean', 'NA')}, "
                f"gate_std={best_info.get('gate_std', 'NA')}, "
                f"L_FL={best_info.get('l_fl', 'NA')}, L_FH={best_info.get('l_fh', 'NA')}, "
                f"L_base={best_info.get('base_loss', 'NA')}\n"
            )
            f.write(f"Modified files: {changed_files}\n")
            f.write(f"New CLI parameters: {new_args}\n")
            f.write("Training objective: L_rank + lambda_base * L_base + lambda_fl * L_FL + lambda_fh * L_FH + "
                    "lambda_mix * L_MIX (L_MIX trains the Eq.12 gate only)\n")
            if "diagnostics_text" in best_info:
                f.write(best_info["diagnostics_text"] + "\n")
            f.write(f"All args: {params}\n")
        finally:
            f.close()


def append_variant_d_run_csv(csv_path, args, best_info, metrics_by_embedding):
    # Variant D addition: one structured row per run for seed/trial aggregation.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {
        "dataset": args.dataset,
        "has_ground_truth": best_info.get("has_ground_truth", True),
        "ablation_name": getattr(args, "ablation_name", ""),
        "seed": args.seed,
        "k": args.k,
        "margin_hom": args.margin_hom,
        "margin_het": args.margin_het,
        "refine_k": args.refine_k,
        "variant_d_gate_mode": getattr(args, "variant_d_gate_mode", "node"),
        "lambda_fl": args.lambda_fl,
        "lambda_fh": args.lambda_fh,
        "lambda_base": args.lambda_base,
        "sem_k": args.sem_k,
        "gate_hidden": args.gate_hidden,
        "variant_d_warmup": args.variant_d_warmup,
        "epochs": args.epochs,
        "eval_freq": args.eval_freq,
        "best_embedding": best_info.get("repr", "NA"),
        "best_raw_ari": best_info.get("raw_ari", np.nan),
        "best_refined_ari": best_info.get("refined_ari", best_info.get("ari", np.nan)),
        "best_boundary_f1": best_info.get("boundary_f1", np.nan),
        "best_epoch": best_info.get("epoch", np.nan),
        "gate_mean": best_info.get("gate_mean", np.nan),
        "gate_std": best_info.get("gate_std", np.nan),
        "l_fl": best_info.get("l_fl", np.nan),
        "l_fh": best_info.get("l_fh", np.nan),
        "base_loss": best_info.get("base_loss", np.nan),
        "figure_path": best_info.get("plot_path", "NA"),
        "checkpoint_path": best_info.get("checkpoint_path", "NA"),
        "command": best_info.get("command", ""),
        "use_hom_pos": getattr(args, "use_hom_pos", 1),
        "fl_mask_mode": getattr(args, "fl_mask_mode", None) or (
            "spatial_homophilic" if getattr(args, "use_hom_pos", 1) else "spatial_all"),
        "fh_mask_mode": getattr(args, "fh_mask_mode", None) or "semantic",
        "no_aux_branch": int(getattr(args, "no_aux_branch", 0)),
    }
    for emb_name in ["z_mix", "z_full", "concat", "z_low", "z_high"]:
        emb_metrics = metrics_by_embedding.get(emb_name, {})
        row[f"{emb_name}_raw_ari"] = emb_metrics.get("raw_ari", np.nan)
        row[f"{emb_name}_refined_ari"] = emb_metrics.get("refined_ari", np.nan)
        row[f"{emb_name}_boundary_f1"] = emb_metrics.get("boundary_f1", np.nan)
        row[f"{emb_name}_best_epoch"] = emb_metrics.get("epoch", np.nan)

    with _VariantDFileLock(csv_path):
        df = pd.DataFrame([row])
        if os.path.exists(csv_path):
            # Variant D addition: migrate older CSVs in place when new bookkeeping
            # columns such as refine_k are added between experiment rounds.
            old_df = pd.read_csv(csv_path)
            all_columns = list(dict.fromkeys(list(old_df.columns) + list(df.columns)))
            old_df = old_df.reindex(columns=all_columns)
            df = df.reindex(columns=all_columns)
            pd.concat([old_df, df], ignore_index=True).to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)

def plot_tsne(embedding, labels, save_name='tsne_plot.png'):
    """ t-SNE visualisation saved to disk. """
    X = embedding.detach().cpu().numpy() if torch.is_tensor(embedding) else embedding
    
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='jet', s=10, alpha=0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"  [Plot] Saved to {save_name}")


def smooth_embedding_spatial(embedding, spatial, k=15, alpha=0.5):

    if spatial is None:
        return embedding
    
    # Ensure a numpy array
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
        
    print(f"  [Refine] Smoothing embedding (k={k}, alpha={alpha})...")
    
    # Fit the spatial coordinates
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(spatial)
    _, indices = nbrs.kneighbors(spatial)
    
    emb_smooth = np.zeros_like(embedding)
    for i in range(len(embedding)):
        # Average embedding of the neighbours
        neighbor_emb = np.mean(embedding[indices[i]], axis=0)
        # Weighted fusion
        emb_smooth[i] = alpha * embedding[i] + (1 - alpha) * neighbor_emb
        
    return emb_smooth

def refine_label_spatial(labels, spatial, k=20):
    """
    Strategy 2: label-space refinement (majority vote).
    If most neighbours belong to class A, the spot is reassigned to A.
    """
    if spatial is None:
        return labels
        
    print(f"  [Refine] Refining labels (k={k})...")
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(spatial)
    _, indices = nbrs.kneighbors(spatial)
    
    new_labels = labels.copy()
    for i in range(len(labels)):
        neighbor_labels = labels[indices[i]]
        # Most frequent label
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        new_labels[i] = most_common
        
    return new_labels


def plot_spatial(
    spatial,
    pred_labels,
    true_labels=None,
    title="Prediction",
    save_name="spatial.png",
    emb=None,
    spot_size=200,
    point_size=None,
    class_names=None,
    all_spatial_plot=None,
    spatial_uns=None,
    library_id=None,
    img_key="hires",
    xlim=None, 
    ylim=None,
    manual_title="Manual Annotation",
    show_manual_panel=True
):
    """Scanpy-like spatial + (optional) UMAP plotting.

    Key features for Visium-style datasets:
    - If `spatial_uns` + `library_id` + `img_key` are provided, draw H&E image as background
      and align scatter using scalefactors_json (fullres pixel coords * scalef).
    - If `all_spatial_plot` is provided, draw empty-ring dots for *all* spots (including in_tissue=0),
      then overlay colored tissue spots.

    Notes:
    - `spatial` should be tissue-spot coordinates in **fullres pixel** space: (x=pxl_col_in_fullres, y=pxl_row_in_fullres).
    - `all_spatial_plot` should be all-spot coordinates in the same fullres pixel space.
    - This function only changes the *presentation* (background/origin/legend layout); it does NOT change model outputs.
    - `point_size` directly controls matplotlib/UMAP scatter size for final result figures.
    """
    if spatial is None:
        return

    # --- to numpy ---
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
    if true_labels is not None and torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()
    if torch.is_tensor(spatial):
        spatial = spatial.detach().cpu().numpy()
    if emb is not None and torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()
    if all_spatial_plot is not None and torch.is_tensor(all_spatial_plot):
        all_spatial_plot = all_spatial_plot.detach().cpu().numpy()

    spatial = np.asarray(spatial)
    if all_spatial_plot is not None:
        all_spatial_plot = np.asarray(all_spatial_plot)

    # --- build categorical labels (for stable color/legend) ---
    # Manual annotation
    if true_labels is None:
        manual = pd.Categorical(["Unknown"] * spatial.shape[0])
    else:
        if class_names is not None:
            mapped = []
            for v in true_labels:
                if v == -1:
                    mapped.append("Unknown")
                elif int(v) < len(class_names):
                    mapped.append(str(class_names[int(v)]))
                else:
                    mapped.append(str(v))
            manual = pd.Categorical(mapped)
        else:
            manual = pd.Categorical([str(v) for v in true_labels])

    # Prediction
    pred_values = [f"Cluster {int(v)}" if str(v).lstrip('-').isdigit() else str(v) for v in pred_labels]
    pred_categories = sorted(
        set(pred_values),
        key=lambda value: (0, int(value.split()[-1]))
        if value.startswith("Cluster ") and value.split()[-1].lstrip("-").isdigit()
        else (1, value),
    )
    pred = pd.Categorical(pred_values, categories=pred_categories)

    def _cat_color_map(cat: pd.Categorical):
        cats = list(cat.categories)
        cmap = plt.get_cmap("tab20")
        return {c: cmap(i % 20) for i, c in enumerate(cats)}

    manual_cmap = _cat_color_map(manual)
    pred_cmap = _cat_color_map(pred)

    # --- background image (optional) ---
    img = None
    sf = 1.0
    if spatial_uns is not None and library_id is not None:
        try:
            lib = spatial_uns[library_id]
            img = lib.get("images", {}).get(img_key, None)
            sfs = lib.get("scalefactors", {})
            # 10x keys: tissue_hires_scalef / tissue_lowres_scalef
            sf = float(sfs.get(f"tissue_{img_key}_scalef", 1.0))
        except Exception:
            img = None
            sf = 1.0

    # scale coords to match chosen image
    spatial_plot = spatial * sf
    grid = None
    if all_spatial_plot is not None:
        # Guard: make sure it's (N,2)
        grid = np.asarray(all_spatial_plot)
        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError(f"all_spatial_plot must be (N,2), got {grid.shape}")
        grid = grid * sf

    def _auto_scatter_s(ax, xy, spot_size, base=200.0, frac=0.85):
        """
        Convert a scanpy-style spot_size (200 is the usual value) into matplotlib scatter s (points^2),
        so that the spot diameter is about frac times the nearest-neighbour distance and dots do not blur together.
        """
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[0] < 5:
            return float(spot_size)

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(xy)
        d, _ = nbrs.kneighbors(xy)
        nn = np.median(d[:, 1])
        if not np.isfinite(nn) or nn <= 0:
            return float(spot_size)

        # spot_size=200 is treated as the 1.0x baseline
        scale = float(spot_size) / float(base)
        diam_data = frac * nn * scale  # target diameter in data coordinates

        fig = ax.figure
        p0 = ax.transData.transform((0, 0))
        p1 = ax.transData.transform((diam_data, 0))
        diam_px = abs(p1[0] - p0[0])
        diam_pt = diam_px * 72.0 / fig.dpi
        return max(1.0, diam_pt ** 2)


    # --- plotting helpers ---
    def _draw_panel(ax, coords, cat, cmap_map, panel_title):
        # background image
        if img is not None:
            ax.imshow(img, origin="upper", zorder=0)
            h, w = img.shape[0], img.shape[1]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # same convention as scanpy: origin at the top left
            base_xy = grid if grid is not None else coords
        else:
            base_xy = grid if grid is not None else coords
            xmin, xmax = base_xy[:, 0].min(), base_xy[:, 0].max()
            ymin, ymax = base_xy[:, 1].min(), base_xy[:, 1].max()
            padx = (xmax - xmin) * 0.02
            pady = (ymax - ymin) * 0.02
            ax.set_xlim(xmin - padx, xmax + padx)
            ax.set_ylim(ymax + pady, ymin - pady)  # keep y pointing down, as usual for ST plots

        # ---- Point size only affects visualization, not model outputs or ARI. ----
        base_xy = grid if grid is not None else coords
        s_plot = float(point_size) if point_size is not None else _auto_scatter_s(ax, base_xy, spot_size)
        # empty rings for all spots
        if grid is not None:
            ax.scatter(
                grid[:, 0], grid[:, 1],
                s=s_plot, facecolors="none",
                edgecolors=(0, 0, 0, 0.25),
                linewidths=0.25, marker="h", zorder=1
            )
        # colored tissue spots (draw per-category for stable legend/colors)
        handles = []
        labels = []
        for c in cat.categories:
            mask = (np.asarray(cat) == c)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=s_plot, c=[cmap_map[c]],
                edgecolors="none", linewidths=0, marker="h", zorder=2
            )

            handles.append(plt.Line2D([0], [0], marker='h', linestyle='', markersize=7,
                                     markerfacecolor=cmap_map[c], markeredgecolor='none'))
            labels.append(str(c))

        ax.set_title(panel_title, fontsize=14)
        ax.set_aspect("equal")

        ax.axis("off")


        # legend at right margin
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize="small", handletextpad=0.4)

        # If image exists, clamp view to image extent so it looks like scanpy
        if img is not None:
            h, w = img.shape[0], img.shape[1]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # keep origin at top
            
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        
        if ylim is not None:
            # Handle the Y-axis direction:
            # ST plots usually flip Y (0 at the top, large values at the bottom).
            # Force large values at the bottom and small values at the top so the figure is never upside down.
            bottom, top = max(ylim), min(ylim)
            ax.set_ylim(bottom, top)


    # --- Spatial: manual + prediction when labels exist, prediction-only otherwise ---
    if show_manual_panel:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        _draw_panel(axes[0], spatial_plot, manual, manual_cmap, manual_title)
        _draw_panel(axes[1], spatial_plot, pred, pred_cmap, title)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        _draw_panel(ax, spatial_plot, pred, pred_cmap, title)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved Spatial map to {save_name}")

    # --- Optional UMAP from emb ---
    if emb is not None:
        try:
            ad = anndata.AnnData(X=emb)
            ad.obs["Manual Annotation"] = manual
            ad.obs["Prediction"] = pred
            sc.pp.neighbors(ad, use_rep="X", metric="cosine")
            sc.tl.umap(ad, random_state=0)

            umap_save = save_name.replace(".png", "_UMAP.png")
            umap_point_size = float(point_size) if point_size is not None else spot_size
            if show_manual_panel:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                sc.pl.umap(
                    ad,
                    color="Manual Annotation",
                    ax=axes[0],
                    show=False,
                    title=f"{manual_title} (UMAP)" if manual_title else "",
                    s=umap_point_size,
                    palette="tab20",
                    frameon=False,
                    legend_loc="right margin",
                    legend_fontsize="small",
                )
                sc.pl.umap(
                    ad,
                    color="Prediction",
                    ax=axes[1],
                    show=False,
                    title=title.replace("Prediction", "UMAP"),
                    s=umap_point_size,
                    palette="tab20",
                    frameon=False,
                    legend_loc="right margin",
                    legend_fontsize="small",
                )
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                sc.pl.umap(
                    ad,
                    color="Prediction",
                    ax=ax,
                    show=False,
                    title=title.replace("Prediction", "UMAP"),
                    s=umap_point_size,
                    palette="tab20",
                    frameon=False,
                    legend_loc="right margin",
                    legend_fontsize="small",
                )
            plt.tight_layout()
            plt.savefig(umap_save, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  [Plot] Saved UMAP projection to {umap_save}")
        except Exception as e:
            print(f"  !! UMAP plot failed: {e}")




def louvain_clustering(embedding, n_clusters, n_neighbors=12, max_steps=20):
    """
    Louvain clustering with a bisection search for the resolution that matches n_clusters.
    """
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
        
    # 1) Build the AnnData object and the neighbour graph
    adata = anndata.AnnData(X=embedding)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    
    # 2) Bisection search over the resolution
    # A larger Louvain resolution yields more clusters, a smaller one fewer.
    res_min = 0.1
    res_max = 3.0
    res_curr = 1.0
    
    best_labels = None
    best_diff = float('inf')
    
    print(f"  [Louvain] Searching resolution for k={n_clusters}...", end='', flush=True)
    
    for step in range(max_steps):
        # Run Louvain
        try:
            sc.tl.louvain(adata, resolution=res_curr, key_added='louvain')
        except ImportError:
            # Fall back to Leiden (a drop-in successor) when louvain is unavailable
            print("\n  !! Louvain not installed. Trying Leiden instead...")
            sc.tl.leiden(adata, resolution=res_curr, key_added='louvain')
            
        labels = adata.obs['louvain'].astype(int).values
        n_found = len(np.unique(labels))
        
        # Keep the closest result
        diff = abs(n_found - n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_labels = labels
            
        # Exact match
        if n_found == n_clusters:
            print(f" Found! (res={res_curr:.3f})")
            return labels
        
        # Update the search interval
        if n_found > n_clusters:
            res_max = res_curr
            res_curr = (res_min + res_curr) / 2
        else:
            res_min = res_curr
            res_curr = (res_max + res_curr) / 2
            
    print(f" Done. (Best match: {len(np.unique(best_labels))} clusters, res={res_curr:.3f})")
    return best_labels


def visualize_scanpy_style(features, spatial, true_labels, pred_labels, ari, save_name_prefix):
    """
    Scanpy-style visualisation:
    1. Spatial map (left: ground truth, right: prediction)
    2. UMAP projection (left: ground truth, right: prediction)
    """
    dataset_title = os.path.basename(str(save_name_prefix))
    for marker in ["_variantD", "_Trial", "_Global", "_Best", "_Spatial"]:
        dataset_title = dataset_title.split(marker)[0]
    if dataset_title.isdigit():
        dataset_title = f"DLPFC{dataset_title}"
    figure_title = f"{dataset_title}\nari={ari:.4f}"

    # 1) Wrap everything in an AnnData object
    if torch.is_tensor(features): features = features.detach().cpu().numpy()
    if torch.is_tensor(true_labels): true_labels = true_labels.detach().cpu().numpy()
    if isinstance(spatial, torch.Tensor): spatial = spatial.detach().cpu().numpy()
    
    # Create the AnnData object
    adata = anndata.AnnData(X=features)
    adata.obsm['spatial'] = spatial
    
    # Store labels as strings so they are treated as categorical colours
    # Handle -1 (Unknown) so plotting does not fail
    true_labels_str = true_labels.astype(str)
    pred_labels_str = pred_labels.astype(str)
    
    adata.obs['Manual Annotation'] = pd.Categorical(true_labels_str)
    adata.obs['Clustering'] = pd.Categorical(pred_labels_str)
    
    # 2) Compute the UMAP projection (metric='cosine')
    # print("  [Vis] Computing UMAP...")
    sc.pp.neighbors(adata, use_rep='X', metric='cosine')
    sc.tl.umap(adata)

    # -------------------------------------------------------
    # Figure 1: spatial plot
    # -------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    
    # Shrink the markers when there are many spots
    spot_size = 30 if adata.n_obs > 5000 else 100
    
    # Left panel: ground truth
    sc.pl.spatial(adata, color='Manual Annotation', ax=axes1[0], show=False, 
                  spot_size=spot_size, title='', frameon=False)
    
    # Right panel: predictions (the refined ARI is shown as ari)
    sc.pl.spatial(adata, color='Clustering', ax=axes1[1], show=False, 
                  spot_size=spot_size, title=figure_title, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"{save_name_prefix}_Spatial.png", dpi=300)
    plt.close()

    # -------------------------------------------------------
    # Figure 2: UMAP plot
    # -------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left panel: ground truth
    sc.pl.umap(adata, color='Manual Annotation', ax=axes2[0], show=False, 
               title='', frameon=False)
    
    # Right panel: predictions
    sc.pl.umap(adata, color='Clustering', ax=axes2[1], show=False, 
               title=figure_title, frameon=False)
    
    # Aspect ratio
    for ax in axes2:
        ax.set_aspect(1)
        
    plt.tight_layout()
    plt.savefig(f"{save_name_prefix}_UMAP.png", dpi=300)
    plt.close()
    
    # print(f"  [Vis] Saved plots to {save_name_prefix}_*.png")


# def run_deg_analysis_151673(pred_labels, output_dir='./results/151673_analysis'):

    
#     # Import the custom plotting helpers
#     try:
#         from utils import plot_deg_visualizations, plot_custom_volcano, plot_custom_split_violin, plot_spatial
#     except ImportError:
#         pass 

#     print("\n[Analysis] Starting DEG/SVG Analysis for 151673...")
    
#     # ==============================================================================
#     # 1) Read the data (with histology images)
#     # ==============================================================================
#     data_root = './data/151673'
#     if not os.path.exists(os.path.join(data_root, 'spatial')):
#         print(f"!! Error: 'spatial' folder not found in {data_root}.")
#         return

#     print(f"  -> Loading Visium data (matrix + images) from {data_root}...")
#     try:
#         # This step loads the image into adata.uns['spatial']
#         adata = sc.read_visium(path=data_root, 
#                                count_file='filtered_feature_bc_matrix.h5',
#                                load_images=True) 
#     except Exception as e:
#         print(f"!! Error loading Visium data: {e}")
#         return
        
#     adata.var_names_make_unique()
#     if 'in_tissue' in adata.obs:
#         adata = adata[adata.obs['in_tissue'] == 1].copy()
    
#     # 2) Preprocessing
#     sc.pp.normalize_total(adata, target_sum=1e6)
#     sc.pp.log1p(adata)

#     # 3) Inject the labels
#     if torch.is_tensor(pred_labels):
#         pred_labels = pred_labels.detach().cpu().numpy()
        
#     if len(pred_labels) != adata.n_obs:
#         print(f"!! Warning: Length mismatch! Labels({len(pred_labels)}) vs Data({adata.n_obs})")
    
#     adata.obs['GREET_Cluster'] = pd.Categorical([f"Cluster {i}" for i in pred_labels])
#     available_clusters = adata.obs['GREET_Cluster'].unique().tolist()

#     # 4) Differential expression (details omitted)
#     print("  -> Calculating marker genes...")
#     # ... rank_genes_groups call omitted ...
#     # Minimal version for completeness:
#     sc.tl.rank_genes_groups(adata, 'GREET_Cluster', method='wilcoxon')
#     try:
#         plot_deg_visualizations(adata, output_dir)
#     except: pass

#     # ... volcano / violin comparison omitted ...

#     # 5) Save the top markers
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
#     pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10).to_csv(os.path.join(output_dir, '151673_Top_Markers.csv'))
    
#     # Image data are available once sc.read_visium succeeded
#     spatial_uns = adata.uns.get('spatial', None)
    
#     if spatial_uns is None:
#         print("!! Warning: No image data found in adata.uns['spatial']. Plot will be unaligned.")
    
#     # Call the custom plotting helper
#     # true_labels is set to pred_labels so both panels show predictions, which makes background alignment easy to check
#     my_xlim = (4000, 12000) 
#     my_ylim = (7000, 14000)

#     try:
#         plot_spatial(
#             spatial=adata.obsm['spatial'],
#             pred_labels=pred_labels,
#             true_labels=pred_labels,
#             title="GREET Clusters (Cropped)",
#             save_name=os.path.join(output_dir, '151673_Domain_Map_Cropped.png'),
#             spot_size=300,           # larger markers for the zoomed view
#             spatial_uns=spatial_uns,
#             img_key="hires",
#             xlim=my_xlim,
#             ylim=my_ylim
#         )
#         print(f"  -> Saved cropped map.")
#     except Exception as e:
#         print(f"!! plot_spatial failed: {e}")
#         # Fall back to sc.pl.spatial when plot_spatial fails
#         plt.figure(figsize=(8,8))
#         sc.pl.spatial(adata, color='GREET_Cluster', img_key='hires', show=False)
#         plt.savefig(os.path.join(output_dir, '151673_Domain_Map_Backup.png'))
#         plt.close()

#     print(f"[Done] Results saved to {output_dir}")
    
def run_deg_analysis_151673(pred_labels, output_dir='./results/151673_analysis'):
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import torch
    
    # Import the custom plotting helpers
    try:
        from utils import plot_deg_visualizations, plot_custom_volcano, plot_custom_split_violin, plot_spatial
    except ImportError:
        pass 

    print("\n[Analysis] Starting DEG/SVG Analysis for 151673...")
    
    # ==============================================================================
    # 1) Read the data (with histology images)
    # ==============================================================================
    data_root = './data/151673'
    if not os.path.exists(os.path.join(data_root, 'spatial')):
        print(f"!! Error: 'spatial' folder not found in {data_root}.")
        return

    print(f"  -> Loading Visium data (matrix + images) from {data_root}...")
    try:
        # This step loads the image into adata.uns['spatial']
        adata = sc.read_visium(path=data_root, 
                               count_file='filtered_feature_bc_matrix.h5',
                               load_images=True) 
    except Exception as e:
        print(f"!! Error loading Visium data: {e}")
        return
        
    adata.var_names_make_unique()
    if 'in_tissue' in adata.obs:
        adata = adata[adata.obs['in_tissue'] == 1].copy()
    
    # 2) Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    # 3) Inject the labels
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
        
    if len(pred_labels) != adata.n_obs:
        print(f"!! Warning: Length mismatch! Labels({len(pred_labels)}) vs Data({adata.n_obs})")
    
    adata.obs['GREET_Cluster'] = pd.Categorical([f"Cluster {i}" for i in pred_labels])
    available_clusters = adata.obs['GREET_Cluster'].unique().tolist()
    print(f"  -> Available clusters: {available_clusters}")

    # 4) Global differential expression (overview)
    print("  -> Calculating marker genes...")
    sc.tl.rank_genes_groups(adata, 'GREET_Cluster', method='wilcoxon')
    try:
        plot_deg_visualizations(adata, output_dir)
    except: pass

    # =================================================================
    # Custom volcano + split-violin comparisons
    # Cluster 4 vs 6, 4 vs 1 and 1 vs 6
    # =================================================================
    
    comparisons = [
        ('Cluster 4', 'Cluster 6'), 
        ('Cluster 4', 'Cluster 1'),
        ('Cluster 1', 'Cluster 6')
    ]

    print(f"  -> Generating requested pairwise comparisons: {comparisons}")

    for group_test, group_ref in comparisons:
        # 1) Sanity check: both clusters must exist in this run
        # (pairs are skipped automatically when a cluster is missing)
        if group_test not in available_clusters or group_ref not in available_clusters:
            print(f"     Skipping {group_test} vs {group_ref} (Cluster missing in this run)")
            continue

        print(f"     Processing {group_test} vs {group_ref}...")

        try:
            # 2) Differential expression for this pair (to pick violin genes)
            sc.tl.rank_genes_groups(adata, groupby='GREET_Cluster', 
                                    groups=[group_test], 
                                    reference=group_ref, 
                                    method='wilcoxon', 
                                    key_added='temp_deg')
            
            # 3) Pick the top six genes (three up, three down)
            df = sc.get.rank_genes_groups_df(adata, group=group_test, key='temp_deg')
            
            # Up-regulated in group_test (logFC > 0)
            top_up = df[df['logfoldchanges'] > 0].sort_values('logfoldchanges', ascending=False).head(3)['names'].tolist()
            # Up-regulated in group_reference (logFC < 0)
            top_down = df[df['logfoldchanges'] < 0].sort_values('logfoldchanges', ascending=True).head(3)['names'].tolist()
            
            violin_genes = top_up + top_down  # merged gene list

            # 4) Volcano plot
            plot_custom_volcano(adata, group_test, group_ref, output_dir)
            
            # 5) Split violin plot
            plot_custom_split_violin(adata, group_test, group_ref, violin_genes, output_dir)
            
        except Exception as e:
            print(f"     !! Failed to process {group_test} vs {group_ref}: {e}")

    # =================================================================
    # 5) Save the top markers and the spatial map
    # =================================================================
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10).to_csv(os.path.join(output_dir, '151673_Top_Markers.csv'))
    
    # Image data
    spatial_uns = adata.uns.get('spatial', None)
    if spatial_uns is None:
        print("!! Warning: No image data found. Plot will be unaligned.")
    
    # Crop window
    my_xlim = (4000, 12000) 
    my_ylim = (7000, 14000)

    try:
        print("  -> Plotting Aligned & Cropped Domain Map...")
        plot_spatial(
            spatial=adata.obsm['spatial'],
            pred_labels=pred_labels,
            true_labels=pred_labels,
            title="GREET Clusters (Cropped)",
            save_name=os.path.join(output_dir, '151673_Domain_Map_Cropped.png'),
            spot_size=300,           
            spatial_uns=spatial_uns,
            img_key="hires",
            xlim=my_xlim,
            ylim=my_ylim
        )
        print(f"  -> Saved cropped map.")
    except Exception as e:
        print(f"!! plot_spatial failed: {e}")
        # Fallback
        plt.figure(figsize=(8,8))
        sc.pl.spatial(adata, color='GREET_Cluster', img_key='hires', show=False)
        plt.savefig(os.path.join(output_dir, '151673_Domain_Map_Backup.png'))
        plt.close()

    print(f"[Done] Results saved to {output_dir}")




def plot_one_split_violin(adata, target_cluster, genes, output_dir):
    """
    Split-violin plot of cluster X versus the rest.
    """
    # 1) Check that the genes exist
    valid_genes = [g for g in genes if g in adata.var_names]
    if len(valid_genes) == 0:
        return

    # 2) Build a temporary grouping column: "Target" vs "Rest"
    # A temporary DataFrame is used so the original adata is left untouched
    obs_df = sc.get.obs_df(adata, keys=valid_genes + ['GREET_Cluster'])
    
    # hue: 'Target' for the target cluster, 'Rest' otherwise
    obs_df['Group'] = obs_df['GREET_Cluster'].apply(
        lambda x: target_cluster if x == target_cluster else 'Others'
    )

    # 3) Melt to long format for seaborn
    # Columns become: [Cell_ID, Gene_Name, Expression, Group]
    df_melted = obs_df.melt(id_vars=['GREET_Cluster', 'Group'], 
                            value_vars=valid_genes,
                            var_name='Gene', 
                            value_name='Expression')

    # 4) Plot
    plt.figure(figsize=(max(6, len(valid_genes) * 1.2), 6))  # adaptive width
    
    # (A) Split violin
    # split=True lets both groups share one violin
    # inner='quart' draws the quartile lines
    ax = sns.violinplot(data=df_melted, x='Gene', y='Expression', hue='Group',
                        split=True, inner='quart',
                        palette={target_cluster: '#1f77b4', 'Others': '#ff7f0e'},  # blue vs orange
                        cut=0)  # truncate the violin at the data range

    # (B) Stripplot with the individual spots
    # Down-sample the strip layer, otherwise large spot counts become slow and unreadable.
    if df_melted.shape[0] > 5000:
        # Draw at most 3000 random spots
        plot_data = df_melted.sample(n=3000, random_state=1)
    else:
        plot_data = df_melted
        
    sns.stripplot(data=plot_data, x='Gene', y='Expression', hue='Group',
                  dodge=True,  # separate the two groups
                  jitter=True,
                  color='black',
                  size=2,
                  alpha=0.3,
                  ax=ax, legend=False)  # no legend for the strip layer

    # Cosmetics
    plt.title(f'Split Violin: {target_cluster} vs Rest')
    plt.xlabel('Marker Genes')
    plt.ylabel('Log1p Expression')
    plt.xticks(rotation=45)
    
    # Legend placement
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save
    safe_name = target_cluster.replace(' ', '_')
    save_path = f"{output_dir}/DEG_SplitViolin_{safe_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     -> Saved Split Violin for {target_cluster}")


def plot_deg_visualizations(adata, output_dir, top_n=5):
    """
    Simplified overview plots:
    1. Matrixplot (global heatmap)
    2. Stacked violin plot
    
    Pairwise comparisons (volcano / split violin) are triggered manually from run_deg_analysis
    via plot_custom_volcano and plot_custom_split_violin.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import os

    print(f"  -> Generating global overview DEG plots in {output_dir}...")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    if 'rank_genes_groups' not in adata.uns:
        print("  !! Error: 'rank_genes_groups' not found. Run rank_genes_groups first.")
        return

    # Marker genes shown in the overview
    feature_names = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(top_n)
    marker_genes_all = feature_names.values.flatten().tolist()
    marker_genes_all = list(dict.fromkeys(marker_genes_all)) 
    marker_genes_all = [g for g in marker_genes_all if not g.startswith('MT-')]

    # -------------------------- 1. Matrixplot (global heatmap) --------------------------
    print("  -> Plotting Global Matrixplot...")
    plt.figure(figsize=(10, 8))
    # dendrogram=True adds the cluster similarity tree
    sc.pl.matrixplot(adata, marker_genes_all, groupby='GREET_Cluster', 
                     standard_scale='var', cmap='viridis', 
                     show=False, dendrogram=True)
    plt.savefig(os.path.join(output_dir, 'DEG_Global_Matrixplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------- 2. Stacked violin (overview) --------------------------
    print("  -> Plotting Global Stacked Violin Plot...")
    plt.figure(figsize=(10, 8))
    sc.pl.stacked_violin(adata, marker_genes_all[:15], groupby='GREET_Cluster', 
                         show=False)
    plt.savefig(os.path.join(output_dir, 'DEG_Global_Stacked_Violin.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("  -> Global overview plots saved.")


def plot_custom_volcano(adata, group_test, group_reference, output_dir):
    """
    Volcano plot: group_test versus group_reference.
    For example cluster 0 (layer 5) versus cluster 3 (white matter).
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    print(f"  -> Plotting Custom Volcano: {group_test} vs {group_reference}...")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1) Differential expression between the two groups
    # Wilcoxon is the usual choice
    sc.tl.rank_genes_groups(adata, groupby='GREET_Cluster', 
                            groups=[group_test], 
                            reference=group_reference, 
                            method='wilcoxon', 
                            key_added='custom_deg')

    # 2) Extract the table
    result = sc.get.rank_genes_groups_df(adata, group=group_test, key='custom_deg')
    
    # 3) Handle p-values and fold changes
    # Clip p-values to avoid log(0)
    result['log10p'] = -np.log10(result['pvals_adj'] + 1e-50)
    
    # Flag up- and down-regulated genes
    # Thresholds are configurable, e.g. log2FC > 1 and P < 0.01
    result['diff'] = 'normal'
    result.loc[(result['logfoldchanges'] > 1) & (result['pvals_adj'] < 0.05), 'diff'] = 'up'
    result.loc[(result['logfoldchanges'] < -1) & (result['pvals_adj'] < 0.05), 'diff'] = 'down'

    # 4) Plot
    plt.figure(figsize=(8, 6))
    
    # Background points
    sns.scatterplot(data=result[result['diff']=='normal'], x='logfoldchanges', y='log10p', 
                    color='lightgrey', s=10, alpha=0.5, edgecolor=None)
    
    # Significant up- and down-regulated points
    # up: higher in group_test
    # down: higher in group_reference
    sns.scatterplot(data=result[result['diff']!='normal'], x='logfoldchanges', y='log10p', 
                    hue='diff', palette={'up': '#d62728', 'down': '#1f77b4'},
                    s=30, alpha=0.8, edgecolor=None)

    # 5) Annotate the most significant genes
    # Top five up- and five down-regulated genes
    top_up = result[result['diff']=='up'].sort_values('log10p', ascending=False).head(5)
    top_down = result[result['diff']=='down'].sort_values('log10p', ascending=False).head(5)
    
    import adjustText  # plt.text is a simple substitute when unavailable
    texts = []
    for idx, row in top_up.iterrows():
        texts.append(plt.text(row['logfoldchanges'], row['log10p'], row['names'], fontsize=9, color='darkred'))
    for idx, row in top_down.iterrows():
        texts.append(plt.text(row['logfoldchanges'], row['log10p'], row['names'], fontsize=9, color='darkblue'))
    
    # Avoid label overlap (skip this block without adjustText)
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    except:
        pass  # without adjustText labels may overlap

    # 6) Cosmetics
    plt.title(f'Volcano: {group_test} (Red) vs {group_reference} (Blue)')
    plt.xlabel(f'log2 Fold Change\n(>0: High in {group_test}, <0: High in {group_reference})')
    plt.ylabel('-log10(adj. p-value)')
    plt.axvline(1, ls='--', c='k', lw=0.5); plt.axvline(-1, ls='--', c='k', lw=0.5)
    plt.axhline(1.3, ls='--', c='k', lw=0.5) # p=0.05
    
    plt.xlim(-5, 5)

    # Fit the Y axis to the visible points
    # so the panel does not look empty
    current_view_genes = result[
        (result['logfoldchanges'] >= -5) & 
        (result['logfoldchanges'] <= 5)
    ]
    if len(current_view_genes) > 0:
        max_y_in_view = np.max(current_view_genes['log10p'].replace([np.inf, -np.inf], 0))
        plt.ylim(0, max_y_in_view * 1.1)
    

    # Save
    save_name = f'Volcano_{group_test}_vs_{group_reference}.png'.replace(' ', '_')
    plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     -> Saved to {save_name}")


from scipy.optimize import linear_sum_assignment

def align_labels(pred_labels, true_labels):
    """
    Reorder pred_labels with the Hungarian algorithm so they best match true_labels.
    The same anatomical domain then receives the same cluster id across runs.
    """
    # Ensure numpy arrays
    if torch.is_tensor(pred_labels): pred_labels = pred_labels.detach().cpu().numpy()
    if torch.is_tensor(true_labels): true_labels = true_labels.detach().cpu().numpy()

    # Drop invalid labels such as -1
    mask = (true_labels >= 0) & (pred_labels >= 0)
    y_true = true_labels[mask]
    y_pred = pred_labels[mask]
    
    if len(y_true) == 0:
        return pred_labels

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # Overlap matrix
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
        
    # Hungarian assignment maximising the total overlap
    # linear_sum_assignment minimises the cost, hence the negative weights
    row_ind, col_ind = linear_sum_assignment(-w)
    
    # row_ind: original prediction, col_ind: target ground-truth label
    # Build the mapping
    reorder_map = {row: col for row, col in zip(row_ind, col_ind)}
    
    # Unmatched labels keep their original id
    new_pred_labels = np.zeros_like(pred_labels)
    for i in range(len(pred_labels)):
        old_lbl = pred_labels[i]
        if old_lbl in reorder_map:
            new_pred_labels[i] = reorder_map[old_lbl]
        else:
            new_pred_labels[i] = old_lbl
            
    return new_pred_labels

def plot_custom_split_violin(adata, group_test, group_reference, genes, output_dir):
    """
    Split violin for two specific clusters (for example cluster 0 versus cluster 3).
    Left: group_test (blue).
    Right: group_reference (orange).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    import os

    # 1) Check the genes
    valid_genes = [g for g in genes if g in adata.var_names]
    if len(valid_genes) == 0:
        print(f"  !! No valid genes found for split violin ({group_test} vs {group_reference})")
        return

    print(f"  -> Plotting Split Violin: {group_test} vs {group_reference}...")

    # 2) Restrict to the two clusters
    # Keep only spots of the two groups
    subset_mask = adata.obs['GREET_Cluster'].isin([group_test, group_reference])
    subset_adata = adata[subset_mask].copy()

    # 3) Prepare the plotting table
    # Expression matrix
    obs_df = sc.get.obs_df(subset_adata, keys=valid_genes + ['GREET_Cluster'])
    
    # Melt to long format for seaborn
    df_melted = obs_df.melt(id_vars=['GREET_Cluster'], 
                            value_vars=valid_genes,
                            var_name='Gene', 
                            value_name='Expression')

    # 4) Plot
    plt.figure(figsize=(max(6, len(valid_genes)*1.2), 6))
    
    # Explicit palette so colours stay stable across runs
    # (red vs blue, matching the volcano plot)
    my_palette = {group_test: '#d62728', group_reference: '#1f77b4'}  # red vs blue
    
    # hue_order fixes the left/right placement
    ax = sns.violinplot(data=df_melted, x='Gene', y='Expression', hue='GREET_Cluster',
                        hue_order=[group_test, group_reference],
                        split=True, inner='quart',
                        palette=my_palette,
                        cut=0)  # truncate at the data range

    # Optional strip layer, down-sampled for large inputs
    if df_melted.shape[0] > 3000:
        plot_data = df_melted.sample(n=3000, random_state=1)
    else:
        plot_data = df_melted
        
    sns.stripplot(data=plot_data, x='Gene', y='Expression', hue='GREET_Cluster',
                  hue_order=[group_test, group_reference],
                  dodge=True, jitter=True, color='black', size=2, alpha=0.3,
                  ax=ax, legend=False)

    # 5) Cosmetics
    plt.title(f'Split Violin: {group_test} (Red) vs {group_reference} (Blue)')
    plt.xlabel('Marker Genes')
    plt.ylabel('Log1p Expression')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster')
    plt.tight_layout()

    # Save
    save_name = f"SplitViolin_{group_test}_vs_{group_reference}.png".replace(' ', '_')
    plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()

    import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _to_1d_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().flatten().cpu().numpy()
    return np.asarray(x).reshape(-1)
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_edge_similarity_histogram(
    embedding: torch.Tensor,     # (N, d) on GPU/CPU
    edges: torch.Tensor,         # (2, E) on same device as embedding
    weights_lp: torch.Tensor = None,  # (E,)
    weights_hp: torch.Tensor = None,  # (E,)
    save_path: str = "./results/edge_sim_hist.png",
    bins: int = 60,
    max_edges: int = 200000,
    seed: int = 0,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    E = edges.shape[1]
    if E > max_edges:
        g = torch.Generator(device=edges.device)
        g.manual_seed(seed)
        idx = torch.randperm(E, generator=g, device=edges.device)[:max_edges]
        e = edges[:, idx]
        wlp = weights_lp[idx] if weights_lp is not None else None
        whp = weights_hp[idx] if weights_hp is not None else None
    else:
        e, wlp, whp = edges, weights_lp, weights_hp

    u, v = e[0], e[1]
    sim_all = F.cosine_similarity(embedding[u], embedding[v], dim=1).detach().cpu().numpy()

    # Optional random-negative reference distribution
    ru = torch.randint(0, embedding.shape[0], (u.numel(),), device=embedding.device)
    rv = torch.randint(0, embedding.shape[0], (u.numel(),), device=embedding.device)
    sim_rand = F.cosine_similarity(embedding[ru], embedding[rv], dim=1).detach().cpu().numpy()

    plt.figure()
    plt.hist(sim_all, bins=bins, alpha=0.6, label="edges")
    plt.hist(sim_rand, bins=bins, alpha=0.6, label="random pairs")

    # Optional split by LP / HP weights
    if wlp is not None:
        m = (wlp > 0.5).detach().cpu().numpy()
        if m.any():
            plt.hist(sim_all[m], bins=bins, alpha=0.6, label="LP edges (w_lp>0.5)")
    if whp is not None:
        m = (whp > 0.5).detach().cpu().numpy()
        if m.any():
            plt.hist(sim_all[m], bins=bins, alpha=0.6, label="HP edges (w_hp>0.5)")

    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_polynomial_filter_shapes(
    k: int,
    alpha_hp: float,
    coef_logits: torch.Tensor = None,   # cl_model.coef_logits
    save_path: str = "./results/poly_filter.png",
    num_points: int = 400,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lam = np.linspace(-1.0, 1.0, num_points)  # normalised adjacency eigenvalues lie in [-1, 1]
    g_lp = lam ** k
    g_hp = (1.0 - alpha_hp * lam) ** k

    plt.figure()
    plt.plot(lam, g_lp, label=f"LP: λ^{k}")
    plt.plot(lam, g_hp, label=f"HP: (1-α·λ)^{k}, α={alpha_hp}")

    if coef_logits is not None:
        coef = torch.softmax(coef_logits.detach().cpu(), dim=0).numpy()  # :contentReference[oaicite:13]{index=13}
        g_mix = coef[0] * g_lp + coef[1] * g_hp
        plt.plot(lam, g_mix, label=f"mix: {coef[0]:.2f}·LP + {coef[1]:.2f}·HP")

    plt.xlabel("eigenvalue λ (normalized adjacency)")
    plt.ylabel("filter response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
