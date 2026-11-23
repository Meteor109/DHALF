import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random
import os
import json
import scanpy as sc
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
from data_loader import load_data, load_visium_spatial_uns
from model import *
from utils import *

EOS = 1e-10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    if os.environ.get("SHARP_SEED_DGL", "0") == "1":
        try:
            dgl.seed(seed)
            dgl.random.seed(seed)
        except Exception as e:
            # Variant D addition: keep Python/NumPy/Torch seeding strict even if DGL's
            # CUDA CURAND state cannot initialize in a fresh conda-run process.
            print(f"[WARN] DGL seed skipped: {type(e).__name__}")




def get_variant_d_scales(epoch, args):
    warmup = int(getattr(args, "variant_d_warmup", 0))
    if warmup <= 0:
        return 1.0, 1.0
    fl_scale = min(1.0, max(0.0, float(epoch) / float(warmup)))
    fh_scale = min(1.0, max(0.0, float(epoch - warmup) / float(warmup)))
    return fl_scale, fh_scale


def _build_mask_from_mode(mode, edges, weights_lp, features, nnodes, args, semantic_pos_mask):
    """Resolve a positive-pair mask for a Full-Low / Full-High term.

    `mode` selects the positive-pair rule only; the encoder that consumes the mask
    is unchanged, so the four ablation rows differ in this rule and nothing else.
    """
    if mode == "semantic":
        if semantic_pos_mask is None:
            semantic_pos_mask = build_variant_d_semantic_mask(features.detach().cpu(), k=args.sem_k)
        return semantic_pos_mask.to(features.device)
    if mode == "spatial_all":
        return build_variant_d_spatial_mask(edges, nnodes, polarity="all")
    if mode == "spatial_heterophilic":
        return build_variant_d_spatial_mask(
            edges, nnodes, weights=weights_lp, threshold=0.5, polarity="heterophilic"
        )
    if mode == "spatial_homophilic":
        return build_variant_d_spatial_mask(
            edges, nnodes, weights=weights_lp, threshold=0.5, polarity="homophilic"
        )
    raise ValueError(f"Unsupported mask mode: {mode}")


def train_model(cl_model, discriminator, optimizer_cl, features, str_encodings, edges, args,
                semantic_pos_mask=None, epoch=1):
    cl_model.train()
    discriminator.eval()

    adj_1, adj_2, weights_lp, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
    base_loss = cl_model(features, adj_1, features, adj_2)
    no_aux = bool(getattr(args, "no_aux_branch", 0))
    fl_mode = getattr(args, "fl_mask_mode", None) or ("spatial_homophilic" if args.use_hom_pos else "spatial_all")
    fh_mode = getattr(args, "fh_mask_mode", None) or "semantic"
    fl_pos_mask = _build_mask_from_mode(
        fl_mode, edges, weights_lp, features, features.shape[0], args, semantic_pos_mask
    )
    fh_pos_mask = _build_mask_from_mode(
        fh_mode, edges, weights_lp, features, features.shape[0], args, semantic_pos_mask
    )

    if no_aux:
        l_fl = torch.zeros((), device=features.device)
        l_fh = torch.zeros((), device=features.device)
        emb_log = cl_model.get_variant_d_embeddings(features, adj_1, adj_2)
    else:
        l_fl, l_fh, emb_log = cl_model.variant_d_losses(features, adj_1, adj_2, fl_pos_mask, fh_pos_mask, tau=0.5)

    fl_scale, fh_scale = get_variant_d_scales(epoch, args)
    eff_lambda_fl = 0.0 if no_aux else args.lambda_fl * fl_scale
    eff_lambda_fh = 0.0 if no_aux else args.lambda_fh * fh_scale
    lambda_mix = float(getattr(args, "lambda_mix", 0.0))
    if lambda_mix > 0:
        mix_anchor = cl_model.variant_d_branch_consensus(emb_log) if no_aux else None
        l_mix = cl_model.variant_d_mix_loss(
            emb_log, fh_pos_mask, tau=0.5,
            detach_branches=bool(getattr(args, "mix_detach_branches", 1)),
            anchor=mix_anchor,
        )
        eff_lambda_mix = lambda_mix * fl_scale
    else:
        l_mix = torch.zeros((), device=features.device)
        eff_lambda_mix = 0.0
    model_loss = (args.lambda_base * base_loss + eff_lambda_fl * l_fl
                  + eff_lambda_fh * l_fh + eff_lambda_mix * l_mix)
    gate_detached = emb_log["gate"].detach()
    cl_model.last_variant_d_log = {
        "base_loss": float(base_loss.detach().cpu()),
        "l_fl": float(l_fl.detach().cpu()),
        "l_fh": float(l_fh.detach().cpu()),
        "l_mix": float(l_mix.detach().cpu()),
        "eff_lambda_fl": float(eff_lambda_fl),
        "eff_lambda_fh": float(eff_lambda_fh),
        "eff_lambda_mix": float(eff_lambda_mix),
        "gate_mean": float(gate_detached.mean().cpu()),
        "gate_std": float(gate_detached.std().cpu()),
        "model_total": float(model_loss.detach().cpu()),
    }

    optimizer_cl.zero_grad()
    model_loss.backward()
    optimizer_cl.step()
    return model_loss.item()



def train_discriminator(cl_model, discriminator, optimizer_disc, features, str_encodings, edges, args):
    cl_model.eval()
    discriminator.train()

    # 1) Forward pass (shared part)
    adj_1, adj_2, weights_lp, weights_hp = discriminator(torch.cat((features, str_encodings), 1), edges)
    embedding = cl_model.get_embedding(features, adj_1, adj_2)

    # 2) Prepare negative samples
    rand_np = generate_random_node_pairs(features.shape[0], edges.shape[1])
    psu_label = torch.ones(edges.shape[1]).cuda()

    # 3) Compute the loss in batches
    edge_batch_size = 5000  # kept small on purpose to bound memory
    num_edges = edges.shape[1]
    
    # Accumulator for the total loss
    total_loss = 0.0
    
    num_batches = (num_edges + edge_batch_size - 1) // edge_batch_size

    optimizer_disc.zero_grad()

    for i in range(num_batches):
        start_idx = i * edge_batch_size
        end_idx = min((i + 1) * edge_batch_size, num_edges)
        
        # Slice the current batch
        batch_edges = edges[:, start_idx:end_idx]
        batch_rand_np = rand_np[:, start_idx:end_idx]
        
        # Corresponding embedding slices
        emb_u = embedding[batch_edges[0]]
        emb_v = embedding[batch_edges[1]]
        emb_ru = embedding[batch_rand_np[0]]
        emb_rv = embedding[batch_rand_np[1]]

        # Cosine similarity
        batch_edge_sim = F.cosine_similarity(emb_u, emb_v)
        batch_rnp_sim_lp = F.cosine_similarity(emb_ru, emb_rv)
        
        # Corresponding edge weights
        batch_weights_lp = weights_lp[start_idx:end_idx]
        batch_weights_hp = weights_hp[start_idx:end_idx]
        batch_psu_label = psu_label[start_idx:end_idx]

        # Loss of the current batch
        loss_lp = F.margin_ranking_loss(batch_edge_sim, batch_rnp_sim_lp, batch_psu_label, margin=args.margin_hom, reduction='none')
        loss_lp = (loss_lp * torch.relu(batch_weights_lp - 0.5)).mean()

        loss_hp = F.margin_ranking_loss(batch_rnp_sim_lp, batch_edge_sim, batch_psu_label, margin=args.margin_het, reduction='none')
        loss_hp = (loss_hp * torch.relu(batch_weights_hp - 0.5)).mean()

        batch_loss = (loss_lp + loss_hp) / 2
        
        # Accumulate the loss here; do not call backward inside the loop
        total_loss = total_loss + batch_loss

    # 4) Average and run a single backward pass
    final_loss = total_loss / num_batches
    final_loss.backward()
    
    optimizer_disc.step()

    return final_loss.item()


def build_variant_d_command(args):
    # Variant D addition: reconstruct a concise runnable command for summaries.
    parts = [
        "python main.py",
        f"-dataset '{args.dataset}'",
        f"-margin_hom {args.margin_hom}",
        f"-margin_het {args.margin_het}",
        f"-k {args.k}",
        f"-refine_k {args.refine_k}",
        f"-epochs {args.epochs}",
        f"-eval_freq {args.eval_freq}",
        f"-gpu {args.gpu}",
        f"--seed {args.seed}",
        f"--lambda_fl {args.lambda_fl}",
        f"--lambda_fh {args.lambda_fh}",
        f"--lambda_base {args.lambda_base}",
        f"--lambda_mix {args.lambda_mix}",
        f"--mix_detach_branches {args.mix_detach_branches}",
        f"--sem_k {args.sem_k}",
        f"--variant_d_warmup {args.variant_d_warmup}",
        f"--gate_hidden {args.gate_hidden}",
        f"--variant_d_gate_mode {args.variant_d_gate_mode}",
        f"--full_mlp_hidden {args.full_mlp_hidden}",
        f"--full_mlp_out {args.full_mlp_out}",
        f"--use_hom_pos {args.use_hom_pos}",
        f"--fl_mask_mode {getattr(args, 'fl_mask_mode', None) or ('spatial_homophilic' if args.use_hom_pos else 'spatial_all')}",
        f"--fh_mask_mode {getattr(args, 'fh_mask_mode', None) or 'semantic'}",
        f"--no_aux_branch {int(getattr(args, 'no_aux_branch', 0))}",
        f"--plot_point_size {args.plot_point_size}",
        f"--eval_aux_embeddings {args.eval_aux_embeddings}",
        f"--save_diagnostics {args.save_diagnostics}",
    ]
    if getattr(args, "data_path", None):
        parts.append(f"--data_path '{args.data_path}'")
    if getattr(args, "result_name", None):
        parts.append(f"--result_name '{args.result_name}'")
    if getattr(args, "n_clusters_override", 0):
        parts.append(f"--n_clusters_override {args.n_clusters_override}")
    if getattr(args, "ablation_name", ""):
        parts.append(f"--ablation_name '{args.ablation_name}'")
    return " ".join(parts)


def format_dataset_for_plot(dataset):
    # Variant D addition: compact title for saved result figures.
    dataset = str(dataset)
    if dataset.isdigit():
        return f"DLPFC{dataset}"
    return dataset


def result_name(args):
    return str(getattr(args, "result_name", None) or args.dataset)


def resolve_h5ad_path(args, default_path):
    return str(getattr(args, "data_path", None) or default_path)


def _mob_subset_indices(adata, expected_nnodes=None, max_per_section=1000, seed=0):
    if expected_nnodes is not None and expected_nnodes <= 0:
        return np.array([], dtype=int)
    if 'section' not in adata.obs.columns:
        n = expected_nnodes or min(int(max_per_section) * 8, adata.n_obs)
        return np.sort(np.random.default_rng(seed).choice(adata.n_obs, size=min(n, adata.n_obs), replace=False))

    rng = np.random.default_rng(seed)
    sections = adata.obs['section'].astype(str).to_numpy()
    keep = []
    for section in sorted(np.unique(sections), key=lambda x: str(x)):
        idx = np.where(sections == section)[0]
        take = min(int(max_per_section), len(idx))
        keep.append(rng.choice(idx, size=take, replace=False))
    selected = np.sort(np.concatenate(keep))
    if expected_nnodes is not None and len(selected) != expected_nnodes:
        selected = selected[:expected_nnodes]
    return selected


def _brca_xenium_subset_indices(adata, expected_nnodes=None, max_cells=20000, seed=0):
    n_select = int(expected_nnodes or min(int(max_cells), adata.n_obs))
    if adata.n_obs <= n_select:
        return np.arange(adata.n_obs)
    return np.sort(np.random.default_rng(seed).choice(adata.n_obs, size=n_select, replace=False))


def _cns_subset_indices(adata, max_nodes=8000, seed=0):
    n_select = min(int(max_nodes), adata.n_obs)
    if n_select == adata.n_obs:
        return np.arange(adata.n_obs, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(adata.n_obs, size=n_select, replace=False))


def compute_ari_score(pred_labels, labels):
    true_Y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    valid_mask = true_Y >= 0
    if np.sum(valid_mask) > 0:
        return adjusted_rand_score(true_Y[valid_mask], pred_labels[valid_mask])
    return 0.0


def has_ground_truth_labels(labels):
    true_Y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    return bool(np.sum(true_Y >= 0) > 0)


def evaluate_variant_d_embedding(embedding, labels, n_clusters, spatial, refine_k, edges=None):
    # Variant D addition: evaluate one candidate representation before and after spatial refinement.
    has_labels = has_ground_truth_labels(labels)
    pred_labels = None
    try:
        pred_labels = mclust_R(embedding, n_clusters)
    except Exception as e:
        print(f"Wrapper failed: {e}")

    if pred_labels is not None and labels is not None and has_labels:
        pred_labels = align_labels(pred_labels, labels)

    if pred_labels is None:
        raw_ari, pred_labels = eval_clustering(embedding, labels, n_clusters)
    else:
        raw_ari = compute_ari_score(pred_labels, labels) if has_labels else np.nan

    if spatial is not None and refine_k > 0:
        pred_labels_refined = refine_label_spatial(pred_labels, spatial, k=refine_k)
        refined_ari = compute_ari_score(pred_labels_refined, labels) if has_labels else np.nan
    else:
        pred_labels_refined = pred_labels
        refined_ari = raw_ari

    boundary = {"f1": np.nan, "precision": np.nan, "recall": np.nan}
    if edges is not None and has_labels:
        boundary = spatial_boundary_f1(pred_labels_refined, labels, edges)

    return {
        "raw_ari": raw_ari,
        "refined_ari": refined_ari,
        "boundary_f1": boundary["f1"],
        "boundary_precision": boundary["precision"],
        "boundary_recall": boundary["recall"],
        "has_ground_truth": has_labels,
        "raw_labels": pred_labels,
        "refined_labels": pred_labels_refined,
    }


def _read_visium_positions(path):
    pos_candidates = [
        os.path.join(path, "spatial", "tissue_positions_list.csv"),
        os.path.join(path, "spatial", "tissue_positions.csv"),
        os.path.join(path, "tissue_positions_list.csv"),
        os.path.join(path, "tissue_positions.csv"),
    ]
    pos_path = next((p for p in pos_candidates if os.path.exists(p)), None)
    if pos_path is None:
        raise FileNotFoundError(f"No tissue_positions file found under {path}")

    positions = pd.read_csv(pos_path, header=None, sep=r"[\t,\s]+", engine="python")
    if str(positions.iloc[0, 0]).lower() == "barcode":
        positions = positions.iloc[1:].copy()
    positions = positions.iloc[:, :6].copy()
    positions.columns = [
        "spot_id", "in_tissue", "array_row", "array_col",
        "pxl_row_in_fullres", "pxl_col_in_fullres",
    ]
    positions["spot_id"] = positions["spot_id"].astype(str)
    for col in ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
        positions[col] = pd.to_numeric(positions[col], errors="coerce")
    return positions


def _align_index_suffix(index, target_index):
    index = pd.Index(index.astype(str))
    target_index = pd.Index(target_index.astype(str))
    if len(index) == 0 or len(target_index) == 0:
        return index
    if target_index[0].endswith("-1") and not index[0].endswith("-1"):
        return index + "-1"
    if index[0].endswith("-1") and not target_index[0].endswith("-1"):
        return index.str.replace("-1$", "", regex=True)
    return index


def _best_artifact_tables_from_visium(args, expected_nnodes):
    dataset_paths = {
        "151673": "./data/151673",
        "151674": "./data/151674",
        "BRCA": "./data/BRCA/Breast_Caner",
        "Breast_Caner": "./data/BRCA/Breast_Caner",
    }
    data_dir = dataset_paths.get(str(args.dataset))
    if data_dir is None:
        return None
    h5_path = os.path.join(data_dir, "filtered_feature_bc_matrix.h5")
    if not os.path.exists(h5_path):
        return None

    adata_raw = sc.read_10x_h5(h5_path)
    adata_raw.var_names_make_unique()
    positions = _read_visium_positions(data_dir)
    positions_idx = positions.set_index("spot_id")
    positions_idx.index = _align_index_suffix(positions_idx.index, adata_raw.obs_names)
    adata_raw.obs = adata_raw.obs.join(positions_idx, how="left")
    adata_tissue = adata_raw[adata_raw.obs["in_tissue"].astype(float) == 1].copy()

    ground_truth = pd.Series("Unknown", index=adata_tissue.obs_names, dtype="object")
    if str(args.dataset) == "151673":
        meta_path = os.path.join(data_dir, "metadata.tsv")
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, sep="\t", index_col=0)
            meta.index = _align_index_suffix(meta.index, adata_tissue.obs_names)
            common = adata_tissue.obs_names.intersection(meta.index)
            adata_tissue = adata_tissue[common].copy()
            if "layer_guess" in meta.columns:
                ground_truth = meta.loc[common, "layer_guess"].astype("string").fillna("Unknown")
            else:
                ground_truth = pd.Series("Unknown", index=common, dtype="object")
    elif str(args.dataset) == "151674":
        label_path = os.path.join(data_dir, "cluster_labels_151674.csv")
        if os.path.exists(label_path):
            lab = pd.read_csv(label_path)
            if "key" in lab.columns and "ground_truth" in lab.columns:
                lab["spot_id"] = lab["key"].astype(str).str.replace(r"^151674_", "", regex=True)
                lab = lab.set_index("spot_id")
                lab.index = _align_index_suffix(lab.index, adata_tissue.obs_names)
                common = adata_tissue.obs_names.intersection(lab.index)
                adata_tissue = adata_tissue[common].copy()
                ground_truth = lab.loc[common, "ground_truth"].astype("string").fillna("Unknown")
        else:
            meta_path = os.path.join(data_dir, "metadata.tsv")
            if os.path.exists(meta_path):
                meta = pd.read_csv(meta_path, sep="\t", index_col=0)
                meta.index = _align_index_suffix(meta.index, adata_tissue.obs_names)
                common = adata_tissue.obs_names.intersection(meta.index)
                adata_tissue = adata_tissue[common].copy()
                label_col = "layer_guess" if "layer_guess" in meta.columns else meta.columns[-1]
                ground_truth = meta.loc[common, label_col].astype("string").fillna("Unknown")
    elif str(args.dataset) in {"BRCA", "Breast_Caner"}:
        ann_path = os.path.join(data_dir, "spatial", "brca_ground_truth_annotation.csv")
        if os.path.exists(ann_path):
            ann = pd.read_csv(ann_path)
            if "spot_id" in ann.columns and "ground_truth_annotation" in ann.columns:
                ann = ann.set_index("spot_id")
                ann.index = _align_index_suffix(ann.index, adata_tissue.obs_names)
                common = adata_tissue.obs_names.intersection(ann.index)
                adata_tissue = adata_tissue[common].copy()
                ground_truth = ann.loc[common, "ground_truth_annotation"].astype("string").fillna("Unknown")

    if adata_tissue.n_obs != expected_nnodes:
        raise ValueError(
            f"Best artifact spot count mismatch for {args.dataset}: "
            f"{adata_tissue.n_obs} metadata rows vs {expected_nnodes} model nodes"
        )

    spot_table = pd.DataFrame({
        "spot_id": adata_tissue.obs_names.astype(str),
        "x": adata_tissue.obs["pxl_col_in_fullres"].astype(float).to_numpy(),
        "y": adata_tissue.obs["pxl_row_in_fullres"].astype(float).to_numpy(),
        "ground_truth_annotation": np.asarray(ground_truth, dtype=object),
    })
    all_spots = positions.rename(columns={
        "pxl_col_in_fullres": "x",
        "pxl_row_in_fullres": "y",
    })[["spot_id", "x", "y"]].copy()
    return spot_table, all_spots, adata_tissue


def _best_artifact_tables_from_h5ad(args, expected_nnodes):
    h5ad_paths = {
        "E18_mouse_brain": "./data/E18.5_mouse_brain/E18_adata_rna.h5ad",
        "Stereo_AB": "./data/Stereo/Stereo_AB.h5ad",
        "MOB": "./data/MOB/MOB_3D.h5ad",
        "Stereo_mouse_brain": "./data/Stereo/Stereo_mouse_brain.h5ad",
        "Stereo_DPI": "./data/Stereo/Stereo_DPI.h5ad",
        "DPI": "./data/DPI/2DPI_1.h5ad",
        "CNS": "./data/CNS/CNS3.h5ad",
        "BRCA_Xexium": "./data/BRCA/BRCA_Xexium.h5ad",
        "mouse_embryo": "./data/mouse_embryo/s1_adata_rna.h5ad",
    }
    path = resolve_h5ad_path(args, h5ad_paths.get(str(args.dataset), ""))
    if path is None or not os.path.exists(path):
        return None
    adata_raw = sc.read_h5ad(path)
    adata_raw.var_names_make_unique()
    if str(args.dataset) == "MOB_section3":
        if "section" not in adata_raw.obs.columns:
            raise ValueError("MOB_section3 requires obs['section'].")
        adata_raw = adata_raw[adata_raw.obs["section"].astype(str).to_numpy() == "3"].copy()
    elif str(args.dataset) == "MOB" and adata_raw.n_obs != expected_nnodes:
        adata_raw = adata_raw[_mob_subset_indices(adata_raw, expected_nnodes)].copy()
    elif str(args.dataset) == "BRCA_Xexium":
        adata_raw = adata_raw[_brca_xenium_subset_indices(adata_raw, expected_nnodes)].copy()
    elif str(args.dataset) == "CNS":
        adata_raw = adata_raw[_cns_subset_indices(adata_raw, max_nodes=expected_nnodes, seed=args.seed)].copy()
    if adata_raw.n_obs != expected_nnodes:
        adata_raw = adata_raw[:expected_nnodes].copy()
    if "spatial" in adata_raw.obsm:
        coords = np.asarray(adata_raw.obsm["spatial"], dtype=float)
    elif {"x", "y"}.issubset(adata_raw.obs.columns):
        coords = adata_raw.obs[["x", "y"]].astype(float).to_numpy()
    elif {"array_col", "array_row"}.issubset(adata_raw.obs.columns):
        coords = adata_raw.obs[["array_col", "array_row"]].astype(float).to_numpy()
    else:
        coords = np.column_stack([np.arange(adata_raw.n_obs), np.zeros(adata_raw.n_obs)])

    label_col = next((c for c in ["ground_truth", "Annotation", "annotation", "layer_guess", "cell_type", "clusters", "Main_molecular_tissue_region", "section"] if c in adata_raw.obs.columns), None)
    if str(args.dataset) == "MOB_section3":
        ground_truth = np.repeat("Unknown", adata_raw.n_obs)
    else:
        ground_truth = adata_raw.obs[label_col].astype("string").fillna("Unknown").to_numpy() if label_col else np.repeat("Unknown", adata_raw.n_obs)
    spot_table = pd.DataFrame({
        "spot_id": adata_raw.obs_names.astype(str),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "ground_truth_annotation": ground_truth,
    })
    return spot_table, None, adata_raw


def _resolve_best_artifact_tables(args, expected_nnodes):
    for resolver in (_best_artifact_tables_from_visium, _best_artifact_tables_from_h5ad):
        resolved = resolver(args, expected_nnodes)
        if resolved is not None:
            return resolved
    return None, None, None


def save_best_downstream_artifacts(args, pred_labels, embedding, labels, refined_ari, repr_name):
    artifact_dir = os.path.join("./result", result_name(args))
    os.makedirs(artifact_dir, exist_ok=True)
    pred_np = pred_labels.detach().cpu().numpy() if torch.is_tensor(pred_labels) else np.asarray(pred_labels)
    emb_np = embedding.detach().cpu().numpy() if torch.is_tensor(embedding) else np.asarray(embedding)
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    spot_table, all_spots, adata_raw = _resolve_best_artifact_tables(args, len(pred_np))
    if spot_table is None:
        if emb_np.shape[0] != len(pred_np):
            raise ValueError("Cannot build fallback artifact table because embedding and labels have different node counts.")
        coords = np.column_stack([np.arange(len(pred_np)), np.zeros(len(pred_np))])
        spot_table = pd.DataFrame({
            "spot_id": [f"spot_{i}" for i in range(len(pred_np))],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "ground_truth_annotation": [
                "Unknown" if label < 0 else str(label) for label in labels_np
            ],
        })
        print(f"[WARN] Falling back to synthetic spot ids for downstream artifacts: {args.dataset}")

    cluster_df = spot_table[["spot_id", "x", "y"]].copy()
    cluster_df.insert(1, "cluster", pred_np.astype(int))
    cluster_path = os.path.join(artifact_dir, "cluster_labels.csv")
    cluster_df.to_csv(cluster_path, index=False)

    gt_path = os.path.join(artifact_dir, "ground_truth_annotations.csv")
    spot_table[["spot_id", "x", "y", "ground_truth_annotation"]].to_csv(gt_path, index=False)

    emb_path = os.path.join(artifact_dir, f"best_{repr_name}_embedding.npy")
    np.save(emb_path, emb_np)
    emb_index_path = os.path.join(artifact_dir, f"best_{repr_name}_embedding_spot_ids.csv")
    pd.DataFrame({"spot_id": spot_table["spot_id"].astype(str)}).to_csv(emb_index_path, index=False)

    if all_spots is not None:
        all_spots_path = os.path.join(artifact_dir, "all_spots.csv")
        all_spots.to_csv(all_spots_path, index=False)
    else:
        all_spots_path = None

    expr_path = None
    if adata_raw is not None:
        expr_path = os.path.join(artifact_dir, "raw_gene_expression.h5ad")
        adata_raw.write_h5ad(expr_path)

    manifest_path = os.path.join(artifact_dir, "best_artifacts_manifest.json")
    manifest = {
        "dataset": str(args.dataset),
        "result_name": result_name(args),
        "seed": int(args.seed),
        "has_ground_truth": has_ground_truth_labels(labels),
        "best_representation": str(repr_name),
        "cluster_labels": cluster_path,
        "ground_truth_annotations": gt_path,
        "raw_gene_expression": expr_path,
        "best_embedding": emb_path,
        "best_embedding_spot_ids": emb_index_path,
        "all_spots": all_spots_path,
    }
    if manifest["has_ground_truth"]:
        manifest["best_refined_ari"] = float(refined_ari)
    if getattr(args, "n_clusters_override", 0):
        manifest["n_clusters_override"] = int(args.n_clusters_override)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Artifacts] Saved downstream best artifacts to {artifact_dir}")
    return manifest


def cleanup_superseded_visualization(previous_plot_path, current_plot_path):
    if not previous_plot_path or previous_plot_path == "NA":
        return
    if previous_plot_path == current_plot_path:
        return

    paths = [previous_plot_path]
    if previous_plot_path.endswith(".png"):
        paths.append(previous_plot_path.replace(".png", "_UMAP.png"))

    for path in paths:
        if path == current_plot_path:
            continue
        if os.path.exists(path):
            os.remove(path)
            print(f"[Cleanup] Removed superseded visualization: {path}")


def save_variant_d_best(cl_model, discriminator, args, trial, epoch, repr_name, raw_ari, refined_ari,
                        pred_labels, embedding, spatial, labels, class_names,
                        spatial_uns, library_id, img_key, all_spatial_plot):
    # Save best figures and downstream artifacts under the dataset-specific result directory.
    run_name = result_name(args)
    result_dir = os.path.join("./result", run_name)
    ckpt_dir = os.path.join("./result", "checkpoints")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    safe_repr = repr_name.replace("+", "plus")
    has_labels = has_ground_truth_labels(labels)
    # Variant D addition: keep the default filename shape, and disambiguate
    # refine_k sensitivity runs when they use a non-standard refinement radius.
    refine_suffix = f"_refinek_{args.refine_k}" if args.refine_k != 25 else ""
    if has_labels:
        plot_name = f"{run_name}_variantD_{safe_repr}_best_ari_{refined_ari:.4f}_seed_{args.seed}{refine_suffix}"
    else:
        plot_name = f"{run_name}_variantD_{safe_repr}_clustered_seed_{args.seed}_epoch_{epoch}{refine_suffix}"
    plot_path = os.path.join(result_dir, f"{plot_name}.png")
    ckpt_path = os.path.join(ckpt_dir, f"{plot_name}.pt")

    if spatial is not None:
        plot_kwargs = {}
        if spatial_uns is not None:
            plot_kwargs.update(dict(
                spatial_uns=spatial_uns,
                library_id=library_id,
                img_key=img_key,
                all_spatial_plot=all_spatial_plot,
            ))
        title = format_dataset_for_plot(run_name)
        if has_labels:
            title = f"{title}\nari={refined_ari:.4f}"
        plot_spatial(
            spatial, pred_labels, labels, title,
            save_name=plot_path,
            spot_size=200, point_size=args.plot_point_size, emb=embedding,
            class_names=class_names,
            manual_title="",
            show_manual_panel=has_labels,
            **plot_kwargs
        )

    artifact_manifest = save_best_downstream_artifacts(
        args, pred_labels, embedding, labels, refined_ari, repr_name
    )

    torch.save({
        "dataset": args.dataset,
        "trial": trial,
        "epoch": epoch,
        "repr": repr_name,
        "raw_ari": raw_ari,
        "refined_ari": refined_ari,
        "ari": refined_ari,
        "has_ground_truth": has_labels,
        "args": vars(args),
        "pred_labels": pred_labels,
        "embedding": embedding,
        "artifact_manifest": artifact_manifest,
        "model_state_dict": cl_model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
    }, ckpt_path)

    return plot_path, ckpt_path


def save_variant_d_diagnostics(cl_model, discriminator, args, best_info, features, str_encodings, edges,
                               labels, n_clusters, spatial):
    # Variant D addition: gate maps/histograms and low/high branch ARI for interpretation.
    if not best_info.get("checkpoint_path") or best_info.get("checkpoint_path") == "NA":
        return {}
    if not os.path.exists(best_info["checkpoint_path"]):
        return {}

    import matplotlib.pyplot as plt

    ckpt = torch.load(best_info["checkpoint_path"], map_location=features.device)
    cl_model.load_state_dict(ckpt["model_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    cl_model.eval()
    discriminator.eval()

    with torch.no_grad():
        adj_1, adj_2, _, _ = discriminator(torch.cat((features, str_encodings), 1), edges, force_uniform=False)
        emb_dict = cl_model.get_variant_d_embeddings(features, adj_1, adj_2)

    gate_np = emb_dict["gate"].detach().cpu().numpy()
    gate_node = gate_np.mean(axis=1)
    stats = {
        "gate_detail_mean": float(np.mean(gate_node)),
        "gate_detail_std": float(np.std(gate_node)),
        "gate_detail_min": float(np.min(gate_node)),
        "gate_detail_max": float(np.max(gate_node)),
        "gate_detail_p10": float(np.percentile(gate_node, 10)),
        "gate_detail_p50": float(np.percentile(gate_node, 50)),
        "gate_detail_p90": float(np.percentile(gate_node, 90)),
    }

    run_name = result_name(args)
    diag_dir = os.path.join("./result", run_name)
    os.makedirs(diag_dir, exist_ok=True)
    gate_map_path = os.path.join(diag_dir, f"{run_name}_variantD_gate_map_seed_{args.seed}.png")
    gate_hist_path = os.path.join(diag_dir, f"{run_name}_variantD_gate_hist_seed_{args.seed}.png")

    if spatial is not None:
        plt.figure(figsize=(6, 6))
        plt.scatter(spatial[:, 0], spatial[:, 1], c=gate_node, s=8, cmap="viridis")
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"{run_name} Variant D gate mean per node")
        plt.tight_layout()
        plt.savefig(gate_map_path, dpi=300)
        plt.close()
        stats["gate_map_path"] = gate_map_path

    plt.figure(figsize=(6, 4))
    plt.hist(gate_node, bins=40, color="#4c78a8", edgecolor="white")
    plt.xlabel("gate mean per node")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(gate_hist_path, dpi=300)
    plt.close()
    stats["gate_hist_path"] = gate_hist_path

    branch_metrics = {}
    for emb_name in ["z_low", "z_high", "z_mix", "z_full", "concat"]:
        if emb_name == "concat":
            emb_tensor = torch.cat((emb_dict["z_mix"], emb_dict["z_full"]), dim=1)
        else:
            emb_tensor = emb_dict[emb_name]
        eval_res = evaluate_variant_d_embedding(
            emb_tensor.detach().cpu().numpy(), labels, n_clusters, spatial, args.refine_k, edges=edges
        )
        branch_metrics[emb_name] = {
            "raw_ari": eval_res["raw_ari"],
            "refined_ari": eval_res["refined_ari"],
            "boundary_f1": eval_res["boundary_f1"],
            "epoch": best_info.get("epoch", np.nan),
        }

    stats["diagnostics_text"] = (
        "Variant D diagnostics: "
        f"gate_mean={stats['gate_detail_mean']:.4f}, gate_std={stats['gate_detail_std']:.4f}, "
        f"gate_min={stats['gate_detail_min']:.4f}, gate_max={stats['gate_detail_max']:.4f}, "
        f"p10={stats['gate_detail_p10']:.4f}, p50={stats['gate_detail_p50']:.4f}, "
        f"p90={stats['gate_detail_p90']:.4f}, gate_map={stats.get('gate_map_path', 'NA')}, "
        f"gate_hist={stats.get('gate_hist_path', 'NA')}; "
        f"z_low_raw/refined={branch_metrics['z_low']['raw_ari']:.4f}/{branch_metrics['z_low']['refined_ari']:.4f}, "
        f"z_high_raw/refined={branch_metrics['z_high']['raw_ari']:.4f}/{branch_metrics['z_high']['refined_ari']:.4f}"
    )
    stats["branch_metrics"] = branch_metrics
    return stats


def main(args):
    setup_seed(args.seed)
    spatial_uns = None
    library_id = None
    img_key = "hires"
    all_spatial_plot = None
    ret = load_data(args.dataset, getattr(args, "data_path", None), spatial_k=args.k)

    if isinstance(ret, (list, tuple)) and len(ret) == 9:
        features, edges, str_encodings, train_mask, val_mask, test_mask, labels, nnodes, nfeats = ret
        class_names = None
    elif isinstance(ret, (list, tuple)) and len(ret) == 10:
        features, edges, str_encodings, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats = ret
    else:
        raise ValueError(f"load_data() should return 9 or 10 values, got {len(ret)}")


    spatial = None
    if args.dataset == 'E18_mouse_brain':
        try:
            adata_path = resolve_h5ad_path(args, './data/E18.5_mouse_brain/E18_adata_rna.h5ad')
            
            if not os.path.exists(adata_path):
                # Fall back to another common location
                adata_path = './data/E18/E18_adata_rna.h5ad'

            if os.path.exists(adata_path):
                print(f"Loading spatial coordinates from {adata_path} for visualization...")
                adata_temp = sc.read_h5ad(adata_path)
                
                # Prefer array_col / array_row when present
                if 'array_col' in adata_temp.obs.columns and 'array_row' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['array_col', 'array_row']].values
                elif 'spatial' in adata_temp.obsm:
                    spatial = adata_temp.obsm['spatial']
                else:
                    print("!! Warning: Could not find 'array_col'/'array_row' or 'spatial' in h5ad.")
            else:
                print(f"!! Warning: {adata_path} not found. Visualization will fail.")
        except Exception as e:
            print(f"!! Error loading spatial data: {e}")

            # 151673: 10x Visium grids, plus histology background
    elif args.dataset == '151673':
        data_dir = './data/151673'
        # --- 10x Visium: expression + spatial files ---
        h5_path = os.path.join(data_dir, 'filtered_feature_bc_matrix.h5')
        pos_path_txt = os.path.join(data_dir, 'spatial', 'tissue_positions_list.txt')
        pos_path_csv = os.path.join(data_dir, 'tissue_positions_list.csv')  # legacy
        pos_path = pos_path_txt if os.path.exists(pos_path_txt) else pos_path_csv
    
        adata_temp = sc.read_10x_h5(h5_path)
        adata_temp.var_names_make_unique()
    
        # tissue_positions_list: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
        positions_df = pd.read_csv(pos_path, header=None, sep=r'[\t,\s]+', engine='python')
        positions_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
        positions_df.set_index('barcode', inplace=True)
    
        # align barcodes (Visium h5 typically has "-1" suffix)
        if (len(adata_temp.obs_names) > 0) and (len(positions_df.index) > 0):
            if adata_temp.obs_names[0].endswith('-1') and (not positions_df.index[0].endswith('-1')):
                positions_df.index = positions_df.index + '-1'
            elif (not adata_temp.obs_names[0].endswith('-1')) and positions_df.index[0].endswith('-1'):
                positions_df.index = positions_df.index.str.replace('-1$', '', regex=True)
    
        adata_temp.obs = adata_temp.obs.join(positions_df, how='inner')
    
        # --- full grid coords for grey ring dots ---
        # (x,y) = (pxl_col_in_fullres, pxl_row_in_fullres)
        all_spatial_plot = adata_temp.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].astype(float).values
    
        # tissue-only coords for model outputs / colored dots
        tissue_mask = adata_temp.obs['in_tissue'].astype(int) == 1
        adata_temp = adata_temp[tissue_mask].copy()
        spatial = adata_temp.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].astype(float).values
    
        spatial_uns = load_visium_spatial_uns("./data/151673", library_id="151673")
        library_id = "151673"
        img_key = "hires"

        # --- load Visium background (H&E + scalefactors) ---
        library_id = '151673'
        img_key = 'hires'
        try:
            spatial_uns = load_visium_spatial_uns(data_dir, library_id=library_id)
        except Exception as e:
            print(f"[WARN] load_visium_spatial_uns failed: {e}")
            spatial_uns = None
    
        # --- labels ---
        labels_path = os.path.join(data_dir, 'metadata.tsv')
        labels_df = pd.read_csv(labels_path, sep='\t')
        labels_df.set_index(labels_df.columns[0], inplace=True)
    
        if (len(adata_temp.obs_names) > 0) and (len(labels_df.index) > 0):
            if adata_temp.obs_names[0].endswith('-1') and (not labels_df.index[0].endswith('-1')):
                labels_df.index = labels_df.index + '-1'
            elif (not adata_temp.obs_names[0].endswith('-1')) and labels_df.index[0].endswith('-1'):
                labels_df.index = labels_df.index.str.replace('-1$', '', regex=True)
    
        labels_df = labels_df.loc[adata_temp.obs_names]
        label_series = labels_df['layer_guess']
        valid_label_mask = label_series.notna()
        # Variant D addition: DLPFC unlabeled spots are excluded from ARI/n_clusters.
        # Without this, "nan" becomes an artificial eighth layer.
        label_names = sorted(label_series[valid_label_mask].astype(str).unique())
        label_map = {label: idx for idx, label in enumerate(label_names)}
        labels_np = np.full(len(label_series), -1, dtype=np.int64)
        for idx_label, label in enumerate(label_series.astype(object)):
            if pd.notna(label):
                labels_np[idx_label] = label_map[str(label)]
        adata_temp.obs['layer_guess'] = label_series.astype('string').fillna('Unknown')
        labels = torch.LongTensor(labels_np)


    elif args.dataset == '151674':
        data_dir = './data/151674'
        h5_path = os.path.join(data_dir, 'filtered_feature_bc_matrix.h5')

        # 151674 mirrors 151673: positions live under spatial/
        pos_path = os.path.join(data_dir, 'spatial', 'tissue_positions_list.txt')
        if not os.path.exists(pos_path):
            # Legacy layout fallback; does not affect other datasets
            pos_path = os.path.join(data_dir, 'tissue_positions_list.txt')
        if not os.path.exists(pos_path):
            raise FileNotFoundError(f"Cannot find tissue_positions_list: {pos_path}")

        # Expression matrix (the filtered matrix already holds tissue spots)
        adata_temp = sc.read_10x_h5(h5_path)
        adata_temp.var_names_make_unique()

        # Positions file (comma / tab / space separated)
        positions = None
        for sep in [',', '\t', ' ']:
            try:
                tmp = pd.read_csv(pos_path, header=None, sep=sep)
                if tmp.shape[1] >= 6:
                    positions = tmp.iloc[:, :6].copy()
                    break
            except Exception:
                pass
        if positions is None:
            positions = pd.read_csv(pos_path, header=None, sep=r'\s+', engine='python').iloc[:, :6].copy()

        # Drop a header row if present, otherwise numeric parsing fails
        if str(positions.iloc[0, 0]).lower() == 'barcode':
            positions = positions.iloc[1:].copy()

        # Visium columns 5 and 6 are (row_in_fullres, col_in_fullres)
        # plot_spatial expects (x=col, y=row)
        positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'py', 'px']  # py=row, px=col

        positions['barcode'] = positions['barcode'].astype(str)
        # Append the -1 suffix only when needed, avoiding "-1-1"
        positions['barcode'] = positions['barcode'].apply(lambda b: b if b.endswith('-1') else b + '-1')
        positions.set_index('barcode', inplace=True)

        # Full-array grid used for the hollow background rings
        all_spatial_plot = positions[['px', 'py']].astype(float).values

        # Join into adata.obs and keep tissue spots only
        adata_temp.obs = adata_temp.obs.join(positions, how='left')
        adata_temp = adata_temp[adata_temp.obs['in_tissue'] == 1].copy()

        # Coordinates of tissue spots for the coloured points (x=px, y=py)
        spatial = adata_temp.obs[['px', 'py']].astype(float).values

        # Histology background and scalefactors (same path as 151673)
        spatial_uns = load_visium_spatial_uns(data_dir, library_id="151674")
        library_id = "151674"
        img_key = "hires"

    elif args.dataset in {'BRCA', 'Breast_Caner'}:
        data_dir = resolve_h5ad_path(args, './data/BRCA/Breast_Caner')
        h5_path = os.path.join(data_dir, 'filtered_feature_bc_matrix.h5')
        positions = _read_visium_positions(data_dir).set_index('spot_id')
        adata_temp = sc.read_10x_h5(h5_path)
        adata_temp.var_names_make_unique()
        positions.index = _align_index_suffix(positions.index, adata_temp.obs_names)
        adata_temp.obs = adata_temp.obs.join(positions, how='left')
        adata_temp = adata_temp[adata_temp.obs['in_tissue'].astype(int) == 1].copy()
        spatial = adata_temp.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(dtype=float)
        all_spatial_plot = positions[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(dtype=float)
        spatial_uns = load_visium_spatial_uns(data_dir, library_id='BRCA')
        library_id = 'BRCA'
        img_key = 'hires'


    elif args.dataset == 'mouse_embryo':
        try:
            import glob
            # Locate the h5ad file automatically
            data_dir = './data/mouse_embryo'
            h5ad_files = glob.glob(os.path.join(data_dir, '*.h5ad'))
            
            if len(h5ad_files) > 0:
                print(f"Loading spatial coordinates for mouse_embryo from {os.path.basename(h5ad_files[0])}...")
                adata_temp = sc.read_h5ad(h5ad_files[0])
                
                # Try the common coordinate layouts
                if 'spatial' in adata_temp.obsm:
                    spatial = adata_temp.obsm['spatial']
                elif 'x' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['x', 'y']].values
            else:
                print("!! Warning: No .h5ad file found for visualization.")
        except Exception as e:
            print(f"Error loading mouse_embryo spatial: {e}")

    elif args.dataset == 'Stereo_AB':
        try:
            adata_path = resolve_h5ad_path(args, './data/Stereo/Stereo_AB.h5ad')
            
            if os.path.exists(adata_path):
                print(f"Loading spatial coordinates from {adata_path} for visualization...")
                adata_temp = sc.read_h5ad(adata_path)

                if 'spatial' in adata_temp.obsm:
                    spatial = adata_temp.obsm['spatial']
                elif 'array_col' in adata_temp.obs.columns and 'array_row' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['array_col', 'array_row']].values
                elif 'x' in adata_temp.obs.columns and 'y' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['x', 'y']].values
                else:
                    print("!! Warning: Could not find 'spatial' or common coordinate columns in Stereo_AB.h5ad.")
            else:
                print(f"!! Warning: {adata_path} not found. Spatial smoothing will be skipped.")
        except Exception as e:
            print(f"!! Error loading spatial data for Stereo_AB: {e}")

    elif args.dataset in {'MOB', 'MOB_section3'}:
        try:
            adata_path = resolve_h5ad_path(args, './data/MOB/MOB_3D.h5ad')

            if os.path.exists(adata_path):
                print(f"Loading MOB spatial coordinates from {adata_path} for visualization...")
                adata_temp = sc.read_h5ad(adata_path)
                if args.dataset == 'MOB_section3':
                    adata_temp = adata_temp[adata_temp.obs['section'].astype(str).to_numpy() == '3'].copy()
                else:
                    subset_idx = _mob_subset_indices(adata_temp, expected_nnodes=features.shape[0])
                    adata_temp = adata_temp[subset_idx].copy()

                if 'spatial' in adata_temp.obsm:
                    spatial = np.asarray(adata_temp.obsm['spatial'], dtype=float)
                elif {'xcoord', 'ycoord'}.issubset(adata_temp.obs.columns):
                    spatial = adata_temp.obs[['xcoord', 'ycoord']].astype(float).to_numpy()
                elif 'coord' in adata_temp.obsm:
                    spatial = np.asarray(adata_temp.obsm['coord'])[:, :2].astype(float)
                else:
                    print("!! Warning: Could not find MOB spatial coordinates.")
            else:
                print(f"!! Warning: {adata_path} not found. Spatial smoothing will be skipped.")
        except Exception as e:
            print(f"!! Error loading spatial data for MOB: {e}")

    elif args.dataset == 'Stereo_mouse_brain':
        try:
            adata_path = './data/Stereo/Stereo_mouse_brain.h5ad'

            if os.path.exists(adata_path):
                print(f"Loading spatial coordinates for Stereo_mouse_brain from {adata_path} ...")
                adata_temp = sc.read_h5ad(adata_path)

                if 'spatial' in adata_temp.obsm:
                    spatial = adata_temp.obsm['spatial']
                elif 'x' in adata_temp.obs.columns and 'y' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['x', 'y']].values
                elif 'x_array' in adata_temp.obs.columns and 'y_array' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['x_array', 'y_array']].values
                else:
                    print("!! Warning: Could not find spatial coordinates in Stereo_mouse_brain h5ad.")
            else:
                print(f"!! Warning: {adata_path} not found. Spatial smoothing/refinement will be skipped.")
        except Exception as e:
            print(f"Error loading Stereo_mouse_brain spatial: {e}")

    elif args.dataset in {'DPI', 'Stereo_DPI'}:
        try:
            adata_path = resolve_h5ad_path(args, './data/DPI/2DPI_1.h5ad')
            if os.path.exists(adata_path):
                print(f"Loading spatial coordinates for DPI from {adata_path} ...")
                adata_temp = sc.read_h5ad(adata_path)
                if 'spatial' in adata_temp.obsm:
                    spatial = adata_temp.obsm['spatial']
                elif 'x_array' in adata_temp.obs.columns and 'y_array' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['x_array', 'y_array']].values
                elif 'x' in adata_temp.obs.columns and 'y' in adata_temp.obs.columns:
                    spatial = adata_temp.obs[['x', 'y']].values
                else:
                    print("!! Warning: Could not find spatial coords for Stereo_DPI.")
            else:
                print(f"!! Warning: {adata_path} not found. Visualization will fail.")
        except Exception as e:
            print(f"!! Error loading spatial data for Stereo_DPI: {e}")

    elif args.dataset == 'CNS':
        try:
            adata_path = resolve_h5ad_path(args, './data/CNS/CNS3.h5ad')
            adata_temp = sc.read_h5ad(adata_path)
            adata_temp = adata_temp[_cns_subset_indices(adata_temp, max_nodes=features.shape[0], seed=args.seed)].copy()
            spatial = np.asarray(adata_temp.obsm['spatial'], dtype=float)
        except Exception as e:
            print(f"!! Error loading CNS spatial data: {e}")

    elif args.dataset == 'BRCA_Xexium':
        try:
            adata_path = resolve_h5ad_path(args, './data/BRCA/BRCA_Xexium.h5ad')
            if os.path.exists(adata_path):
                print(f"Loading spatial coordinates for BRCA Xenium from {adata_path} ...")
                adata_temp = sc.read_h5ad(adata_path)
                adata_temp = adata_temp[_brca_xenium_subset_indices(adata_temp, features.shape[0])].copy()
                if 'spatial' in adata_temp.obsm:
                    spatial = np.asarray(adata_temp.obsm['spatial'], dtype=float)
                elif {'x_centroid', 'y_centroid'}.issubset(adata_temp.obs.columns):
                    spatial = adata_temp.obs[['x_centroid', 'y_centroid']].astype(float).to_numpy()
                else:
                    print("!! Warning: Could not find spatial coordinates for BRCA Xenium.")
            else:
                print(f"!! Warning: {adata_path} not found. Visualization will fail.")
        except Exception as e:
            print(f"!! Error loading spatial data for BRCA Xenium: {e}")


    # --------------------------------


    valid_cluster_labels = labels[labels >= 0] if torch.is_tensor(labels) else np.asarray(labels)[np.asarray(labels) >= 0]
    has_labels = has_ground_truth_labels(labels)
    if getattr(args, "n_clusters_override", 0) and int(args.n_clusters_override) > 0:
        n_clusters = int(args.n_clusters_override)
    else:
        n_clusters = len(torch.unique(valid_cluster_labels)) if torch.is_tensor(valid_cluster_labels) else len(np.unique(valid_cluster_labels))
    if n_clusters <= 0:
        raise ValueError("No ground-truth labels are available to infer n_clusters; set --n_clusters_override.")
    print(f"Number of clusters: {n_clusters}")
    if not has_labels:
        print("[Evaluation] Ground-truth labels are unavailable; ARI will be skipped.")

    results = []
    global_best_score = -1.0
    global_best_info = {"score": -1.0, "has_ground_truth": has_labels}

    os.makedirs("./result/checkpoints", exist_ok=True)
    semantic_pos_mask = build_variant_d_semantic_mask(features, k=args.sem_k)
    print(
        f"[Variant D] Cached semantic cosine top-k mask with sem_k={args.sem_k}; "
        f"source=expression/PCA features only, shape={tuple(features.shape)}, "
        f"str_encoding_shape={tuple(str_encodings.shape)}"
    )

    for trial in range(args.ntrials):
        setup_seed(args.seed + trial)
        trial_best_score = -1.0
        variant_d_run_metrics = {}

        cl_model = GCL(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim,
                    proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse,
                    gate_hidden=args.gate_hidden, full_mlp_hidden=args.full_mlp_hidden,
                    full_mlp_out=args.full_mlp_out,
                    variant_d_gate_mode=args.variant_d_gate_mode).cuda()


        discriminator = Edge_Discriminator(nnodes, nfeats + str_encodings.shape[1], args.alpha, args.sparse,temperature=args.temp_disc).cuda()

        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)

        features = features.cuda()
        str_encodings = str_encodings.cuda()
        edges = edges.cuda()

        for epoch in range(1, args.epochs + 1):
            for _ in range(args.cl_rounds):
                model_loss = train_model(
                    cl_model, discriminator, optimizer_cl, features, str_encodings, edges,
                    args, semantic_pos_mask, epoch=epoch
                )
            rank_loss = train_discriminator(cl_model, discriminator, optimizer_discriminator, features, str_encodings, edges, args)

            should_eval = (epoch % args.eval_freq == 0) or (epoch == args.epochs)
            vd_log = cl_model.last_variant_d_log
            current_best_repr = global_best_info.get("repr", "NA")
            current_best_score = global_best_info.get("score", -1.0)
            best_label = f"{current_best_score:.4f}" if has_labels else ("saved" if current_best_score >= 0 else "NA")
            print(
                "[TRAIN] Trial:{} | Epoch:{:04d} | CL {:.4f} | RANK {:.4f} | "
                "FL {:.4f}*{:.3f} | FH {:.4f}*{:.3f} | MIX {:.4f}*{:.3f} | Base {:.4f} | "
                "Gate mean/std {:.4f}/{:.4f} | Best {} {}".format(
                    trial, epoch, model_loss, rank_loss,
                    vd_log["l_fl"], vd_log["eff_lambda_fl"],
                    vd_log["l_fh"], vd_log["eff_lambda_fh"],
                    vd_log.get("l_mix", 0.0), vd_log.get("eff_lambda_mix", 0.0),
                    vd_log["base_loss"],
                    vd_log["gate_mean"], vd_log["gate_std"],
                    current_best_repr, best_label
                )
            )
            if not should_eval:
                continue

            cl_model.eval()
            discriminator.eval()
            with torch.no_grad():
                adj_1, adj_2, _, _ = discriminator(
                    torch.cat((features, str_encodings), 1), edges, force_uniform=False
                )
                emb_dict = cl_model.get_variant_d_embeddings(features, adj_1, adj_2)
                candidates = {
                    "z_mix": emb_dict["z_mix"],
                }
                if args.eval_aux_embeddings and not bool(getattr(args, "no_aux_branch", 0)):
                    candidates["z_full"] = emb_dict["z_full"]
                    if emb_dict["z_mix"].shape[0] == emb_dict["z_full"].shape[0]:
                        candidates["concat"] = torch.cat((emb_dict["z_mix"], emb_dict["z_full"]), dim=1)

                epoch_best_repr = None
                epoch_best_score = -1.0
                epoch_eval_rows = []
                for repr_name, emb_tensor in candidates.items():
                    embedding = emb_tensor.detach().cpu().numpy()
                    eval_res = evaluate_variant_d_embedding(
                        embedding, labels, n_clusters, spatial, args.refine_k, edges=edges
                    )
                    raw_ari = eval_res["raw_ari"]
                    refined_ari = eval_res["refined_ari"]
                    boundary_f1 = eval_res["boundary_f1"]
                    pred_labels = eval_res["refined_labels"]
                    epoch_eval_rows.append((repr_name, raw_ari, refined_ari, boundary_f1))
                    score = float(refined_ari) if has_labels else float(trial * args.epochs + epoch)

                    prev_repr_best = variant_d_run_metrics.get(repr_name)
                    if prev_repr_best is None or score > prev_repr_best["score"]:
                        variant_d_run_metrics[repr_name] = {
                            "raw_ari": raw_ari,
                            "refined_ari": refined_ari,
                            "boundary_f1": boundary_f1,
                            "score": score,
                            "epoch": epoch,
                        }

                    if score > epoch_best_score:
                        epoch_best_score = score
                        epoch_best_repr = repr_name

                    is_main_repr = (repr_name == "z_mix")
                    if is_main_repr and score > trial_best_score:
                        trial_best_score = score
                        if has_labels:
                            print(f"     [Trial {trial} Best] Variant D {repr_name} ARI {trial_best_score:.4f}")
                        else:
                            print(f"     [Trial {trial} Saved] Variant D {repr_name} clustering at epoch {epoch}")

                    if is_main_repr and score > global_best_score:
                        global_best_score = score
                        if has_labels:
                            print(f"     [GLOBAL Best] Variant D {repr_name} ARI {global_best_score:.4f}")
                        else:
                            print(f"     [GLOBAL Saved] Variant D {repr_name} clustering at epoch {epoch}")
                        previous_best_plot = global_best_info.get("plot_path")
                        plot_path, ckpt_path = save_variant_d_best(
                            cl_model, discriminator, args, trial, epoch, repr_name, raw_ari, refined_ari,
                            pred_labels, embedding, spatial, labels, class_names,
                            spatial_uns, library_id, img_key, all_spatial_plot
                        )
                        cleanup_superseded_visualization(previous_best_plot, plot_path)
                        vd_log = getattr(cl_model, "last_variant_d_log", {})
                        global_best_info = {
                            "score": score,
                            "ari": refined_ari if has_labels else np.nan,
                            "raw_ari": raw_ari,
                            "refined_ari": refined_ari,
                            "boundary_f1": boundary_f1,
                            "has_ground_truth": has_labels,
                            "repr": repr_name,
                            "epoch": epoch,
                            "trial": trial,
                            "seed": args.seed,
                            "gate_mean": vd_log.get("gate_mean", None),
                            "gate_std": vd_log.get("gate_std", None),
                            "l_fl": vd_log.get("l_fl", None),
                            "l_fh": vd_log.get("l_fh", None),
                            "base_loss": vd_log.get("base_loss", None),
                            "plot_path": plot_path,
                            "checkpoint_path": ckpt_path,
                            "command": build_variant_d_command(args),
                        }

                if has_labels:
                    print("  -> Variant D eval table:")
                    print("     embedding | raw_ARI | refined_ARI | B-F1")
                    for repr_name, raw_ari, refined_ari, boundary_f1 in epoch_eval_rows:
                        bf1_str = "NA" if boundary_f1 is None or (isinstance(boundary_f1, float) and np.isnan(boundary_f1)) else f"{boundary_f1:.4f}"
                        print(f"     {repr_name:9s} | {raw_ari:.4f} | {refined_ari:.4f} | {bf1_str}")
                    print(f"  -> Epoch {epoch} best embedding: {epoch_best_repr} ARI={epoch_best_score:.4f}")
                else:
                    print("  -> Variant D clustering saved/evaluated without ARI because labels are unavailable.")
                    print(f"  -> Epoch {epoch} selected embedding: {epoch_best_repr}")

        results.append(trial_best_score)

    if has_labels:
        print('\n[FINAL RESULT] Dataset:{} | Run:{} | Avg ARI:{:.4f}+-{:.4f}'.format(args.dataset, args.ntrials, np.mean(results), np.std(results)))
    else:
        print('\n[FINAL RESULT] Dataset:{} | Run:{} | ARI skipped because ground-truth labels are unavailable.'.format(args.dataset, args.ntrials))

    diag_info = {}
    if has_labels and args.save_diagnostics:
        diag_info = save_variant_d_diagnostics(
            cl_model, discriminator, args, global_best_info, features, str_encodings, edges,
            labels, n_clusters, spatial
        )
    if diag_info:
        branch_metrics = diag_info.pop("branch_metrics", {})
        variant_d_run_metrics.update(branch_metrics)
        global_best_info.update(diag_info)
    summary_path = os.environ.get("DHALF_SUMMARY_PATH", "./result/variantD_summary.txt")
    run_csv_path = os.environ.get("DHALF_RUN_CSV", "./result/variantD_151673_151674_runs.csv")
    append_variant_d_summary(summary_path, args, global_best_info)
    append_variant_d_run_csv(run_csv_path, args, global_best_info, variant_d_run_metrics)
    print(f"[Variant D] Summary appended to {summary_path}")
    print(f"[Variant D] CSV row appended to {run_csv_path}")
    print(f"[Variant D] Best plot: {global_best_info.get('plot_path', 'NA')}")
    print(f"[Variant D] Best checkpoint: {global_best_info.get('checkpoint_path', 'NA')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument('-dataset', type=str, default='E18_mouse_brain',choices=['E18_mouse_brain', '151673' ,'151674', 'Stereo_AB', 'DPI', 'CNS', 'BRCA', 'Breast_Caner', 'BRCA_Xexium', 'MOB', 'MOB_section3'])
    parser.add_argument('--data_path', type=str, default=None,
                    help='Optional dataset file or directory override, e.g. an E18_adata_rna.h5ad path.')
    parser.add_argument('--result_name', type=str, default=None,
                    help='Optional result subdirectory and output filename prefix.')
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-refine_k', type=int, default=5,
                    help='k for refine_label_spatial majority vote; set 0 to disable refinement')

    parser.add_argument('-ntrials', type=int, default=1)
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-eval_freq', type=int, default=20)
    parser.add_argument('-epochs', type=int, default=600)
    parser.add_argument('-lr_gcl', type=float, default=0.0001)
    parser.add_argument('-lr_disc', type=float, default=0.0001)
    parser.add_argument('-cl_rounds', type=int, default=2)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.5)
    # Edge-discriminator options
    parser.add_argument('-alpha', type=float, default=0.1)  # alpha: strength of the negative high-pass term
    parser.add_argument('-margin_hom', type=float, default=0.3)
    parser.add_argument('-margin_het', type=float, default=0.7)
    parser.add_argument('--temp_disc', type=float, default=1.0)  # discriminator temperature
    # Graph contrastive learning options
    parser.add_argument('-nlayers_enc', type=int, default=2)
    parser.add_argument('-nlayers_proj', type=int, default=1)
    parser.add_argument('-emb_dim', type=int, default=512)
    parser.add_argument('-proj_dim', type=int, default=512)
    parser.add_argument('-k', type=int, default=10) 
    parser.add_argument('--lambda_fl', type=float, default=1.0)
    parser.add_argument('--lambda_fh', type=float, default=1.0)
    parser.add_argument('--lambda_base', type=float, default=0.0,
                    help='Weight of the DGI-style global-summary loss L_base.')
    parser.add_argument('--lambda_mix', type=float, default=1.0,
                    help='Weight of the fusion-consistency loss L_MIX that trains the Eq.(12) gate. '
                         'Enabled by default so that the gate is optimised together with the '
                         'representation modules; set 0 only to reproduce the legacy frozen-gate ablation.')
    parser.add_argument('--mix_detach_branches', type=int, default=1, choices=[0, 1],
                    help='1 stops the gradient of L_mix at Z_L/Z_H so that only the gate is trained.')
    parser.add_argument('--sem_k', type=int, default=10)
    parser.add_argument('--variant_d_warmup', type=int, default=0)
    parser.add_argument('--gate_hidden', type=int, default=64)
    parser.add_argument('--variant_d_gate_mode', type=str, default='node', choices=['node', 'scalar'])
    parser.add_argument('--full_mlp_hidden', type=int, default=256)
    parser.add_argument('--full_mlp_out', type=int, default=None)
    parser.add_argument('--use_hom_pos', type=int, default=1)
    parser.add_argument('--fl_mask_mode', type=str, default=None,
                    choices=[None, 'spatial_homophilic', 'spatial_all', 'spatial_heterophilic', 'semantic'],
                    help='Positive-pair rule for the Full-Low term. Default resolves from --use_hom_pos.')
    parser.add_argument('--fh_mask_mode', type=str, default=None,
                    choices=[None, 'semantic', 'spatial_all', 'spatial_homophilic', 'spatial_heterophilic'],
                    help='Positive-pair rule for the Full-High term. Default is cosine top-k semantic.')
    parser.add_argument('--no_aux_branch', type=int, default=0, choices=[0, 1],
                    help='1 removes the graph-free full-pass branch (and its FL/FH terms) while '
                         'keeping the dual-frequency encoders, node gate, and discriminator.')
    parser.add_argument('--ablation_name', type=str, default='')
    parser.add_argument('--plot_point_size', type=float, default=80.0)
    parser.add_argument('--n_clusters_override', type=int, default=0,
                    help='Override clustering count for unsupervised evaluation/visualization.')
    parser.add_argument('--eval_aux_embeddings', type=int, default=1, choices=[0, 1],
                    help='Evaluate z_full/concat in addition to the saved z_mix output.')
    parser.add_argument('--save_diagnostics', type=int, default=1, choices=[0, 1],
                    help='Save gate and branch diagnostics after training.')


    args = parser.parse_args()
    if args.full_mlp_out is None:
        args.full_mlp_out = args.proj_dim
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    main(args)
