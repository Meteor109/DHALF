DHALF - dataset checklist
=========================

Raw datasets are NOT redistributed with this repository. Place the files below
under ./data/ (paths are relative to the repository root, which must also be
the working directory when running main.py).

[1] data/151673/            DLPFC sample 151673 (10x Visium, Lieber Institute)
    filtered_feature_bc_matrix.h5
    tissue_positions_list.csv          no header, 6 columns:
                                       barcode, in_tissue, array_row, array_col,
                                       pxl_row_in_fullres, pxl_col_in_fullres
    metadata.tsv                       tab separated, first column = barcode,
                                       must contain a "layer_guess" column
    spatial/tissue_hires_image.png
    spatial/tissue_lowres_image.png
    spatial/scalefactors_json.json

[2] data/151674/            DLPFC sample 151674 (same study)
    filtered_feature_bc_matrix.h5
    cluster_labels_151674.csv          columns: key ("151674_<barcode>"),
                                       ground_truth (layer name)
    spatial/tissue_positions_list.txt  same 6 columns, no header
    spatial/tissue_hires_image.png
    spatial/tissue_lowres_image.png
    spatial/scalefactors_json.json

[3] data/BRCA/Breast_Caner/ Visium CytAssist human breast cancer
    filtered_feature_bc_matrix.h5
    spatial/tissue_positions_list.csv  6 columns, no header
    spatial/brca_ground_truth_annotation.csv
                                       columns: spot_id, x, y,
                                       ground_truth_annotation
    spatial/tissue_hires_image.png
    spatial/tissue_lowres_image.png
    spatial/scalefactors_json.json

[4] data/E18.5_mouse_brain/ E18.5 mouse brain (MISAR-seq RNA)
    E18_adata_rna.h5ad                 obs needs array_row / array_col and
                                       Combined_Clusters (14 domains)

[5] data/Stereo/            ARTISTA adult telencephalon (Stereo-seq)
    Stereo_AB.h5ad                     obsm["spatial"] and obs["Annotation"]
                                       (16 annotated cell types)

Notes
-----
* The annotation tables (metadata.tsv, cluster_labels_151674.csv,
  brca_ground_truth_annotation.csv) are required: the loaders need them to
  compute ARI and to fix the number of clusters.
* The structural-encoding cache is written to <cwd>/../data/se/<dataset>/ and is
  created automatically on the first run.
* See README.md for download links and the exact file layout of each dataset.
