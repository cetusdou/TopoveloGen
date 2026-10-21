"""Native scVelo and explicit graph operators for projected comparisons."""

import numpy as np
from scipy import sparse


NATIVE_SETTINGS = dict(xkey="Ms", n_neighbors=20, n_recurse_neighbors=1,
                       sqrt_transform=False, approx=False, mode_neighbors="distances",
                       n_jobs=1, backend="threading", show_progress_bar=False)
EMBEDDING_SETTINGS = dict(basis="umap", scale=10, self_transitions=True,
                          use_negative_cosines=True, direct_pca_projection=False,
                          retain_scale=False, autoscale=False, all_comps=True)


def region_labels(topology, z):
    x, y = z[:, 0], z[:, 1]
    if topology == "cycle":
        angle = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        return np.array(["quadrant_1", "quadrant_2", "quadrant_3", "quadrant_4"])[
            np.floor(angle / (np.pi / 2)).astype(int)]
    if topology == "bifurcation":
        return np.where(x < 0, "trunk_x_lt_0", np.where(y >= 0, "upper_x_ge_0", "lower_x_ge_0"))
    return np.where(x < 1, "cycle_x_lt_1", np.where(y >= 0, "upper_x_ge_1", "lower_x_ge_1"))


def fixed_distances(d):
    indices = d["neighbor_indices"]
    state = np.log1p(d["sc"] / d["capture"])
    distance = np.linalg.norm(state[indices] - state[:, None, :], axis=2)
    distance = np.maximum(distance, np.finfo(np.float32).eps)
    n, k = indices.shape
    graph = sparse.csr_matrix((distance.ravel(),
                               (np.repeat(np.arange(n), k), indices.ravel())), shape=(n, n))
    if np.any(np.diff(graph.indptr) != k):
        raise ValueError("Archived neighbor indices are not unique per cell")
    return graph


def native_project(d, velocity, distances):
    import anndata as ad
    import scvelo as scv

    adata = ad.AnnData(X=d["observed_smoothed"].astype(np.float32))
    adata.layers["Ms"] = d["observed_smoothed"].astype(np.float32)
    adata.layers["velocity"] = velocity.astype(np.float32)
    adata.var["velocity_genes"] = np.ones(adata.n_vars, dtype=bool)
    adata.obsm["X_umap"] = d["umap"].copy()
    adata.obsp["distances"] = distances.copy()
    adata.obsp["connectivities"] = distances.copy().astype(bool).astype(float)
    adata.uns["neighbors"] = {"distances_key": "distances",
                               "connectivities_key": "connectivities",
                               "params": {"n_neighbors": 20, "method": "archived_fixed_knn"}}
    scv.tl.velocity_graph(adata, vkey="velocity", **NATIVE_SETTINGS)
    scv.tl.velocity_embedding(adata, vkey="velocity", **EMBEDDING_SETTINGS)
    result = np.asarray(adata.obsm["velocity_umap"], dtype=float)
    if result.shape != (adata.n_obs, 2) or not np.isfinite(result).all():
        raise ValueError("Native embedding is nonfinite or has unexpected shape")
    return result
