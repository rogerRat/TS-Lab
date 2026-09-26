"""FAISS IVF matching ZifanL/TSDS @ 3dfe139, faiss_helper.py.

Both candidate retrieval and retrieved-union KDE use upstream defaults.
No exact-search fallback or nprobe enlargement is applied.
"""
import numpy as np


class FaissIndexIVFFlat:
    def __init__(self, data, nprobe=10):
        import faiss
        data = np.ascontiguousarray(data, dtype=np.float32)
        nlist = int(np.sqrt(data.shape[0])) // 2
        if nlist < 1:
            raise ValueError("Upstream TSDS IVF needs at least four trajectory points")
        quantizer = faiss.IndexFlatL2(data.shape[-1])
        self.index = faiss.IndexIVFFlat(quantizer, data.shape[-1], nlist)
        self.index.train(data)
        self.index.add(data)
        self.index.nprobe = nprobe

    def search(self, query, K):
        return self.index.search(np.ascontiguousarray(query, dtype=np.float32), K)


def trajectory_neighbours(xb, xq, *, max_k=64, kde_k=16, sigma=4.):
    max_k = min(max_k, len(xb) // 10)
    kde_k = min(kde_k, len(xb) // 10)
    if max_k < 2:
        raise ValueError("Upstream TSDS requires at least 20 candidate trajectories (MAX_K >= 2)")
    distances, indices = FaissIndexIVFFlat(xb).search(xq, max_k)
    if np.any(indices < 0):
        raise ValueError("TSDS IVF returned missing neighbours with upstream nprobe=10")
    order = np.argsort(distances, axis=-1)
    distances = np.sqrt(np.take_along_axis(distances, order, axis=1))
    indices = np.take_along_axis(indices, order, axis=1)
    unique = list(set(indices.reshape(-1).tolist()))
    if sigma == 0:
        kdes = np.ones_like(indices, dtype=np.float32)
    else:
        features = xb[unique]
        d2, _ = FaissIndexIVFFlat(features).search(features, kde_k)
        kernel = 1 - d2 / sigma ** 2
        kde = (kernel * (kernel > 0)).sum(axis=-1)
        lookup = dict(zip(unique, kde))
        kdes = np.vectorize(lookup.__getitem__)(indices)
    return distances, indices, kdes, {"max_k": max_k, "kde_k": kde_k,
                                      "kde_points": len(unique), "nprobe": 10,
                                      "nlist": int(np.sqrt(len(xb))) // 2}
