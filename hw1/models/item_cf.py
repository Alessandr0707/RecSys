import numpy as np
import scipy.sparse as sp
import pandas as pd

class ItemKNN:
    def __init__(self, k_neighbors=50, min_common=1, shrinkage=0.0, verbose=True):
        self.k_neighbors = k_neighbors
        self.min_common = min_common
        self.shrinkage = shrinkage
        self.verbose = verbose
        self.user_index_ = None
        self.item_index_ = None
        self.item_sim_ = None
        self.rating_matrix_ = None

    def fit(self, train_df, n_users=None, n_items=None):
        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique()
        
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {it: i for i, it in enumerate(items)}

        rows = train_df["user_id"].map(self.user_index_).to_numpy()
        cols = train_df["parent_asin"].map(self.item_index_).to_numpy()
        vals = train_df["rating"].to_numpy(dtype=np.float32)

        n_u = len(self.user_index_)
        n_i = len(self.item_index_)
        
        R = sp.csr_matrix((vals, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
        self.rating_matrix_ = R

        norms = np.sqrt(R.multiply(R).sum(axis=0)).A1.astype(np.float32)
        norms[norms == 0] = 1.0
        R_norm = R.multiply(1.0 / norms).tocsr()

        sim = (R_norm.T @ R_norm).tocsr()
        sim.setdiag(0)
        sim.eliminate_zeros()

        k = min(self.k_neighbors, n_i - 1)
        out_rows, out_cols, out_vals = [], [], []
        indptr, indices, data = sim.indptr, sim.indices, sim.data
        for i in range(n_i):
            s, e = indptr[i], indptr[i + 1]
            if s == e:
                continue
            rv = data[s:e]
            rc = indices[s:e]
            if len(rv) <= k:
                out_rows.append(np.full(len(rv), i, dtype=np.int32))
                out_cols.append(rc)
                out_vals.append(rv)
            else:
                idx = np.argpartition(-rv, k)[:k]
                out_rows.append(np.full(k, i, dtype=np.int32))
                out_cols.append(rc[idx])
                out_vals.append(rv[idx])

        if out_rows:
            sim_top = sp.csr_matrix(
                (np.concatenate(out_vals),
                 (np.concatenate(out_rows), np.concatenate(out_cols))),
                shape=(n_i, n_i),
            )
        else:
            sim_top = sp.csr_matrix((n_i, n_i), dtype=np.float32)

        self.item_sim_ = sim_top
        if self.verbose:
            print(f"[ItemKNN] users={n_u}, items={n_i}, nnz_R={R.nnz}, nnz_S={sim_top.nnz}")
        return self

    def recommend(self, user_ids, candidates, k=10, exclude_seen=True, train_df=None):
        R = self.rating_matrix_
        S = self.item_sim_
        item_ids = np.array(list(self.item_index_.keys()))
        item_pos = {it: idx for idx, it in enumerate(item_ids)}

        if candidates is not None:
            # Добавили dtype=np.int64
            pool_idx = np.array([item_pos[it] for it in candidates if it in item_pos], dtype=np.int64)
        else:
            pool_idx = np.arange(len(item_ids), dtype=np.int64)

        seen = {}
        if exclude_seen and train_df is not None:
            for u, grp in train_df.groupby("user_id"):
                seen[u] = set(grp["parent_asin"].tolist())

        result = {}
        n_pool = len(pool_idx)
        kk = min(k, n_pool)
        
        for u in user_ids:
            ju = self.user_index_.get(u)
            if ju is None:
                result[u] = item_ids[pool_idx[:kk]].tolist()
                continue
                
            scores = (R.getrow(ju) @ S).toarray().ravel()
            s = scores[pool_idx].copy()
            
            if exclude_seen and u in seen:
                # Добавили dtype=np.int64
                seen_idx = np.array([item_pos[it] for it in seen[u] if it in item_pos], dtype=np.int64)
                if len(seen_idx) > 0:
                    s[np.isin(pool_idx, seen_idx)] = -np.inf
                    
            top_local = np.argpartition(-s, kth=kk - 1)[:kk]
            top_local = top_local[np.argsort(-s[top_local])]
            result[u] = item_ids[pool_idx[top_local]].tolist()
            
        return result