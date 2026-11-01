import numpy as np
import scipy.sparse as sp

class ALS:
    def __init__(self, n_factors=32, n_iter=15, reg=0.1, implicit=False,
                 alpha=40.0, random_state=42, verbose=False):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.reg = reg
        self.implicit = implicit
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        self.user_index_ = None
        self.item_index_ = None
        self.user_factors_ = None
        self.item_factors_ = None
        self.global_mean_ = 0.0
        self.R_ = None

    def fit(self, train_df, n_users=None, n_items=None):
        rng = np.random.default_rng(self.random_state)
        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique() # Адаптировано
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {it: i for i, it in enumerate(items)}
        
        rows = train_df["user_id"].map(self.user_index_).to_numpy()
        cols = train_df["parent_asin"].map(self.item_index_).to_numpy() # Адаптировано
        vals = train_df["rating"].to_numpy(dtype=np.float32)
        
        n_u, n_i = len(self.user_index_), len(self.item_index_)
        if not self.implicit:
            self.global_mean_ = float(vals.mean())
            
        R = sp.csr_matrix((vals, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
        self.R_ = R
        
        P = rng.normal(0, 0.01, size=(n_u, self.n_factors)).astype(np.float32)
        Q = rng.normal(0, 0.01, size=(n_i, self.n_factors)).astype(np.float32)
        
        I = np.eye(self.n_factors, dtype=np.float32)
        Rt = R.T.tocsr()
        
        for it in range(self.n_iter):
            G_Q = Q.T @ Q + self.reg * I
            for u in range(n_u):
                s, e = R.indptr[u], R.indptr[u + 1]
                idx = R.indices[s:e]
                r = R.data[s:e]
                Qi = Q[idx]
                if self.implicit:
                    c = self.alpha * r
                    A = G_Q + (Qi.T * c) @ Qi
                    b = (Qi.T * c).sum(axis=1)
                else:
                    A = Qi.T @ Qi + self.reg * I
                    b = Qi.T @ r
                P[u] = np.linalg.solve(A, b)
                
            G_P = P.T @ P + self.reg * I
            for i in range(n_i):
                s, e = Rt.indptr[i], Rt.indptr[i + 1]
                idx = Rt.indices[s:e]
                r = Rt.data[s:e]
                Pi = P[idx]
                if self.implicit:
                    c = self.alpha * r
                    A = G_P + (Pi.T * c) @ Pi
                    b = (Pi.T * c).sum(axis=1)
                else:
                    A = Pi.T @ Pi + self.reg * I
                    b = Pi.T @ r
                Q[i] = np.linalg.solve(A, b)
                
            if self.verbose:
                print(f"[ALS] iter {it + 1}/{self.n_iter}")
                
        self.user_factors_ = P
        self.item_factors_ = Q
        return self

    def recommend(self, user_ids, candidates, k=10, exclude_seen=True, train_df=None):
        item_ids = np.array(list(self.item_index_.keys()))
        item_pos = {it: idx for idx, it in enumerate(item_ids)}
        
        if candidates is not None:
            pool_idx = np.array([item_pos[it] for it in candidates if it in item_pos])
        else:
            pool_idx = np.arange(len(item_ids))
            
        kk = min(k, len(pool_idx))
        Q_pool = self.item_factors_[pool_idx]
        
        seen = {}
        if exclude_seen and train_df is not None:
            for u, grp in train_df.groupby("user_id"):
                seen[u] = set(grp["parent_asin"].tolist()) # Адаптировано
                
        result = {}
        for u in user_ids:
            ju = self.user_index_.get(u)
            if ju is None:
                result[u] = item_ids[pool_idx[:kk]].tolist()
                continue
                
            s = Q_pool @ self.user_factors_[ju]
            if not self.implicit:
                s = s + self.global_mean_
                
            if exclude_seen and u in seen:
                seen_idx = np.array([item_pos[it] for it in seen[u] if it in item_pos])
                if len(seen_idx) > 0:
                    s[np.isin(pool_idx, seen_idx)] = -np.inf
                    
            top_local = np.argpartition(-s, kth=kk - 1)[:kk]
            top_local = top_local[np.argsort(-s[top_local])]
            result[u] = item_ids[pool_idx[top_local]].tolist()
            
        return result