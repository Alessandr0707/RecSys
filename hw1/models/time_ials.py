import numpy as np
import scipy.sparse as sp

class TimeIALS:
    def __init__(self, n_factors=32, n_iter=15, reg=0.1, alpha=40.0, half_life_days=30.0, random_state=42, verbose=False):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.reg = reg
        self.alpha = alpha
        self.half_life_days = half_life_days
        self.random_state = random_state
        self.verbose = verbose
        self.user_index_ = None
        self.item_index_ = None
        self.user_factors_ = None
        self.item_factors_ = None
        self.R_ = None

    def fit(self, train_df):
        rng = np.random.default_rng(self.random_state)
        
        # 1. Вычисляем временной распад (Target Signal)
        # Переводим миллисекунды в дни
        max_ts = train_df['timestamp'].max()
        days_diff = (max_ts - train_df['timestamp']) / (1000 * 60 * 60 * 24)
        # Формула полураспада: 0.5 ^ (days / half_life)
        time_decay = 0.5 ** (days_diff / self.half_life_days)
        
        # 2. Строим матрицы
        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique()
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {it: i for i, it in enumerate(items)}
        
        rows = train_df["user_id"].map(self.user_index_).to_numpy()
        cols = train_df["parent_asin"].map(self.item_index_).to_numpy()
        
        # Ratings (оценки) и P (наши кастомные таргеты)
        vals_r = train_df["rating"].to_numpy(dtype=np.float32)
        vals_p = time_decay.to_numpy(dtype=np.float32) # Используем время как target
        
        n_u, n_i = len(self.user_index_), len(self.item_index_)
        
        R = sp.csr_matrix((vals_r, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
        P = sp.csr_matrix((vals_p, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
        
        self.R_ = R
        Rt = R.T.tocsr()
        Pt = P.T.tocsr()
        
        # Инициализация факторов
        P_fact = rng.normal(0, 0.01, size=(n_u, self.n_factors)).astype(np.float32)
        Q_fact = rng.normal(0, 0.01, size=(n_i, self.n_factors)).astype(np.float32)
        
        I = np.eye(self.n_factors, dtype=np.float32)
        
        # Цикл обучения ALS
        for it in range(self.n_iter):
            # --- Обновляем пользователей (P_fact) ---
            G_Q = Q_fact.T @ Q_fact + self.reg * I
            for u in range(n_u):
                s, e = R.indptr[u], R.indptr[u + 1]
                if s == e: continue
                
                idx = R.indices[s:e]
                r = R.data[s:e] # ratings
                p = P[u].data # наши кастомные таргеты для этого юзера
                
                Qi = Q_fact[idx]
                
                # c = alpha * rating
                c = self.alpha * r 
                
                # A = G_Q + Qi^T diag(c) Qi
                A = G_Q + (Qi.T * c) @ Qi
                
                # ИЗМЕНЕНИЕ: b = Qi^T (c * p)
                b = (Qi.T * (c * p)).sum(axis=1) 
                
                P_fact[u] = np.linalg.solve(A, b)
                
            # --- Обновляем объекты (Q_fact) ---
            G_P = P_fact.T @ P_fact + self.reg * I
            for i in range(n_i):
                s, e = Rt.indptr[i], Rt.indptr[i + 1]
                if s == e: continue
                
                idx = Rt.indices[s:e]
                r = Rt.data[s:e]
                p = Pt[i].data
                
                Pi = P_fact[idx]
                c = self.alpha * r
                
                A = G_P + (Pi.T * c) @ Pi
                # ИЗМЕНЕНИЕ: b = Pi^T (c * p)
                b = (Pi.T * (c * p)).sum(axis=1)
                
                Q_fact[i] = np.linalg.solve(A, b)
                
            if self.verbose:
                print(f"[TimeIALS] iter {it + 1}/{self.n_iter}")
                
        self.user_factors_ = P_fact
        self.item_factors_ = Q_fact
        return self

    def recommend(self, user_ids, k=10, exclude_seen=True, train_df=None):
        item_ids = np.array(list(self.item_index_.keys()))
        seen = {}
        if exclude_seen and train_df is not None:
            for u, grp in train_df.groupby("user_id"):
                seen[u] = set(grp["parent_asin"].tolist())
                
        result = {}
        for u in user_ids:
            ju = self.user_index_.get(u)
            if ju is None:
                result[u] = []
                continue
            
            # Скоры: скалярное произведение
            scores = self.user_factors_[ju] @ self.item_factors_.T
            
            if exclude_seen and u in seen:
                for it in seen[u]:
                    if it in self.item_index_:
                        scores[self.item_index_[it]] = -np.inf
                        
            top_local = np.argpartition(-scores, kth=min(k, len(scores)) - 1)[:k]
            top_local = top_local[np.argsort(-scores[top_local])]
            
            result[u] = [item_ids[i] for i in top_local if scores[i] > -np.inf]
            
        return result