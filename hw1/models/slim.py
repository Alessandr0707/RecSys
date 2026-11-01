import numpy as np
import scipy.sparse as sp

class SLIM:
    def __init__(self, l1_reg=0.1, l2_reg=0.1, max_iter=10, learning_rate=0.01):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.user_index_ = None
        self.item_index_ = None
        self.idx_to_item = None
        self.W = None
        self.rating_matrix_ = None

    def fit(self, train_df):
        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique()
        
        self.user_index_ = {u: i for i, u in enumerate(users)}
        self.item_index_ = {it: i for i, it in enumerate(items)}
        self.idx_to_item = {i: it for it, i in self.item_index_.items()}

        rows = train_df["user_id"].map(self.user_index_).to_numpy()
        cols = train_df["parent_asin"].map(self.item_index_).to_numpy()
        data = np.ones(len(train_df), dtype=np.float32)
        
        n_u = len(self.user_index_)
        n_i = len(self.item_index_)

        X = sp.csr_matrix((data, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
        self.rating_matrix_ = X
        X_T = X.T.tocsr()

        self.W = np.zeros((n_i, n_i), dtype=np.float32)

        print("Обучение SLIM (это может занять несколько минут)...")
        
        for j in range(n_i):
            y = X[:, j].toarray().flatten()
            if np.sum(y) == 0:
                continue
                
            w = np.zeros(n_i, dtype=np.float32)
            
            for it in range(self.max_iter):
                pred = X @ w
                error = pred - y
                
                grad = X_T @ error + self.l2_reg * w
                w = w - self.learning_rate * grad
                
                w = np.sign(w) * np.maximum(np.abs(w) - self.learning_rate * self.l1_reg, 0)
                w = np.maximum(w, 0)
                w[j] = 0 
                
            self.W[:, j] = w
            
            if (j + 1) % 5000 == 0:
                print(f"Обработано {j + 1} / {n_i} объектов...")
                
        print("Обучение SLIM завершено.")
        return self

    def recommend(self, user_ids, k=10, exclude_seen=True, train_df=None):
        seen = {}
        if exclude_seen and train_df is not None:
            for u, grp in train_df.groupby("user_id"):
                seen[u] = set(grp["parent_asin"].tolist())

        item_ids = np.array(list(self.item_index_.keys()))
        result = {}
        
        for u in user_ids:
            ju = self.user_index_.get(u)
            if ju is None:
                result[u] = []
                continue
            
            user_row = self.rating_matrix_.getrow(ju)
            # ИСПРАВЛЕНО: используем flatten() вместо toarray()
            scores = (user_row @ self.W).flatten()
            
            if exclude_seen and u in seen:
                for it in seen[u]:
                    if it in self.item_index_:
                        scores[self.item_index_[it]] = -np.inf
                        
            top_k_idx = np.argpartition(-scores, kth=k-1)[:k]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
            
            result[u] = [item_ids[i] for i in top_k_idx if scores[i] > -np.inf]
            
        return result