import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class FM(nn.Module):
    def __init__(self, n_features, k=16):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Embedding(n_features, 1)
        self.v = nn.Embedding(n_features, k)

        nn.init.normal_(self.v.weight, std=0.01)
        nn.init.zeros_(self.w.weight)

    def forward(self, x):
        w_sum = self.w0 + torch.sum(self.w(x).squeeze(-1), dim=1, keepdim=True)
        v = self.v(x)
        v_sum = torch.sum(v, dim=1)
        inter = 0.5 * torch.sum((v_sum ** 2) - torch.sum(v ** 2, dim=1), dim=1, keepdim=True)
        return w_sum + inter


class FMWrapper:
    def __init__(self, k=32, lr=0.01, epochs=5, batch_size=8192, device="cpu"):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.user_idx = {}
        self.item_idx = {}
        self.store_idx = {}
        self.cat_idx = {}
        self.price_idx = {}
        self.n_features = 0
        self.item_features_matrix = None

    def fit(self, train_df, neg_samples=3, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        train_df = train_df.copy()
        train_df["price_bin"] = (
            pd.qcut(train_df["price"], q=4, labels=False, duplicates="drop")
            .fillna(0)
            .astype(int)
        )

        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique()
        stores = train_df["store"].astype(str).unique()
        cats = train_df["main_category"].astype(str).unique()
        prices = train_df["price_bin"].unique()

        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {it: i + len(users) for i, it in enumerate(items)}
        offset = len(users) + len(items)
        self.store_idx = {s: i + offset for i, s in enumerate(stores)}
        offset += len(stores)
        self.cat_idx = {c: i + offset for i, c in enumerate(cats)}
        offset += len(cats)
        self.price_idx = {p: i + offset for i, p in enumerate(prices)}
        self.n_features = offset + len(prices)

        # Fallback indices used when a categorical value is unseen.
        unk_store_idx = offset - len(cats) - len(prices) - len(stores)
        unk_cat_idx = offset - len(prices) - len(cats)
        unk_price_idx = offset - len(prices)

        item_meta = (
            train_df[["parent_asin", "store", "main_category", "price_bin"]]
            .drop_duplicates("parent_asin")
            .set_index("parent_asin")
            .to_dict("index")
        )

        # Positive examples.
        X_pos = np.zeros((len(train_df), 5), dtype=np.int64)
        X_pos[:, 0] = train_df["user_id"].map(self.user_idx).values
        X_pos[:, 1] = train_df["parent_asin"].map(self.item_idx).values
        X_pos[:, 2] = train_df["store"].astype(str).map(self.store_idx).fillna(unk_store_idx).values
        X_pos[:, 3] = train_df["main_category"].astype(str).map(self.cat_idx).fillna(unk_cat_idx).values
        X_pos[:, 4] = train_df["price_bin"].map(self.price_idx).values
        y_pos = np.ones(len(train_df), dtype=np.float32)

        # Negative sampling: draw `neg_samples` random items per user,
        # skipping items the user has already interacted with.
        user_hist = train_df.groupby("user_id")["parent_asin"].apply(set).to_dict()
        items_array = np.array(items)

        X_neg_list, y_neg_list = [], []
        for u, hist in user_hist.items():
            u_idx = self.user_idx[u]
            for _ in range(neg_samples):
                neg_item = np.random.choice(items_array)
                if neg_item in hist:
                    continue
                info = item_meta[neg_item]
                s_idx = self.store_idx.get(str(info["store"]), unk_store_idx)
                c_idx = self.cat_idx.get(str(info["main_category"]), unk_cat_idx)
                p_idx = self.price_idx.get(info["price_bin"], unk_price_idx)
                X_neg_list.append([u_idx, self.item_idx[neg_item], s_idx, c_idx, p_idx])
                y_neg_list.append(0.0)

        X_neg = np.array(X_neg_list, dtype=np.int64)
        y_neg = np.array(y_neg_list, dtype=np.float32)

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        # Training.
        self.model = FM(self.n_features, self.k).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X, dtype=torch.long).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        print(f"Training FM on {len(X_t)} examples")
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_t), device=self.device)
            total_loss = 0.0
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                preds = self.model(X_t[idx])
                loss = loss_fn(preds, y_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / (len(X_t) / self.batch_size):.4f}")

        # Cache item features for inference.
        self.item_features_matrix = np.zeros((len(self.item_idx), 4), dtype=np.int64)
        for item, info in item_meta.items():
            idx = self.item_idx.get(item)
            if idx is None:
                continue
            s_idx = self.store_idx.get(str(info["store"]), unk_store_idx)
            c_idx = self.cat_idx.get(str(info["main_category"]), unk_cat_idx)
            p_idx = self.price_idx.get(info["price_bin"], unk_price_idx)
            self.item_features_matrix[idx - len(users)] = [idx, s_idx, c_idx, p_idx]

    def recommend(self, user_ids, train_df, k=10):
        self.model.eval()
        seen = train_df.groupby("user_id")["parent_asin"].apply(set).to_dict()
        result = {}

        with torch.no_grad():
            item_tensor = torch.tensor(self.item_features_matrix, dtype=torch.long).to(self.device)
            w_items = self.model.w(item_tensor).squeeze(-1).sum(dim=1)
            v_items_raw = self.model.v(item_tensor)
            v_item_total = v_items_raw.sum(dim=1)
            v_item_sq_total = (v_items_raw ** 2).sum(dim=1)

            inv_item_idx = {v - len(self.user_idx): name for name, v in self.item_idx.items()}

            for u in user_ids:
                if u not in self.user_idx:
                    result[u] = []
                    continue

                u_idx = self.user_idx[u]
                u_w = self.model.w(torch.tensor([u_idx], device=self.device))
                u_v = self.model.v(torch.tensor([u_idx], device=self.device))

                w_sum = self.model.w0 + u_w.squeeze(-1) + w_items
                v_sum = u_v + v_item_total
                v_sq_sum = u_v ** 2 + v_item_sq_total

                inter = 0.5 * torch.sum(v_sum ** 2 - v_sq_sum, dim=1)
                scores = (w_sum + inter).cpu().numpy()

                if u in seen:
                    for it in seen[u]:
                        if it in self.item_idx:
                            scores[self.item_idx[it] - len(self.user_idx)] = -np.inf

                top_k_idx = np.argpartition(-scores, kth=k - 1)[:k]
                top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

                result[u] = [inv_item_idx[i] for i in top_k_idx if scores[i] > -np.inf]

        return result