# gnn_cf_multi_datasets.py
# GNN-Based Collaborative Filtering with Pairwise Preferences
# Supports: movielens, yelp, amazon (Beauty), gowalla, epinions

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import itertools
import random

# ---------------------------
# DATA CONFIG (edit if needed)
# Place CSVs under data/<dataset>/
# ---------------------------
# movielens:
#   ratings: data/movielens/ratings.csv (userId,movieId,rating,...)
#   metadata: data/movielens/movies.csv (movieId, title, genres)
#
# yelp:
#   ratings: data/yelp/yelp_reviews.csv (user_id,business_id,stars,...)
#   metadata: data/yelp/yelp_businesses.csv (business_id, categories,...)
#
# amazon (Beauty):
#   ratings: data/amazon/beauty_reviews.csv (reviewerID,asin,overall,... or rating)
#   metadata: data/amazon/beauty_meta.csv (asin, categories,...)
#
# gowalla:
#   ratings: data/gowalla/gowalla.csv (user_id,poi_id,timestamp)
#   metadata: optional data/gowalla/poi_meta.csv (poi_id, categories)
#
# epinions:
#   ratings: data/epinions/epinions_reviews.csv (user_id,item_id,rating,...)
#   metadata: optional data/epinions/epinions_meta.csv (item_id, categories)
#
# ---------------------------

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(name):
    """
    Load interactions and metadata for dataset `name`.
    Returns:
      interactions (DataFrame) with columns ['user','item'] (encoded ints)
      metadata (DataFrame) with column 'item' (encoded ints) and metadata column
      num_users, num_items, meta_col (string)
    """
    base = os.path.join("data", name)
    if name == "movielens":
        ratings_path = os.path.join(base, "ratings.csv")
        metadata_path = os.path.join(base, "movies.csv")
        interactions_raw = pd.read_csv(ratings_path)
        metadata = pd.read_csv(metadata_path)
        # treat rating >=4 as positive
        if "rating" in interactions_raw.columns:
            interactions_raw = interactions_raw[interactions_raw["rating"] >= 4.0]
        user_col, item_col, meta_col = "userId", "movieId", "genres"

    elif name == "yelp":
        ratings_path = os.path.join(base, "yelp_reviews.csv")
        metadata_path = os.path.join(base, "yelp_businesses.csv")
        interactions_raw = pd.read_csv(ratings_path)
        metadata = pd.read_csv(metadata_path)
        if "stars" in interactions_raw.columns:
            interactions_raw = interactions_raw[interactions_raw["stars"] >= 4.0]
        user_col, item_col, meta_col = "user_id", "business_id", "categories"

    elif name == "amazon":
        ratings_path = os.path.join(base, "beauty_reviews.csv")
        metadata_path = os.path.join(base, "beauty_meta.csv")
        interactions_raw = pd.read_csv(ratings_path)
        metadata = pd.read_csv(metadata_path)
        # many Amazon files use "overall" or "rating"
        rating_col = "overall" if "overall" in interactions_raw.columns else ("rating" if "rating" in interactions_raw.columns else None)
        if rating_col:
            interactions_raw = interactions_raw[interactions_raw[rating_col] >= 4.0]
        user_col, item_col, meta_col = "reviewerID", "asin", "categories"

    elif name == "gowalla":
        ratings_path = os.path.join(base, "gowalla.csv")
        metadata_path = os.path.join(base, "poi_meta.csv")  # optional
        interactions_raw = pd.read_csv(ratings_path)
        metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
        # Gowalla check-in may not have ratings -> treat all interactions as positive
        user_col, item_col, meta_col = "user_id", "poi_id", "categories"

    elif name == "epinions":
        ratings_path = os.path.join(base, "epinions_reviews.csv")
        metadata_path = os.path.join(base, "epinions_meta.csv")  # optional
        interactions_raw = pd.read_csv(ratings_path)
        metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
        rating_col = "rating" if "rating" in interactions_raw.columns else None
        if rating_col:
            interactions_raw = interactions_raw[interactions_raw[rating_col] >= 4.0]
        user_col, item_col, meta_col = "user_id", "item_id", "categories"

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # Ensure columns exist
    if user_col not in interactions_raw.columns or item_col not in interactions_raw.columns:
        raise ValueError(f"Expect columns {user_col},{item_col} in interactions for {name}")

    # encode users and items (fit on interaction set)
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    interactions_raw[user_col] = interactions_raw[user_col].astype(str)
    interactions_raw[item_col] = interactions_raw[item_col].astype(str)

    interactions_raw["user"] = user_encoder.fit_transform(interactions_raw[user_col])
    interactions_raw["item"] = item_encoder.fit_transform(interactions_raw[item_col])

    num_users = interactions_raw["user"].nunique()
    num_items = interactions_raw["item"].nunique()

    # map metadata items to encoded ids safely; if metadata missing or item unseen, set -1
    if metadata is None or metadata.empty or meta_col not in metadata.columns or item_col not in metadata.columns:
        # create empty metadata with encoded item ids only
        metadata_encoded = pd.DataFrame({ "item": list(range(num_items)) })
    else:
        # Ensure metadata item ids are strings to match encoder
        metadata[item_col] = metadata[item_col].astype(str)
        def map_item_to_idx(raw_item):
            try:
                return int(item_encoder.transform([raw_item])[0])
            except Exception:
                return -1
        # apply mapping
        metadata["item"] = metadata[item_col].apply(map_item_to_idx)
        # keep only rows with a valid mapped item
        metadata_encoded = metadata[metadata["item"] >= 0].copy()
        # ensure the meta_col exists; if not, create empty
        if meta_col not in metadata_encoded.columns:
            metadata_encoded[meta_col] = metadata_encoded[item_col].astype(str)

    # reduce interactions to required columns and return
    interactions = interactions_raw[["user", "item"]].reset_index(drop=True)
    return interactions, metadata_encoded, num_users, num_items, meta_col

# Build item feature matrix (binary bag-of-features from meta_col)
def build_item_features(metadata, num_items, meta_col):
    # metadata expected to have columns: 'item' (encoded index), and meta_col with pipe-delimited tokens
    if metadata is None or metadata.empty or meta_col not in metadata.columns:
        # no metadata: return zero feature vector of dimension 1
        return torch.zeros((num_items, 1), dtype=torch.float32)

    # collect features
    feature_set = set()
    for val in metadata[meta_col].fillna("").astype(str):
        for tok in val.split("|"):
            tok = tok.strip()
            if tok:
                feature_set.add(tok)
    if len(feature_set) == 0:
        return torch.zeros((num_items, 1), dtype=torch.float32)

    feature_list = sorted(feature_set)
    feature_encoder = {g: idx for idx, g in enumerate(feature_list)}
    feat_dim = len(feature_list)
    item_features = torch.zeros((num_items, feat_dim), dtype=torch.float32)

    # fill features for items present in metadata
    for _, row in metadata.iterrows():
        i = int(row["item"])
        if i < 0 or i >= num_items:
            continue
        tokens = str(row[meta_col]).split("|")
        for tok in tokens:
            tok = tok.strip()
            if tok in feature_encoder:
                item_features[i, feature_encoder[tok]] = 1.0
    return item_features

# Pairwise data generation with safety checks
def generate_pairwise_data(interactions, neg_per_pos=5, seed=RANDOM_SEED):
    user_item_dict = defaultdict(set)
    for _, row in interactions.iterrows():
        user_item_dict[int(row["user"])].add(int(row["item"]))

    pairwise_data = []
    all_items = set(interactions["item"].unique())
    rng = np.random.default_rng(seed)
    for user, pos_items in user_item_dict.items():
        neg_items = np.array(list(all_items - pos_items))
        if len(neg_items) == 0:
            continue
        for i in pos_items:
            k = min(len(neg_items), neg_per_pos)
            # sample without replacement
            neg_samples = rng.choice(neg_items, size=k, replace=False)
            for j in neg_samples:
                pairwise_data.append((user, int(i), int(j)))
    return pairwise_data

# Dataset
class PairwiseDataset(Dataset):
    def __init__(self, pairwise_data):
        self.data = pairwise_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u,i,j = self.data[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)

# Model
class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, item_feat_dim, embed_dim=64, num_layers=2):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.item_feat_proj = nn.Linear(item_feat_dim, embed_dim)
        self.gnn_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])

        # initialization
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.item_feat_proj.weight)

    def forward(self, user, item_i, item_j, item_features):
        # user: (B,), item_i: (B,), item_j: (B,)
        u_e = self.user_emb(user)                      # (B,d)
        # item_features: (num_items, feat_dim)
        # index into features:
        i_feat = item_features[item_i].to(u_e.device) # (B,feat_dim)
        j_feat = item_features[item_j].to(u_e.device)
        i_e = self.item_emb(item_i) + self.item_feat_proj(i_feat)
        j_e = self.item_emb(item_j) + self.item_feat_proj(j_feat)

        for layer in self.gnn_layers:
            i_e = F.relu(layer(i_e))
            j_e = F.relu(layer(j_e))

        score_i = torch.sum(u_e * i_e, dim=1)
        score_j = torch.sum(u_e * j_e, dim=1)
        return score_i, score_j

    def get_top_k(self, user_id, item_features, k=10):
        # returns numpy indices of top-k items for user_id (int)
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            u = torch.tensor([user_id], dtype=torch.long, device=device)
            u_e = self.user_emb(u).squeeze(0)                   # (d,)
            all_items = torch.arange(self.item_emb.num_embeddings, device=device)
            feats = item_features.to(device)
            i_e = self.item_emb(all_items) + self.item_feat_proj(feats)
            for layer in self.gnn_layers:
                i_e = F.relu(layer(i_e))
            # scores: (num_items,)
            scores = torch.matmul(i_e, u_e)
            topk = torch.topk(scores, k=min(k, scores.numel())).indices
            return topk.cpu().numpy()

# BPR Loss
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score_i, score_j):
        return -torch.log(torch.sigmoid(score_i - score_j) + 1e-10).mean()

# Training loop
def train(model, dataloader, item_features, optimizer, criterion, epochs=10, device=device):
    model.to(device)
    item_features = item_features.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            user, i, j = [t.to(device) for t in batch]   # tensors
            optimizer.zero_grad()
            score_i, score_j = model(user, i, j, item_features)
            loss = criterion(score_i, score_j)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * user.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

# Hyperparameter grid search (example)
def run_experiments(dataset_name):
    print(f"\n=== Dataset: {dataset_name} ===")
    interactions, metadata, num_users, num_items, meta_col = load_dataset(dataset_name)
    print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(interactions)}")
    pairwise_data = generate_pairwise_data(interactions, neg_per_pos=5, seed=RANDOM_SEED)
    if len(pairwise_data) == 0:
        print("No pairwise data generated; skipping.")
        return
    dataset = PairwiseDataset(pairwise_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    item_features = build_item_features(metadata, num_items, meta_col)
    # if item_features has single dim and is zero, that's fine

    # quick grid for demonstration (adjust as needed)
    embed_dims = [32, 64]
    num_layers_list = [1, 2]
    lrs = [0.001, 0.0005]

    for embed_dim, num_layers, lr in itertools.product(embed_dims, num_layers_list, lrs):
        print(f"\nConfig: embed_dim={embed_dim}, layers={num_layers}, lr={lr}")
        model = GNNRecommender(num_users, num_items, item_feat_dim=item_features.shape[1], embed_dim=embed_dim, num_layers=num_layers)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = BPRLoss()
        train(model, dataloader, item_features, optimizer, criterion, epochs=5, device=device)

        # top-k example for user 0 (if exists)
        try:
            top_k_items = model.get_top_k(user_id=0, item_features=item_features, k=10)
            print(f"Top-10 for user 0: {top_k_items}")
        except Exception as e:
            print("Top-k generation failed:", e)

if __name__ == "__main__":
    for ds in ["movielens", "yelp", "amazon", "gowalla", "epinions"]:
        try:
            run_experiments(ds)
        except Exception as e:
            print(f"Error on dataset {ds}: {e}")
