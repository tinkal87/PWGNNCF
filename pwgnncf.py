# GNN-Based Collaborative Filtering with Pairwise Preferences for MovieLens, Yelp, and Amazon Beauty

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import itertools
import os

# Load datasets (MovieLens, Yelp, Amazon Beauty)
def load_dataset(name):
    if name == "movielens":
        ratings = pd.read_csv("data/movielens/ratings.csv")
        metadata = pd.read_csv("data/movielens/movies.csv")
        interactions = ratings[ratings['rating'] >= 4.0]
        user_col, item_col, meta_col = 'userId', 'movieId', 'genres'
    elif name == "yelp":
        ratings = pd.read_csv("data/yelp/yelp_reviews.csv")
        metadata = pd.read_csv("data/yelp/yelp_businesses.csv")
        interactions = ratings[ratings['stars'] >= 4.0]
        user_col, item_col, meta_col = 'user_id', 'business_id', 'categories'
    elif name == "amazon":
        ratings = pd.read_csv("data/amazon/beauty_reviews.csv")
        metadata = pd.read_csv("data/amazon/beauty_meta.csv")
        interactions = ratings[ratings['rating'] >= 4.0]
        user_col, item_col, meta_col = 'reviewerID', 'asin', 'categories'
    else:
        raise ValueError("Unsupported dataset")

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    interactions['user'] = user_encoder.fit_transform(interactions[user_col])
    interactions['item'] = item_encoder.fit_transform(interactions[item_col])
    metadata['item'] = item_encoder.transform(metadata[item_col].fillna('unknown'))

    num_users = interactions['user'].nunique()
    num_items = interactions['item'].nunique()

    return interactions, metadata, num_users, num_items, meta_col

# Generate pairwise triplets
def generate_pairwise_data(interactions):
    user_item_dict = defaultdict(set)
    for _, row in interactions.iterrows():
        user_item_dict[row['user']].add(row['item'])

    pairwise_data = []
    all_items = set(interactions['item'].unique())
    for user, pos_items in user_item_dict.items():
        neg_items = all_items - pos_items
        for i in pos_items:
            neg_samples = np.random.choice(list(neg_items), size=min(len(neg_items), 5), replace=False)
            for j in neg_samples:
                pairwise_data.append((user, i, j))
    return pairwise_data

# Dataset for pairwise triplets
class PairwiseDataset(Dataset):
    def __init__(self, pairwise_data):
        self.data = pairwise_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Build item feature matrix
def build_item_features(metadata, num_items, meta_col):
    feature_set = set()
    for f in metadata[meta_col]:
        for g in str(f).split('|'):
            feature_set.add(g.strip())
    feature_list = sorted(list(feature_set))
    feature_encoder = {g: idx for idx, g in enumerate(feature_list)}

    item_features = torch.zeros((num_items, len(feature_list)))
    for _, row in metadata.iterrows():
        i = row['item']
        for g in str(row[meta_col]).split('|'):
            g = g.strip()
            if g in feature_encoder:
                item_features[i][feature_encoder[g]] = 1.0
    return item_features

# GNN-based recommender model
class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, item_feat_dim, embed_dim, num_layers):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.item_feat_proj = nn.Linear(item_feat_dim, embed_dim)
        self.gnn_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])

    def forward(self, user, item_i, item_j, item_features):
        u_e = self.user_emb(user)
        i_e = self.item_emb(item_i) + self.item_feat_proj(item_features[item_i])
        j_e = self.item_emb(item_j) + self.item_feat_proj(item_features[item_j])

        for layer in self.gnn_layers:
            i_e = F.relu(layer(i_e))
            j_e = F.relu(layer(j_e))

        score_i = (u_e * i_e).sum(dim=1)
        score_j = (u_e * j_e).sum(dim=1)
        return score_i, score_j

    def get_top_k(self, user_id, item_features, k=10):
        with torch.no_grad():
            u_e = self.user_emb(torch.tensor([user_id]))
            all_items = torch.arange(self.item_emb.num_embeddings)
            i_e = self.item_emb(all_items) + self.item_feat_proj(item_features)
            for layer in self.gnn_layers:
                i_e = F.relu(layer(i_e))
            scores = torch.matmul(i_e, u_e.squeeze())
            topk = torch.topk(scores, k=k).indices
            return topk.cpu().numpy()

# BPR loss
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score_i, score_j):
        return -torch.log(torch.sigmoid(score_i - score_j)).mean()

# Training loop
def train(model, dataloader, item_features, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            user, i, j = [x.long() for x in batch]
            optimizer.zero_grad()
            score_i, score_j = model(user, i, j, item_features)
            loss = criterion(score_i, score_j)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Hyperparameter grid search
def run_experiments(dataset_name):
    interactions, metadata, num_users, num_items, meta_col = load_dataset(dataset_name)
    pairwise_data = generate_pairwise_data(interactions)
    dataset = PairwiseDataset(pairwise_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    item_features = build_item_features(metadata, num_items, meta_col)

    embed_dims = [32, 64]
    num_layers_list = [1, 2, 3]
    lrs = [0.001, 0.0005]

    for embed_dim, num_layers, lr in itertools.product(embed_dims, num_layers_list, lrs):
        print(f"\nRunning {dataset_name} with embed_dim={embed_dim}, layers={num_layers}, lr={lr}")
        model = GNNRecommender(num_users, num_items, item_feat_dim=item_features.shape[1], embed_dim=embed_dim, num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = BPRLoss()
        train(model, dataloader, item_features, optimizer, criterion, epochs=5)

        # Example: Top-K recommendation for user 0
        top_k_items = model.get_top_k(user_id=0, item_features=item_features, k=10)
        print(f"Top-10 recommended items for user 0: {top_k_items}")

# Main
if __name__ == '__main__':
    for dataset in ["movielens", "yelp", "amazon"]:
        run_experiments(dataset)
