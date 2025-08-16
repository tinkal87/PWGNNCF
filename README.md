# Modeling Feature Contextualized Pair-Wise Item Interactions in Graph Neural Network for Collaborative Filtering

This repository implements a Graph Neural Network (GNN) for collaborative filtering using pairwise item preferences and item-side features, tested on Gowalla, Yelp, Epinions, and Amazon Beauty datasets.

## Features

- Multi-layer GNN architecture
- Pairwise BPR loss for implicit feedback
- Genre/category-based item features
- Top-K recommendation generation
- Modular dataset support (Amazon-Beauty, Yelp, Gowalla, Epinions)

## Usage

```bash
python pwgnncf.py
