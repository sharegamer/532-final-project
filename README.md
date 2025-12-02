# MIND Dataset Recommendation System

A comprehensive hybrid recommendation system implementing **CBF (TF-IDF + Entity) + CF (ALS) + Knowledge-Aware** approaches with Spark optimization.

## Features

✅ **Multiple Recommendation Approaches**
- Content-Based Filtering (CBF) with TF-IDF
- Content-Based Filtering (CBF) with Entity Embeddings
- Collaborative Filtering (CF) using ALS
- Knowledge-Aware recommendations using entity/relation embeddings
- Hybrid model combining all approaches

✅ **Spark Optimization**
- Distributed data processing
- Parallel feature extraction
- Scalable ALS training with MLlib
- Performance benchmarking vs. Pandas

✅ **Comprehensive Evaluation**
- Ranking metrics: NDCG@5, NDCG@10, MRR
- Classification metrics: AUC, Accuracy, Precision, Recall
- Diversity and coverage analysis
- Model comparison framework

✅ **Interactive UI**
- User-friendly recommendation display
- Real-time model comparison
- Reading history visualization
- Interactive parameter tuning

✅ **Model Persistence**
- Save/load all trained models
- Reproducible results

## Project Structure

```
project/
├── main.ipynb                          # Main Jupyter notebook
├── src/
│   ├── data_loader.py                 # Spark-based data loading
│   ├── preprocessing.py               # Feature extraction
│   ├── models/
│   │   ├── cbf_tfidf.py              # TF-IDF recommender
│   │   ├── cbf_entity.py             # Entity recommender
│   │   ├── cf_als.py                 # ALS recommender
│   │   ├── knowledge_aware.py        # Knowledge-aware recommender
│   │   └── hybrid.py                 # Hybrid recommender
│   ├── evaluation.py                  # Evaluation metrics
│   ├── utils.py                       # Utility functions
│   └── ui.py                          # Interactive UI
├── tests/
│   └── test_models.py                 # Unit tests
├── models/                            # Saved models (created at runtime)
├── results/                           # Results and visualizations
├── data/                              # MIND dataset
│   ├── MINDsmall_train/
│   └── MINDsmall_dev/
├── requirements.txt
├── IMPLEMENTATION_PLAN.md
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Dataset

The MIND dataset should already be in the `data/` directory:
- `data/MINDsmall_train/` - Training data
- `data/MINDsmall_dev/` - Development/test data

Each directory should contain:
- `behaviors.tsv` - User click history
- `news.tsv` - News articles
- `entity_embedding.vec` - Entity embeddings
- `relation_embedding.vec` - Relation embeddings

### 3. Run the Notebook

```bash
jupyter notebook main.ipynb
```

Or use JupyterLab:
```bash
jupyter lab main.ipynb
```

## Usage

### Quick Start

1. Open `main.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load and preprocess data
   - Train all models
   - Evaluate performance
   - Display interactive UI
   - Save trained models

### Running Tests

```bash
cd tests
python test_models.py
```

## Models

### 1. TF-IDF Recommender (CBF)
- Uses **Spark MLlib HashingTF and IDF** for scalable TF-IDF computation
- Can handle millions of articles without memory issues
- Extracts TF-IDF features from news titles and abstracts
- Computes cosine similarity between articles
- Recommends based on content similarity to user history

### 2. Entity Recommender (CBF)
- Uses pre-trained entity embeddings from knowledge graph
- Aggregates entity vectors per article
- Recommends based on entity similarity

### 3. ALS Recommender (CF)
- Collaborative filtering using Spark MLlib ALS
- Learns latent factors for users and news articles
- Distributed training for scalability

### 4. Knowledge-Aware Recommender
- Leverages entity and relation embeddings
- Incorporates knowledge graph structure
- Combines entity similarity with overlap scoring

### 5. Hybrid Recommender
- Combines all four approaches with weighted scores
- Configurable weights for each component
- Optimal balance of different recommendation signals

## Evaluation Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Precision@k**: Precision at k
- **Recall@k**: Recall at k
- **AUC**: Area Under ROC Curve
- **Accuracy**: Classification accuracy
- **Catalog Coverage**: Percentage of items recommended
- **Diversity**: Uniqueness of recommendations

## Spark Optimization

The system demonstrates Spark optimization in:

1. **Data Loading**: Parallel reading of TSV files
2. **TF-IDF Feature Extraction**: Uses HashingTF and IDF from Spark MLlib for scalable text vectorization
3. **Preprocessing**: Distributed feature extraction
4. **ALS Training**: Distributed matrix factorization
5. **Batch Prediction**: Parallel recommendation generation

**Key Advantage**: Spark's HashingTF and IDF keep vectors in Spark, allowing you to featurize millions of articles without crashing RAM. The vectors are only collected to the driver after distributed computation.

## Configuration

### Hybrid Model Weights

Adjust weights in the notebook:

```python
hybrid_recommender.set_weights(
    tfidf=0.25,      # TF-IDF weight
    entity=0.25,     # Entity weight
    als=0.3,         # ALS weight
    knowledge=0.2    # Knowledge-aware weight
)
```

### ALS Parameters

Tune in the notebook:

```python
als_recommender.train(
    train_df,
    rank=10,         # Latent factors
    max_iter=10,     # Iterations
    reg_param=0.1    # Regularization
)
```

## Results

After running the notebook, results are saved to `results/`:
- `evaluation_report.csv` - Performance metrics for all models
- `metrics_comparison.png` - Visualization comparing models
- Recommendation outputs in JSON format

Models are saved to `models/` for later reuse.

## Key Findings

The hybrid approach typically achieves the best performance by:
- Leveraging content similarity (TF-IDF + Entity)
- Capturing collaborative patterns (ALS)
- Incorporating knowledge graph structure
- Balancing different recommendation signals

## References

- MIND Dataset: https://msnews.github.io/
- Spark MLlib: https://spark.apache.org/mllib/
- ALS: Collaborative Filtering for Implicit Feedback Datasets

## License
We used Claude Code in VSCode to assist with coding in this assignment.
This project is for educational purposes.
