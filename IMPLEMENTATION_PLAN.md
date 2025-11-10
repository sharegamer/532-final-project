# MIND Dataset Recommendation System - Implementation Plan

## Project Overview
Build a hybrid recommendation system using the MIND (Microsoft News Dataset) with three approaches:
1. **CBF (Content-Based Filtering)**: TF-IDF + Entity Embeddings
2. **CF (Collaborative Filtering)**: ALS (Alternating Least Squares)
3. **Knowledge-Aware**: Utilizing entity and relation embeddings
4. **Spark Optimization**: For efficient data processing

## Dataset Structure
- **Train**: `data/MINDsmall_train/`
  - `behaviors.tsv`: User click history and impressions
  - `news.tsv`: News article metadata (title, abstract, category, entities)
  - `entity_embedding.vec`: Pre-trained entity embeddings
  - `relation_embedding.vec`: Pre-trained relation embeddings

- **Dev**: `data/MINDsmall_dev/` (same structure for validation)

## Architecture Design

### Phase 1: Data Processing & Preparation
1. Load and parse TSV files using Spark
2. Create user-item interaction matrix
3. Extract and preprocess text features (titles, abstracts)
4. Load entity/relation embeddings
5. Build feature matrices for each recommendation approach

### Phase 2: Model Implementation

#### 2.1 Content-Based Filtering (CBF)
- **TF-IDF Component**:
  - Extract TF-IDF features from news titles + abstracts
  - Compute cosine similarity between news articles
  - Generate recommendations based on user history

- **Entity Component**:
  - Use pre-trained entity embeddings
  - Aggregate entity embeddings per article
  - Compute similarity using entity vectors

#### 2.2 Collaborative Filtering (CF)
- Implement ALS using Spark MLlib
- Create user-news rating matrix from click data
- Train ALS model with hyperparameter tuning
- Generate collaborative recommendations

#### 2.3 Knowledge-Aware Recommendation
- Leverage entity and relation embeddings
- Implement knowledge graph-based scoring
- Combine structural knowledge with user preferences

#### 2.4 Hybrid Model
- Combine CBF, CF, and Knowledge-Aware scores
- Weighted ensemble or learned combination
- Optimize weights for best performance

### Phase 3: Spark Optimization
1. Demonstrate processing time with/without Spark
2. Use Spark for:
   - Parallel data loading and preprocessing
   - Distributed TF-IDF computation
   - ALS training (built-in distributed)
   - Batch prediction generation
3. Benchmark and visualize speedup

### Phase 4: Evaluation
Implement multiple metrics:
- **Ranking Metrics**: NDCG@5, NDCG@10, MRR
- **Classification Metrics**: AUC, Accuracy
- **Coverage**: Catalog coverage, diversity
- **A/B Comparison**: Compare each method individually and combined

### Phase 5: Model Persistence
- Save trained models:
  - TF-IDF vectorizer
  - ALS model
  - Feature matrices
  - Hybrid model weights
- Implement load/reload functionality

### Phase 6: UI & Visualization
Create interactive interface showing:
- Top-K recommendations for sample users
- Comparison across different methods
- Evaluation metrics visualization
- Real-time recommendation demo (optional)

### Phase 7: Testing & Validation
- Unit tests for data loading
- Integration tests for each model component
- Parameter sensitivity analysis
- Edge case handling

## File Structure
```
project/
├── main.ipynb                          # Main Jupyter notebook
├── src/
│   ├── data_loader.py                 # Spark-based data loading
│   ├── preprocessing.py               # Feature extraction and preprocessing
│   ├── models/
│   │   ├── cbf_tfidf.py              # TF-IDF based CBF
│   │   ├── cbf_entity.py             # Entity-based CBF
│   │   ├── cf_als.py                 # ALS collaborative filtering
│   │   ├── knowledge_aware.py        # Knowledge graph-based model
│   │   └── hybrid.py                 # Hybrid recommendation
│   ├── evaluation.py                  # Metrics and evaluation
│   ├── utils.py                       # Helper functions
│   └── ui.py                          # UI components
├── tests/
│   └── test_models.py                 # Test cases
├── models/                            # Saved models directory
├── results/                           # Output results and visualizations
└── data/                              # MIND dataset
```

## Implementation Timeline
1. ✅ Dataset already downloaded
2. Data loading and preprocessing (Spark)
3. CBF implementation (TF-IDF + Entity)
4. CF implementation (ALS)
5. Knowledge-Aware implementation
6. Hybrid model integration
7. Evaluation framework
8. Model saving/loading
9. UI development
10. Testing and parameter tuning
11. Spark optimization demonstration
12. Final documentation and demo

## Key Dependencies
- PySpark (MLlib for ALS, core for distributed processing)
- scikit-learn (TF-IDF, metrics)
- pandas, numpy
- matplotlib, seaborn (visualization)
- jupyter notebook/lab
- ipywidgets (interactive UI)

## Success Criteria
- ✅ All three recommendation approaches working
- ✅ Spark optimization demonstrated with benchmarks
- ✅ Multiple evaluation metrics computed
- ✅ Models saved and loadable
- ✅ Clear, user-friendly UI
- ✅ Comprehensive test coverage
- ✅ Clean, documented code
