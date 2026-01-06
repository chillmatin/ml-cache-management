# ML-Based Cache Management System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

A research implementation of machine learning-driven cache replacement policies for content delivery networks. Demonstrates intelligent cache management using Gradient Boosting alongside traditional LRU/LFU policies.

## 📊 Project Highlights

| Metric | Value |
|--------|-------|
| **ML Model Accuracy** | 83.3% (test), 100% (train) |
| **LFU Hit Rate** | 43.31% ± 5.84% |
| **LRU Hit Rate** | 34.56% ± 5.30% |
| **ML-Cache Hit Rate** | 31.01% ± 4.35% |
| **Top Feature** | Inter-arrival Time (55.5%) |
| **Workload** | Zipfian α=1.2, 1000 contents |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd ml-cache-management

# Install dependencies
pip install numpy scikit-learn matplotlib seaborn
```

### Run Experiments

```bash
# Run main evaluation (generates all plots and statistics)
python src/run_final_experiments.py

# Results saved to assets/results/
```

## 📁 Project Structure

```
ml-cache-management/
├── src/                          # Python source code
│   ├── cache_policies.py         # LRU, LFU implementations
│   ├── cache_enhanced.py         # Enhanced ML cache with 8 features
│   ├── cache_hybrid.py           # Hybrid ML-LFU approach
│   ├── traffic_gen.py            # CDN workload generation
│   ├── ml_model.py               # Random Forest baseline
│   ├── ml_model_enhanced.py      # Gradient Boosting model
│   ├── simulator.py              # Discrete-event simulator
│   ├── run_experiments.py        # Main evaluation runner
│   ├── run_final_experiments.py  # Enhanced experiments
│   ├── quick_reference.py        # Quick result summary
│   └── PROJECT_COMPLETE.md       # Implementation checklist
│
├── docs/                         # Documentation
│   └── ieee/                     # IEEE-style papers
│       ├── paper.tex             # Conference paper (5 pages)
│       ├── report.tex            # Full project report (15 pages)
│       ├── proposal.tex          # Original proposal
│       └── gannt.tex             # Gantt chart
│       
├── assets/                       # Results and media
│   ├── results/                  # Experimental results
│   │   ├── *.png                 # 7 publication-quality plots
│   │   ├── *.json                # Raw experimental data
│   │   └── summary.txt           # Results summary
│   │
│   └── Matin-Huseynzade-Network-Proposal.pdf
│
│
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── LICENSE                       # MIT License
```

## 🔬 Core Algorithms

### Cache Policies

**LRU (Least Recently Used)**
- Time complexity: O(1) per operation
- Tracks access recency via OrderedDict
- Strong performance on temporal patterns

**LFU (Least Frequently Used)**
- Time complexity: O(1) per operation  
- Counts total access frequency
- Best performer on Zipfian workloads (43.31% hit rate)

**ML-Cache (Gradient Boosting)**
- Binary classifier: "Will content be re-accessed?"
- 8 features: recency, frequency, inter-arrival statistics
- Test accuracy: 83.3%
- Prediction-based eviction policy

### Workload Model

```
Traffic = Zipfian(α=1.2) + Poisson(λ=10) + Temporal Patterns
```

- **1,000 unique content objects**
- **100-unit cache capacity**
- **Realistic CDN simulation** with inter-arrival statistics
- **Temporal modulation** for daily cycles and flash crowds

## 📈 Experimental Evaluation

### Methodology

1. **Workload Generation**: Hybrid Zipfian + Poisson arrivals
2. **Training Phase**: First 50% of trace (1,014 requests)
3. **Evaluation**: 5 independent runs for statistical significance
4. **Metrics**: Hit rate, latency, prediction accuracy

### Key Findings

1. **LFU dominates** on Zipfian workloads (expected frequency signal)
2. **ML model achieves 83.3% accuracy** but hits lower due to:
   - Strong frequency bias in Zipfian distribution
   - Simple LFU already captures Zipf well
3. **Temporal patterns matter**: inter-arrival time is 55.5% important
4. **Hybrid approaches** promising for mixed workloads (future work)

### Results Files

- `assets/results/hit_rate_comparison.png` - Cache performance
- `assets/results/feature_importance.png` - ML feature rankings
- `assets/results/confusion_matrix.png` - ML model evaluation
- `assets/results/final_hit_rate_comparison.png` - Statistical results
- `assets/results/final_results.json` - Raw metrics
- `assets/results/ml_metrics.json` - Model performance
- `assets/results/batch_results.json` - Batch statistics

## 💻 Usage Examples

### Run Experiments

```python
from src.simulator import CacheSimulator, BatchSimulator
from src.traffic_gen import TrafficGenerator
from src.cache_policies import LRUCache, LFUCache

# Generate realistic workload
gen = TrafficGenerator(num_contents=1000, cache_size=100)
requests = gen.generate_hybrid_workload(duration=10000)

# Evaluate LFU
cache = LFUCache(capacity=100)
sim = CacheSimulator(cache)
metrics = sim.run(requests)
print(f"Hit Rate: {metrics['hit_rate']:.2%}")

# Statistical validation
batch = BatchSimulator(LFUCache(capacity=100), num_runs=5)
results = batch.run_batch(requests)
print(f"Mean Hit Rate: {results['mean_hit_rate']:.2%}")
print(f"Std Dev: {results['std_hit_rate']:.2%}")
```

### Get Key Metrics

```bash
python src/quick_reference.py
```

## 🔧 Implementation Details

### Machine Learning Pipeline

1. **Feature Extraction** (8 features):
   - Recency: Time since last access
   - Frequency: Total access count
   - Mean inter-arrival: Average time between accesses
   - Variance inter-arrival: Temporal pattern variance
   - Size: Content size in cache units
   - Normalized recency: [0,1] scale
   - Frequency rank: Percentile ranking
   - Time since first: Lifetime in cache

2. **Model Training**:
   - Algorithm: Gradient Boosting (200 estimators, depth=5)
   - Target: Binary (re-accessed in next 50 requests)
   - Train/Test: 70/30 split
   - Cross-validation: 5-fold

3. **Prediction & Eviction**:
   - Predict probability for all cached items
   - Evict lowest probability item
   - Retrain every 500 requests

## 📊 Reproducibility

All experiments are reproducible:

```bash
# Reset results
rm -rf assets/results/*.png assets/results/*.json

# Re-run experiments
python src/run_final_experiments.py

# Compare with previous results
diff assets/results/final_results.json assets/results/final_results.json.bak
```

Random seeds are fixed in all simulators for deterministic results.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@project{cache2024,
  title={Machine Learning-Based Cache Management for Edge Networks},
  author={Huseynzade, Matin},
  year={2026},
  institution={Izmir Institute of Technology}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ⚡ Performance Tips

1. **Reduce workload size** for faster iteration:
   ```python
   gen.generate_hybrid_workload(duration=1000)  # instead of 10000
   ```

2. **Disable visualization** for speed:
   Comment out `plt.show()` in experiment runners

3. **Parallel batch runs** with multiprocessing:
   See `BatchSimulator` implementation for extension

## 🐛 Troubleshooting

**ImportError: No module named 'sklearn'**
```bash
pip install scikit-learn
```

**Memory issues on large workloads**
- Reduce `duration` parameter in `generate_hybrid_workload()`
- Reduce `num_contents` in `TrafficGenerator`

**Plots not showing**
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

## 🔮 Future Work

1. **Online learning**: Adapt ML model during operation
2. **Hybrid policies**: Combine ML with LFU using ensemble methods
3. **Real traces**: Validate on CDN provider datasets
4. **ns-3 integration**: Network simulator validation
5. **Distributed caching**: Multi-tier cache hierarchies

## 📧 Contact

**Author**: Matin Huseynzade  
**Institution**: Izmir Institute of Technology

---

**Last Updated**: January 2026
