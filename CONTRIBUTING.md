# Contributing to ML-Cache Management System

Thank you for your interest in this project!

## Project Structure

The project is organized as follows:

```
ml-cache-management/
├── src/                    # All Python source code
├── docs/                   # Documentation and papers
├── assets/                 # Results, plots, and media
├── tests/                  # Unit tests (can be extended)
└── README.md              # Main documentation
```

## Getting Started

### Clone and Install

```bash
git clone https://github.com/chillmatin/ml-cache-management.git
cd ml-cache-management
pip install -r requirements.txt
```

### Run Experiments

```bash
python src/run_final_experiments.py
```

Results will be saved to `assets/results/`.

## Folder Organization

### `src/` - Source Code
- **cache_policies.py**: Base LRU and LFU implementations
- **cache_enhanced.py**: ML cache with 8 features
- **cache_hybrid.py**: Hybrid ML-LFU approach
- **ml_model_enhanced.py**: Gradient Boosting model
- **traffic_gen.py**: Workload generator
- **simulator.py**: Cache simulation framework
- **run_final_experiments.py**: Main experiment runner

### `docs/` - Documentation
- **ieee/**: IEEE-formatted papers (paper.tex, report.tex)
- **guides/**: Implementation guides and checklists

### `assets/` - Results
- **results/**: Generated plots (PNG) and data (JSON)

## How to Extend

### Add New Cache Policy

```python
from src.cache_policies import CachePolicy

class MyCache(CachePolicy):
    def __init__(self, capacity):
        super().__init__(capacity)
    
    def get(self, key):
        # Implement get logic
        pass
    
    def put(self, key, size=1):
        # Implement put logic
        pass
    
    def evict(self):
        # Implement eviction logic
        pass
```

Then test in `run_final_experiments.py`.

### Add New Workload

```python
from src.traffic_gen import TrafficGenerator

gen = TrafficGenerator(num_contents=1000, cache_size=100)
requests = gen.generate_hybrid_workload(duration=10000)
```

See `traffic_gen.py` for available generation methods.

## Testing

Run the experiments to validate:

```bash
python src/run_final_experiments.py
```

Check that:
- Plots are generated in `assets/results/`
- Results JSON files are created
- No errors are raised

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Bug Reports

If you find a bug, please create an issue with:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Your environment (Python version, OS, dependencies)

## Citation

If you use this code in your research:

```bibtex
@project{cache2024,
  title={Machine Learning-Based Cache Management for Edge Networks},
  author={Huseynzade, Matin},
  year={2024},
  institution={Izmir Institute of Technology}
}
```

## Questions?

Feel free to open an issue or contact the authors.

---

**Happy researching!** 🚀
