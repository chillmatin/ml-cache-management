"""
Cache simulator with event-driven request processing.
"""

from cache_policies import LRUCache, LFUCache, MLCache
from traffic_gen import Request
from typing import List, Dict
import json


class CacheSimulator:
    """
    Discrete-event simulator for cache performance evaluation.
    """
    
    def __init__(self, cache_policy, cache_capacity):
        """
        Args:
            cache_policy: Cache policy class (LRUCache, LFUCache, or MLCache)
            cache_capacity: Maximum cache size
        """
        self.policy_class = cache_policy
        self.capacity = cache_capacity
        self.cache = None
        self.metrics_history = []
        
    def run(self, requests: List[Request], collect_interval=100):
        """
        Run simulation on a trace of requests.
        
        Args:
            requests: List of Request objects
            collect_interval: Collect metrics every N requests
            
        Returns:
            Dictionary with final statistics and time series
        """
        # Initialize cache
        if self.policy_class == MLCache:
            # MLCache requires predictor - will be set separately
            self.cache = self.policy_class(self.capacity, predictor=None)
        else:
            self.cache = self.policy_class(self.capacity)
        
        self.metrics_history = []
        
        # Process requests
        for idx, req in enumerate(requests):
            # Process request
            hit, latency = self.cache.get(
                key=req.content_id,
                size=req.size,
                timestamp=req.timestamp
            )
            
            # Collect metrics periodically
            if (idx + 1) % collect_interval == 0:
                stats = self.cache.get_stats()
                stats['request_num'] = idx + 1
                stats['timestamp'] = req.timestamp
                self.metrics_history.append(stats)
        
        # Final statistics
        final_stats = self.cache.get_stats()
        final_stats['metrics_history'] = self.metrics_history
        
        return final_stats
    
    def set_predictor(self, predictor):
        """Set ML predictor for MLCache policy."""
        if self.cache is not None and hasattr(self.cache, 'predictor'):
            self.cache.predictor = predictor


def run_comparison(requests: List[Request], 
                   cache_capacity=100,
                   ml_predictor=None) -> Dict:
    """
    Run simulation with all cache policies for comparison.
    
    Args:
        requests: Request trace
        cache_capacity: Cache size
        ml_predictor: Trained ML model for MLCache
        
    Returns:
        Dictionary mapping policy names to results
    """
    results = {}
    
    policies = {
        'LRU': LRUCache,
        'LFU': LFUCache,
    }
    
    # Add MLCache if predictor is available
    if ml_predictor is not None:
        policies['ML-Cache'] = MLCache
    
    for name, policy_class in policies.items():
        print(f"Running {name} simulation...")
        sim = CacheSimulator(policy_class, cache_capacity)
        
        if name == 'ML-Cache':
            sim.cache = MLCache(cache_capacity, predictor=ml_predictor)
            stats = sim.run(requests)
        else:
            stats = sim.run(requests)
        
        results[name] = stats
        print(f"  Hit rate: {stats['hit_rate']:.3f}")
        print(f"  Avg latency: {stats['avg_latency']:.2f} ms")
        print(f"  Evictions: {stats['evictions']}")
    
    return results


def save_results(results: Dict, filepath: str):
    """Save simulation results to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    
    results_serializable = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """Load simulation results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


class BatchSimulator:
    """
    Run multiple simulations with different random seeds for statistical robustness.
    """
    
    def __init__(self, cache_capacity=100):
        self.capacity = cache_capacity
        
    def run_batch(self, workloads: List[List[Request]], 
                  ml_predictor=None) -> Dict:
        """
        Run simulations on multiple workloads.
        
        Args:
            workloads: List of request traces
            ml_predictor: Trained ML model
            
        Returns:
            Aggregated results with mean and std
        """
        all_results = {
            'LRU': [],
            'LFU': [],
        }
        
        if ml_predictor is not None:
            all_results['ML-Cache'] = []
        
        for idx, workload in enumerate(workloads):
            print(f"\n=== Workload {idx+1}/{len(workloads)} ===")
            results = run_comparison(workload, self.capacity, ml_predictor)
            
            for policy_name, stats in results.items():
                all_results[policy_name].append(stats)
        
        # Aggregate statistics
        aggregated = {}
        for policy_name, results_list in all_results.items():
            hit_rates = [r['hit_rate'] for r in results_list]
            latencies = [r['avg_latency'] for r in results_list]
            evictions = [r['evictions'] for r in results_list]
            
            aggregated[policy_name] = {
                'hit_rate_mean': sum(hit_rates) / len(hit_rates),
                'hit_rate_std': (sum((x - sum(hit_rates)/len(hit_rates))**2 for x in hit_rates) / len(hit_rates))**0.5,
                'latency_mean': sum(latencies) / len(latencies),
                'latency_std': (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies))**0.5,
                'evictions_mean': sum(evictions) / len(evictions),
                'evictions_std': (sum((x - sum(evictions)/len(evictions))**2 for x in evictions) / len(evictions))**0.5,
                'individual_results': results_list
            }
        
        return aggregated
