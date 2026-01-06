"""
Enhanced MLCache with better eviction strategy.
"""

from cache_policies import CachePolicy
from collections import defaultdict
import numpy as np


class EnhancedMLCache(CachePolicy):
    """
    Enhanced ML-based cache policy with hybrid approach.
    Combines ML predictions with LRU fallback and smarter eviction.
    """
    
    def __init__(self, capacity, predictor=None):
        super().__init__(capacity)
        self.cache = {}
        self.sizes = {}
        self.current_size = 0
        self.predictor = predictor
        
        # Track access history for feature extraction
        self.access_times = defaultdict(list)
        self.access_counts = defaultdict(int)
        self.first_access = {}
        self.all_access_counts = defaultdict(int)
        
    def _on_hit(self, key, timestamp):
        """Record access pattern."""
        self.access_times[key].append(timestamp)
        self.access_counts[key] += 1
        self.all_access_counts[key] += 1
    
    def _on_miss(self, key, size, timestamp):
        """Add new item, use enhanced ML model to decide eviction."""
        # Record access
        if key not in self.first_access:
            self.first_access[key] = timestamp
        self.access_times[key].append(timestamp)
        self.access_counts[key] += 1
        self.all_access_counts[key] += 1
        
        # Evict until we have space
        while self.current_size + size > self.capacity and self.cache:
            if self.predictor is not None and len(self.cache) > 1:
                # Use ML model to decide which item to evict
                evict_key = self._ml_evict_enhanced(timestamp)
            else:
                # Fallback to LRU if no predictor or only one item
                evict_key = min(self.cache.keys(), key=lambda k: self.cache[k])
            
            del self.cache[evict_key]
            self.current_size -= self.sizes[evict_key]
            del self.sizes[evict_key]
            # Don't delete from access_times - keep history
            self.evictions += 1
        
        # Add new item
        if size <= self.capacity:
            self.cache[key] = timestamp
            self.sizes[key] = size
            self.current_size += size
    
    def _ml_evict_enhanced(self, current_time):
        """
        Enhanced ML-based eviction with better features.
        Evicts the item least likely to be accessed again soon.
        """
        features_list = []
        keys_list = list(self.cache.keys())
        
        for key in keys_list:
            features = self._extract_enhanced_features(key, current_time)
            features_list.append(features)
        
        # Predict re-access probability for each item
        try:
            predictions = self.predictor.predict_proba(features_list)
            reaccess_probs = [pred[1] for pred in predictions]
            
            # Evict item with lowest probability of being accessed
            evict_idx = reaccess_probs.index(min(reaccess_probs))
            return keys_list[evict_idx]
        except:
            # Fallback to LRU on any error
            return min(keys_list, key=lambda k: self.cache[k])
    
    def _extract_enhanced_features(self, key, current_time):
        """
        Extract enhanced features matching the training model.
        """
        # Basic features
        last_access = self.access_times[key][-1] if self.access_times[key] else current_time
        recency = current_time - last_access
        
        frequency = self.access_counts[key]
        
        if len(self.access_times[key]) > 1:
            inter_arrivals = [
                self.access_times[key][i] - self.access_times[key][i-1]
                for i in range(1, len(self.access_times[key]))
            ]
            mean_inter_arrival = np.mean(inter_arrivals)
            var_inter_arrival = np.var(inter_arrivals)
        else:
            mean_inter_arrival = 0
            var_inter_arrival = 0
        
        size = self.sizes.get(key, 1)
        
        # Enhanced features
        max_recency = max([current_time - min(times) for times in self.access_times.values() 
                          if times] + [1])
        recency_norm = recency / max_recency if max_recency > 0 else 0
        
        # Frequency rank
        sorted_freqs = sorted(self.all_access_counts.values(), reverse=True)
        freq_rank = sorted_freqs.index(self.all_access_counts[key]) + 1 if self.all_access_counts[key] in sorted_freqs else len(sorted_freqs)
        freq_rank = freq_rank / max(len(sorted_freqs), 1)
        
        # Time since first access
        time_since_first = current_time - self.first_access.get(key, current_time)
        
        return [recency, frequency, mean_inter_arrival, var_inter_arrival, 
                size, recency_norm, freq_rank, time_since_first]
