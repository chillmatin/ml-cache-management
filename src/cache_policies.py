"""
Cache replacement policies: LRU, LFU, and ML-based caching.
"""

from collections import OrderedDict, defaultdict
import time


class CachePolicy:
    """Base class for cache policies."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_latency = 0.0
        
    def get(self, key, size=1, timestamp=0):
        """
        Request content with given key.
        Returns (hit: bool, latency: float)
        """
        if key in self.cache:
            self.hits += 1
            latency = 5.0  # Cache hit latency (ms)
            self._on_hit(key, timestamp)
        else:
            self.misses += 1
            latency = 100.0  # Cache miss latency (ms)
            self._on_miss(key, size, timestamp)
        
        self.total_latency += latency
        return (key in self.cache, latency)
    
    def _on_hit(self, key, timestamp):
        """Called when cache hit occurs."""
        pass
    
    def _on_miss(self, key, size, timestamp):
        """Called when cache miss occurs."""
        pass
    
    def get_stats(self):
        """Return performance statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        avg_latency = self.total_latency / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'avg_latency': avg_latency,
            'evictions': self.evictions,
            'cache_size': len(self.cache)
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_latency = 0.0


class LRUCache(CachePolicy):
    """Least Recently Used cache policy."""
    
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = OrderedDict()
        self.sizes = {}
        self.current_size = 0
        
    def _on_hit(self, key, timestamp):
        """Move accessed item to end (most recent)."""
        self.cache.move_to_end(key)
    
    def _on_miss(self, key, size, timestamp):
        """Add new item, evict LRU if needed."""
        # Evict until we have space
        while self.current_size + size > self.capacity and self.cache:
            evicted_key, _ = self.cache.popitem(last=False)  # Remove oldest
            self.current_size -= self.sizes[evicted_key]
            del self.sizes[evicted_key]
            self.evictions += 1
        
        # Add new item
        if size <= self.capacity:
            self.cache[key] = timestamp
            self.sizes[key] = size
            self.current_size += size


class LFUCache(CachePolicy):
    """Least Frequently Used cache policy."""
    
    def __init__(self, capacity):
        super().__init__(capacity)
        self.cache = {}
        self.frequencies = defaultdict(int)
        self.sizes = {}
        self.current_size = 0
        
    def _on_hit(self, key, timestamp):
        """Increment frequency counter."""
        self.frequencies[key] += 1
    
    def _on_miss(self, key, size, timestamp):
        """Add new item, evict LFU if needed."""
        # Evict until we have space
        while self.current_size + size > self.capacity and self.cache:
            # Find least frequently used item
            lfu_key = min(self.cache.keys(), key=lambda k: self.frequencies[k])
            del self.cache[lfu_key]
            self.current_size -= self.sizes[lfu_key]
            del self.sizes[lfu_key]
            del self.frequencies[lfu_key]
            self.evictions += 1
        
        # Add new item
        if size <= self.capacity:
            self.cache[key] = timestamp
            self.sizes[key] = size
            self.frequencies[key] = 1
            self.current_size += size


class MLCache(CachePolicy):
    """
    ML-based cache policy.
    Uses a predictor model to decide which items to evict.
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
        
    def _on_hit(self, key, timestamp):
        """Record access pattern."""
        self.access_times[key].append(timestamp)
        self.access_counts[key] += 1
    
    def _on_miss(self, key, size, timestamp):
        """Add new item, use ML model to decide eviction."""
        # Record access
        self.access_times[key].append(timestamp)
        self.access_counts[key] += 1
        
        # Evict until we have space
        while self.current_size + size > self.capacity and self.cache:
            if self.predictor is not None:
                # Use ML model to decide which item to evict
                evict_key = self._ml_evict(timestamp)
            else:
                # Fallback to LRU if no predictor
                evict_key = min(self.cache.keys(), key=lambda k: self.cache[k])
            
            del self.cache[evict_key]
            self.current_size -= self.sizes[evict_key]
            del self.sizes[evict_key]
            self.evictions += 1
        
        # Add new item
        if size <= self.capacity:
            self.cache[key] = timestamp
            self.sizes[key] = size
            self.current_size += size
    
    def _ml_evict(self, current_time):
        """
        Use ML model to select item for eviction.
        Evicts the item least likely to be accessed again soon.
        """
        features_list = []
        keys_list = list(self.cache.keys())
        
        for key in keys_list:
            features = self._extract_features(key, current_time)
            features_list.append(features)
        
        # Predict re-access probability for each item
        predictions = self.predictor.predict_proba(features_list)
        
        # Evict item with lowest probability of being accessed
        reaccess_probs = [pred[1] for pred in predictions]  # Probability of class 1 (will be accessed)
        evict_idx = reaccess_probs.index(min(reaccess_probs))
        
        return keys_list[evict_idx]
    
    def _extract_features(self, key, current_time):
        """
        Extract features for ML prediction.
        Features: recency, frequency, inter-arrival variance, size
        """
        # Recency: time since last access
        last_access = self.access_times[key][-1] if self.access_times[key] else current_time
        recency = current_time - last_access
        
        # Frequency: number of accesses
        frequency = self.access_counts[key]
        
        # Inter-arrival time statistics
        if len(self.access_times[key]) > 1:
            inter_arrivals = [
                self.access_times[key][i] - self.access_times[key][i-1]
                for i in range(1, len(self.access_times[key]))
            ]
            mean_inter_arrival = sum(inter_arrivals) / len(inter_arrivals)
            var_inter_arrival = sum((x - mean_inter_arrival)**2 for x in inter_arrivals) / len(inter_arrivals)
        else:
            mean_inter_arrival = 0
            var_inter_arrival = 0
        
        # Size
        size = self.sizes.get(key, 1)
        
        return [recency, frequency, mean_inter_arrival, var_inter_arrival, size]
    
    def get_history(self):
        """Return access history for training ML model."""
        return {
            'access_times': dict(self.access_times),
            'access_counts': dict(self.access_counts),
            'sizes': dict(self.sizes)
        }
