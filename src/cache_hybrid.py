"""
Hybrid ML-LFU cache that combines ML predictions with frequency-based fallback.
This version provides more robust performance.
"""

from cache_policies import CachePolicy
from collections import defaultdict
import numpy as np


class HybridMLCache(CachePolicy):
    """
    Hybrid cache policy combining ML predictions with LFU fallback.
    Uses ML when confident, falls back to LFU otherwise.
    """
    
    def __init__(self, capacity, predictor=None, confidence_threshold=0.7):
        super().__init__(capacity)
        self.cache = {}
        self.sizes = {}
        self.current_size = 0
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold
        
        # Track access history
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
        """Add new item using hybrid eviction strategy."""
        # Record access
        if key not in self.first_access:
            self.first_access[key] = timestamp
        self.access_times[key].append(timestamp)
        self.access_counts[key] += 1
        self.all_access_counts[key] += 1
        
        # Evict until we have space
        while self.current_size + size > self.capacity and self.cache:
            evict_key = self._hybrid_evict(timestamp)
            
            del self.cache[evict_key]
            self.current_size -= self.sizes[evict_key]
            del self.sizes[evict_key]
            self.evictions += 1
        
        # Add new item
        if size <= self.capacity:
            self.cache[key] = timestamp
            self.sizes[key] = size
            self.current_size += size
    
    def _hybrid_evict(self, current_time):
        """
        Hybrid eviction: Use ML if confident, otherwise LFU.
        """
        if self.predictor is None or len(self.cache) <= 1:
            # Fallback to LFU
            return min(self.cache.keys(), key=lambda k: self.access_counts[k])
        
        try:
            # Get ML predictions
            features_list = []
            keys_list = list(self.cache.keys())
            
            for key in keys_list:
                features = self._extract_enhanced_features(key, current_time)
                features_list.append(features)
            
            predictions = self.predictor.predict_proba(features_list)
            
            # Calculate confidence and prediction
            ml_scores = []
            confidences = []
            for pred in predictions:
                prob_reaccess = pred[1]  # Probability of being accessed again
                confidence = abs(prob_reaccess - 0.5) * 2  # How confident (0=uncertain, 1=very confident)
                ml_scores.append(prob_reaccess)
                confidences.append(confidence)
            
            # Find most confident prediction
            max_confidence_idx = np.argmax(confidences)
            
            if confidences[max_confidence_idx] >= self.confidence_threshold:
                # Use ML: evict item with lowest re-access probability
                evict_idx = np.argmin(ml_scores)
                return keys_list[evict_idx]
            else:
                # Low confidence: fallback to LFU
                return min(keys_list, key=lambda k: self.access_counts[k])
                
        except Exception as e:
            # Any error: fallback to LFU
            return min(self.cache.keys(), key=lambda k: self.access_counts[k])
    
    def _extract_enhanced_features(self, key, current_time):
        """Extract enhanced features for ML prediction."""
        # Recency
        last_access = self.access_times[key][-1] if self.access_times[key] else current_time
        recency = current_time - last_access
        
        # Frequency
        frequency = self.access_counts[key]
        
        # Inter-arrival statistics
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
        
        sorted_freqs = sorted(self.all_access_counts.values(), reverse=True)
        freq_rank = sorted_freqs.index(self.all_access_counts[key]) + 1 if self.all_access_counts[key] in sorted_freqs else len(sorted_freqs)
        freq_rank = freq_rank / max(len(sorted_freqs), 1)
        
        time_since_first = current_time - self.first_access.get(key, current_time)
        
        return [recency, frequency, mean_inter_arrival, var_inter_arrival, 
                size, recency_norm, freq_rank, time_since_first]
