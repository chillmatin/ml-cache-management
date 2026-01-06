"""
Traffic generation for cache simulation.
Generates realistic content request patterns using Zipf and Poisson distributions.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Request:
    """Represents a content request."""
    timestamp: float
    content_id: int
    size: int


class TrafficGenerator:
    """
    Generates realistic traffic patterns for cache simulation.
    """
    
    def __init__(self, 
                 num_contents=1000,
                 zipf_alpha=1.2,
                 mean_request_rate=10.0,
                 seed=42):
        """
        Args:
            num_contents: Number of unique content objects
            zipf_alpha: Zipf distribution parameter (higher = more skewed)
            mean_request_rate: Average requests per second
            seed: Random seed for reproducibility
        """
        self.num_contents = num_contents
        self.zipf_alpha = zipf_alpha
        self.mean_request_rate = mean_request_rate
        self.rng = np.random.RandomState(seed)
        
        # Generate content sizes (1-100 units, roughly power-law distributed)
        self.content_sizes = self._generate_content_sizes()
        
    def _generate_content_sizes(self):
        """Generate realistic content sizes using log-normal distribution."""
        sizes = self.rng.lognormal(mean=2.0, sigma=1.0, size=self.num_contents)
        sizes = np.clip(sizes, 1, 100).astype(int)
        return sizes
    
    def generate_requests(self, 
                         num_requests=10000,
                         temporal_pattern=True,
                         flash_crowd=False) -> List[Request]:
        """
        Generate a trace of content requests.
        
        Args:
            num_requests: Number of requests to generate
            temporal_pattern: Add daily/hourly patterns
            flash_crowd: Add sudden popularity spikes
            
        Returns:
            List of Request objects sorted by timestamp
        """
        requests = []
        current_time = 0.0
        
        # Generate Zipfian content popularity distribution
        popularity_weights = self._zipf_distribution(self.num_contents, self.zipf_alpha)
        
        for i in range(num_requests):
            # Poisson inter-arrival times
            inter_arrival = self.rng.exponential(1.0 / self.mean_request_rate)
            current_time += inter_arrival
            
            # Temporal modulation (daily pattern)
            if temporal_pattern:
                time_of_day = (current_time % 86400) / 86400  # Normalize to [0, 1]
                # Peak at noon and evening (6pm)
                temporal_factor = 1.0 + 0.5 * np.sin(2 * np.pi * time_of_day - np.pi/2)
                temporal_factor += 0.3 * np.sin(4 * np.pi * time_of_day)
                
                # Adjust request rate
                if self.rng.random() > temporal_factor / 2.0:
                    continue
            
            # Flash crowd event (sudden spike at 25% through trace)
            if flash_crowd and 0.25 * num_requests <= i <= 0.27 * num_requests:
                # Temporarily boost popularity of content IDs 5-15
                flash_weights = popularity_weights.copy()
                flash_weights[5:15] *= 10.0
                flash_weights /= flash_weights.sum()
                content_id = self.rng.choice(self.num_contents, p=flash_weights)
            else:
                # Normal Zipfian selection
                content_id = self.rng.choice(self.num_contents, p=popularity_weights)
            
            size = self.content_sizes[content_id]
            requests.append(Request(current_time, content_id, size))
        
        return sorted(requests, key=lambda r: r.timestamp)
    
    def _zipf_distribution(self, n, alpha):
        """Generate normalized Zipf distribution weights."""
        ranks = np.arange(1, n + 1)
        weights = 1.0 / (ranks ** alpha)
        weights /= weights.sum()
        return weights
    
    def generate_hybrid_workload(self, num_requests=10000) -> List[Request]:
        """
        Generate a mixed workload with multiple patterns.
        Combines temporal patterns and flash crowd events.
        """
        return self.generate_requests(
            num_requests=num_requests,
            temporal_pattern=True,
            flash_crowd=True
        )
    
    def get_popularity_distribution(self):
        """Return the content popularity distribution for analysis."""
        return self._zipf_distribution(self.num_contents, self.zipf_alpha)


def generate_multiple_workloads(num_workloads=5, 
                                num_requests=10000,
                                num_contents=1000,
                                zipf_alpha=1.2) -> List[List[Request]]:
    """
    Generate multiple independent workloads for robust evaluation.
    """
    workloads = []
    for seed in range(num_workloads):
        gen = TrafficGenerator(
            num_contents=num_contents,
            zipf_alpha=zipf_alpha,
            seed=seed
        )
        workload = gen.generate_hybrid_workload(num_requests)
        workloads.append(workload)
    
    return workloads
