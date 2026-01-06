"""
Machine learning model for cache eviction prediction.
Predicts whether a cached item will be accessed again soon.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import pickle


class CachePredictionModel:
    """
    ML model to predict content re-access probability.
    Uses Random Forest classifier trained on access patterns.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Args:
            n_estimators: Number of trees in Random Forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = ['recency', 'frequency', 'mean_inter_arrival', 
                             'var_inter_arrival', 'size']
    
    def prepare_training_data(self, requests, cache_capacity=100, lookback_window=100):
        """
        Generate training data from request trace.
        
        For each request, extract features and label based on whether
        the content is accessed again within the lookback window.
        
        Args:
            requests: List of Request objects
            cache_capacity: Cache size for simulation
            lookback_window: Number of future requests to check for re-access
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 if accessed again, 0 otherwise)
        """
        X = []
        y = []
        
        # Track access patterns
        access_times = defaultdict(list)
        access_counts = defaultdict(int)
        sizes = {}
        
        for idx, req in enumerate(requests):
            content_id = req.content_id
            timestamp = req.timestamp
            size = req.size
            
            # Record current state
            access_times[content_id].append(timestamp)
            access_counts[content_id] += 1
            sizes[content_id] = size
            
            # Extract features for this access
            if access_counts[content_id] > 1:  # Need at least one prior access
                features = self._extract_features(
                    content_id, timestamp, access_times, access_counts, sizes
                )
                
                # Determine label: will this content be accessed in next N requests?
                label = self._check_future_access(
                    content_id, idx, requests, lookback_window
                )
                
                X.append(features)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def _extract_features(self, content_id, current_time, access_times, access_counts, sizes):
        """
        Extract features for ML prediction.
        Features: recency, frequency, inter-arrival mean/variance, size
        """
        # Recency: time since last access
        if len(access_times[content_id]) > 1:
            recency = current_time - access_times[content_id][-2]
        else:
            recency = 0
        
        # Frequency: number of accesses
        frequency = access_counts[content_id]
        
        # Inter-arrival time statistics
        if len(access_times[content_id]) > 1:
            inter_arrivals = [
                access_times[content_id][i] - access_times[content_id][i-1]
                for i in range(1, len(access_times[content_id]))
            ]
            mean_inter_arrival = np.mean(inter_arrivals)
            var_inter_arrival = np.var(inter_arrivals)
        else:
            mean_inter_arrival = 0
            var_inter_arrival = 0
        
        # Size
        size = sizes.get(content_id, 1)
        
        return [recency, frequency, mean_inter_arrival, var_inter_arrival, size]
    
    def _check_future_access(self, content_id, current_idx, requests, window):
        """
        Check if content will be accessed within next 'window' requests.
        """
        end_idx = min(current_idx + window, len(requests))
        for i in range(current_idx + 1, end_idx):
            if requests[i].content_id == content_id:
                return 1
        return 0
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model on prepared data.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        print(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance
        }
        
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        print("\nFeature importance:")
        for name, importance in sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {name}: {importance:.3f}")
        
        return metrics
    
    def predict(self, features):
        """Predict class (0 or 1) for given features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict([features])[0]
    
    def predict_proba(self, features):
        """Predict probability distribution for given features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        if len(np.array(features).shape) == 1:
            features = [features]
        return self.model.predict_proba(features)
    
    def save(self, filepath):
        """Save trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def quick_train_model(requests, cache_capacity=100):
    """
    Quick training function for use in simulations.
    
    Args:
        requests: List of Request objects
        cache_capacity: Cache size
        
    Returns:
        Trained CachePredictionModel
    """
    print("Preparing training data...")
    model = CachePredictionModel()
    
    # Use first 50% of trace for training
    train_requests = requests[:len(requests)//2]
    
    X, y = model.prepare_training_data(train_requests, cache_capacity)
    
    print(f"Generated {len(X)} training samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train
    metrics = model.train(X, y)
    
    return model, metrics
