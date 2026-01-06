"""
Enhanced ML model with better feature engineering and hyperparameters.
This version should provide better cache performance.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import pickle


class EnhancedCachePredictionModel:
    """
    Enhanced ML model with better features and ensemble methods.
    """
    
    def __init__(self, n_estimators=200, random_state=42):
        """Use Gradient Boosting for better performance."""
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
        self.is_trained = False
        self.feature_names = ['recency', 'frequency', 'mean_inter_arrival', 
                             'var_inter_arrival', 'size', 'recency_norm',
                             'freq_rank', 'time_since_first']
    
    def prepare_training_data(self, requests, cache_capacity=100, lookback_window=50):
        """
        Generate enhanced training data with additional features.
        Shorter lookback window for better temporal predictions.
        """
        X = []
        y = []
        
        access_times = defaultdict(list)
        access_counts = defaultdict(int)
        sizes = {}
        first_access = {}
        
        # Track all content frequencies for ranking
        all_frequencies = defaultdict(int)
        for req in requests:
            all_frequencies[req.content_id] += 1
        
        for idx, req in enumerate(requests):
            content_id = req.content_id
            timestamp = req.timestamp
            size = req.size
            
            if content_id not in first_access:
                first_access[content_id] = timestamp
            
            access_times[content_id].append(timestamp)
            access_counts[content_id] += 1
            sizes[content_id] = size
            
            if access_counts[content_id] > 1:
                features = self._extract_enhanced_features(
                    content_id, timestamp, access_times, access_counts, 
                    sizes, first_access, all_frequencies
                )
                
                label = self._check_future_access(
                    content_id, idx, requests, lookback_window
                )
                
                X.append(features)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def _extract_enhanced_features(self, content_id, current_time, access_times, 
                                   access_counts, sizes, first_access, all_frequencies):
        """Enhanced feature extraction."""
        # Basic features
        if len(access_times[content_id]) > 1:
            recency = current_time - access_times[content_id][-2]
        else:
            recency = 0
        
        frequency = access_counts[content_id]
        
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
        
        size = sizes.get(content_id, 1)
        
        # Enhanced features
        # Normalized recency (0-1 scale)
        max_recency = max([current_time - min(times) for times in access_times.values()] + [1])
        recency_norm = recency / max_recency if max_recency > 0 else 0
        
        # Frequency rank (higher frequency = lower rank number)
        sorted_freqs = sorted(all_frequencies.values(), reverse=True)
        freq_rank = sorted_freqs.index(frequency) + 1 if frequency in sorted_freqs else len(sorted_freqs)
        freq_rank = freq_rank / len(sorted_freqs)  # Normalize
        
        # Time since first access (content age)
        time_since_first = current_time - first_access.get(content_id, current_time)
        
        return [recency, frequency, mean_inter_arrival, var_inter_arrival, 
                size, recency_norm, freq_rank, time_since_first]
    
    def _check_future_access(self, content_id, current_idx, requests, window):
        """Check if content will be accessed within next 'window' requests."""
        end_idx = min(current_idx + window, len(requests))
        for i in range(current_idx + 1, end_idx):
            if requests[i].content_id == content_id:
                return 1
        return 0
    
    def train(self, X, y, test_size=0.2):
        """Train the enhanced model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training enhanced model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        
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
