"""
Main experiment runner with visualization.
Generates plots and metrics for the project report.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from traffic_gen import TrafficGenerator, generate_multiple_workloads
from ml_model import quick_train_model, CachePredictionModel
from simulator import run_comparison, BatchSimulator, save_results
from cache_policies import LRUCache, LFUCache, MLCache
import json
import os


# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def create_results_dir():
    """Create results directory if it doesn't exist."""
    if not os.path.exists('results'):
        os.makedirs('results')
    print("Results directory ready.")


def plot_hit_rate_comparison(results, save_path='results/hit_rate_comparison.png'):
    """Bar chart comparing hit rates across policies."""
    policies = list(results.keys())
    hit_rates = [results[p]['hit_rate'] for p in policies]
    
    plt.figure(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(policies)]
    bars = plt.bar(policies, hit_rates, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Cache Hit Rate', fontweight='bold')
    plt.xlabel('Cache Policy', fontweight='bold')
    plt.title('Cache Hit Rate Comparison', fontweight='bold', fontsize=14)
    plt.ylim([0, max(hit_rates) * 1.15])
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_latency_comparison(results, save_path='results/latency_comparison.png'):
    """Bar chart comparing average latencies."""
    policies = list(results.keys())
    latencies = [results[p]['avg_latency'] for p in policies]
    
    plt.figure(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(policies)]
    bars = plt.bar(policies, latencies, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Average Latency (ms)', fontweight='bold')
    plt.xlabel('Cache Policy', fontweight='bold')
    plt.title('Average Request Latency Comparison', fontweight='bold', fontsize=14)
    plt.ylim([0, max(latencies) * 1.15])
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_hit_rate_over_time(results, save_path='results/hit_rate_timeline.png'):
    """Line plot showing hit rate evolution over time."""
    plt.figure(figsize=(12, 6))
    
    for policy_name, stats in results.items():
        if 'metrics_history' in stats and len(stats['metrics_history']) > 0:
            history = stats['metrics_history']
            requests = [h['request_num'] for h in history]
            hit_rates = [h['hit_rate'] for h in history]
            plt.plot(requests, hit_rates, marker='o', markersize=3, 
                    label=policy_name, linewidth=2)
    
    plt.xlabel('Request Number', fontweight='bold')
    plt.ylabel('Cache Hit Rate', fontweight='bold')
    plt.title('Cache Hit Rate Over Time', fontweight='bold', fontsize=14)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_popularity_distribution(traffic_gen, save_path='results/content_popularity.png'):
    """Plot Zipf distribution of content popularity."""
    popularity = traffic_gen.get_popularity_distribution()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    ranks = np.arange(1, len(popularity) + 1)
    ax1.plot(ranks[:100], popularity[:100], 'b-', linewidth=2)
    ax1.set_xlabel('Content Rank', fontweight='bold')
    ax1.set_ylabel('Popularity (Probability)', fontweight='bold')
    ax1.set_title('Content Popularity Distribution (Top 100)', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Log-log scale (shows power law)
    ax2.loglog(ranks, popularity, 'r-', linewidth=2)
    ax2.set_xlabel('Content Rank (log scale)', fontweight='bold')
    ax2.set_ylabel('Popularity (log scale)', fontweight='bold')
    ax2.set_title('Zipf Distribution (Log-Log Plot)', fontweight='bold')
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_feature_importance(ml_metrics, save_path='results/feature_importance.png'):
    """Bar chart of ML model feature importance."""
    if 'feature_importance' not in ml_metrics:
        return
    
    importance = ml_metrics['feature_importance']
    features = list(importance.keys())
    values = list(importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(values)[::-1]
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = plt.barh(features, values, color=colors, alpha=0.8, edgecolor='black')
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(val, i, f' {val:.3f}', va='center', fontweight='bold')
    
    plt.xlabel('Importance Score', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title('ML Model Feature Importance', fontweight='bold', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(ml_metrics, save_path='results/confusion_matrix.png'):
    """Heatmap of confusion matrix."""
    if 'confusion_matrix' not in ml_metrics:
        return
    
    cm = ml_metrics['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Re-access', 'Re-access'],
                yticklabels=['No Re-access', 'Re-access'],
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.title('ML Model Confusion Matrix', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_batch_comparison(batch_results, save_path='results/batch_comparison.png'):
    """Bar chart with error bars showing results across multiple runs."""
    policies = list(batch_results.keys())
    means = [batch_results[p]['hit_rate_mean'] for p in policies]
    stds = [batch_results[p]['hit_rate_std'] for p in policies]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(policies)]
    x_pos = np.arange(len(policies))
    bars = plt.bar(x_pos, means, yerr=stds, capsize=10, 
                   color=colors, alpha=0.8, edgecolor='black', 
                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        plt.text(i, height + std, f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xticks(x_pos, policies)
    plt.ylabel('Cache Hit Rate', fontweight='bold')
    plt.xlabel('Cache Policy', fontweight='bold')
    plt.title('Cache Hit Rate Comparison (5 Independent Runs)', 
              fontweight='bold', fontsize=14)
    plt.ylim([0, max(means) * 1.2])
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_summary_report(results, batch_results, ml_metrics, 
                           save_path='results/summary.txt'):
    """Generate text summary of results."""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ML-BASED EDGE CACHE MANAGEMENT - EXPERIMENT RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("### Single Run Results ###\n\n")
        for policy_name, stats in results.items():
            f.write(f"{policy_name}:\n")
            f.write(f"  Hit Rate: {stats['hit_rate']:.4f}\n")
            f.write(f"  Avg Latency: {stats['avg_latency']:.2f} ms\n")
            f.write(f"  Total Requests: {stats['hits'] + stats['misses']}\n")
            f.write(f"  Hits: {stats['hits']}\n")
            f.write(f"  Misses: {stats['misses']}\n")
            f.write(f"  Evictions: {stats['evictions']}\n")
            f.write("\n")
        
        f.write("\n### Batch Results (5 Runs) ###\n\n")
        for policy_name, stats in batch_results.items():
            f.write(f"{policy_name}:\n")
            f.write(f"  Hit Rate: {stats['hit_rate_mean']:.4f} ± {stats['hit_rate_std']:.4f}\n")
            f.write(f"  Avg Latency: {stats['latency_mean']:.2f} ± {stats['latency_std']:.2f} ms\n")
            f.write(f"  Evictions: {stats['evictions_mean']:.1f} ± {stats['evictions_std']:.1f}\n")
            f.write("\n")
        
        if ml_metrics:
            f.write("\n### ML Model Performance ###\n\n")
            f.write(f"Train Accuracy: {ml_metrics['train_accuracy']:.4f}\n")
            f.write(f"Test Accuracy: {ml_metrics['test_accuracy']:.4f}\n")
            f.write("\nFeature Importance:\n")
            for feat, imp in sorted(ml_metrics['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True):
                f.write(f"  {feat}: {imp:.4f}\n")
        
        # Performance improvement
        if 'ML-Cache' in results and 'LRU' in results:
            improvement = ((results['ML-Cache']['hit_rate'] - results['LRU']['hit_rate']) 
                          / results['LRU']['hit_rate'] * 100)
            f.write(f"\n### Performance Gain ###\n\n")
            f.write(f"ML-Cache vs LRU: {improvement:+.2f}% hit rate improvement\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Saved: {save_path}")


def main():
    """Main experiment execution."""
    print("=" * 70)
    print("ML-BASED EDGE CACHE MANAGEMENT - EXPERIMENT RUNNER")
    print("=" * 70)
    
    # Configuration
    NUM_CONTENTS = 1000
    CACHE_CAPACITY = 100
    NUM_REQUESTS = 10000
    ZIPF_ALPHA = 1.2
    NUM_WORKLOADS = 5
    
    print(f"\nConfiguration:")
    print(f"  Unique Contents: {NUM_CONTENTS}")
    print(f"  Cache Capacity: {CACHE_CAPACITY}")
    print(f"  Requests per trace: {NUM_REQUESTS}")
    print(f"  Zipf alpha: {ZIPF_ALPHA}")
    print(f"  Number of workloads: {NUM_WORKLOADS}")
    
    # Create results directory
    create_results_dir()
    
    # Step 1: Generate traffic
    print("\n[1/5] Generating traffic patterns...")
    traffic_gen = TrafficGenerator(
        num_contents=NUM_CONTENTS,
        zipf_alpha=ZIPF_ALPHA,
        seed=42
    )
    main_workload = traffic_gen.generate_hybrid_workload(NUM_REQUESTS)
    print(f"  Generated {len(main_workload)} requests")
    
    # Plot popularity distribution
    plot_popularity_distribution(traffic_gen)
    
    # Step 2: Train ML model
    print("\n[2/5] Training ML model...")
    ml_model, ml_metrics = quick_train_model(main_workload, CACHE_CAPACITY)
    
    # Plot ML metrics
    plot_feature_importance(ml_metrics)
    plot_confusion_matrix(ml_metrics)
    
    # Step 3: Run single simulation
    print("\n[3/5] Running single simulation comparison...")
    results = run_comparison(main_workload, CACHE_CAPACITY, ml_model)
    
    # Save results
    save_results(results, 'results/single_run_results.json')
    
    # Plot single run results
    plot_hit_rate_comparison(results)
    plot_latency_comparison(results)
    plot_hit_rate_over_time(results)
    
    # Step 4: Run batch simulations
    print("\n[4/5] Running batch simulations (5 workloads)...")
    workloads = generate_multiple_workloads(
        num_workloads=NUM_WORKLOADS,
        num_requests=NUM_REQUESTS,
        num_contents=NUM_CONTENTS,
        zipf_alpha=ZIPF_ALPHA
    )
    
    batch_sim = BatchSimulator(CACHE_CAPACITY)
    batch_results = batch_sim.run_batch(workloads, ml_model)
    
    # Save batch results
    save_results(batch_results, 'results/batch_results.json')
    
    # Plot batch results
    plot_batch_comparison(batch_results)
    
    # Step 5: Generate summary report
    print("\n[5/5] Generating summary report...")
    generate_summary_report(results, batch_results, ml_metrics)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    print("\nResults saved to ./results/ directory:")
    print("  - hit_rate_comparison.png")
    print("  - latency_comparison.png")
    print("  - hit_rate_timeline.png")
    print("  - content_popularity.png")
    print("  - feature_importance.png")
    print("  - confusion_matrix.png")
    print("  - batch_comparison.png")
    print("  - summary.txt")
    print("  - single_run_results.json")
    print("  - batch_results.json")
    print("\nUse these for your presentation and report!")


if __name__ == '__main__':
    main()
