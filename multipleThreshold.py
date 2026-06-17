import numpy as np
import matplotlib.pyplot as plt


def generate_pareto_frontier(num_samples=250000, true_anomaly_rate=0.015, baseline_retention=0.01):
    """
    Executes a parameter sweep to find the Pareto optimal trade-offs
    between raw data storage costs and biological anomaly recall.
    """
    # 1. Generate Ground Truth (1 = Rare Anomaly, 0 = Normal Background)
    ground_truth = np.random.binomial(1, true_anomaly_rate, num_samples)
    total_anomalies = np.sum(ground_truth)

    # 2. Simulate Latent Encoder Confidence (Beta Distributions)
    # Normal data tends to have low uncertainty (skewed left)
    normal_uncertainty = np.random.beta(a=2, b=8, size=num_samples)
    # Anomalous data tends to have higher uncertainty (skewed right)
    anomaly_uncertainty = np.random.beta(a=7, b=3, size=num_samples)

    # Assign uncertainty based on ground truth
    encoder_uncertainty = np.where(ground_truth == 1, anomaly_uncertainty, normal_uncertainty)

    # 3. Define the parameter sweep range (100 threshold levels from 0.01 to 0.99)
    thresholds = np.linspace(0.01, 0.99, 100)

    storage_costs = []
    anomaly_recalls = []

    # 4. Generate the static random baseline slice (e.g., 1% random retention)
    baseline_mask = np.random.binomial(1, baseline_retention, num_samples)

    # 5. Execute the sweep
    for thresh in thresholds:
        # The model flags anything above the current threshold as uncertain
        uncertainty_mask = (encoder_uncertainty >= thresh).astype(int)

        # HAS Policy: Retain if in random baseline OR if model is uncertain
        has_mask = np.logical_or(baseline_mask, uncertainty_mask)

        # Calculate metrics for this specific threshold
        storage_retained_pct = np.mean(has_mask) * 100
        anomalies_caught = np.sum(ground_truth * has_mask)
        recall_pct = (anomalies_caught / total_anomalies) * 100

        storage_costs.append(storage_retained_pct)
        anomaly_recalls.append(recall_pct)

    # 6. Plotting the Pareto Frontier
    plt.figure(figsize=(10, 6))
    plt.plot(storage_costs, anomaly_recalls, color='blue', linewidth=2, label='HAS Efficiency Curve')

    # Plot the naive static baseline for comparison
    plt.scatter([baseline_retention * 100], [(baseline_retention * 100)],
                color='red', zorder=5, label='Pure Random Sampling')

    # Formatting the plot
    plt.title('Pareto Frontier: Storage Cost vs. Anomaly Recall', fontsize=14, fontweight='bold')
    plt.xlabel('Total Raw Data Retained (%)', fontsize=12)
    plt.ylabel('Rare Event Recall (%)', fontsize=12)
    plt.xlim(0, max(storage_costs) + 2)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=95, color='gray', linestyle=':', label='95% Recall Target')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return thresholds, storage_costs, anomaly_recalls


# Execute the simulation
t, s, r = generate_pareto_frontier()

# Output the optimal threshold for a 95% recall target
target_idx = next(i for i, recall in enumerate(r) if recall <= 95.0)
print(f"To guarantee ~95% Anomaly Recall:")
print(f"Optimal Uncertainty Threshold: {t[target_idx]:.3f}")
print(f"Required Storage Retention: {s[target_idx]:.2f}%")