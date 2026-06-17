import numpy as np


def simulate_has_policy(
        num_samples=100000,
        true_anomaly_rate=0.02,
        baseline_retention=0.01,
        uncertainty_threshold=0.85
):
    """
    Simulates the Human Audit Slice (HAS) retention policy against a static baseline.
    """
    # 1. Simulate Ground Truth Data
    # 1 represents a rare phenotype/anomaly, 0 represents standard background data
    ground_truth = np.random.binomial(1, true_anomaly_rate, num_samples)

    # 2. Simulate Latent Encoder Outputs
    # Encoder is imperfect. It generates uncertainty scores (0 to 1).
    # True anomalies generally cause higher uncertainty, but with added noise.
    base_uncertainty = np.random.uniform(0.1, 0.5, num_samples)
    anomaly_spike = ground_truth * np.random.uniform(0.3, 0.6, num_samples)
    encoder_uncertainty = np.clip(base_uncertainty + anomaly_spike, 0, 1)

    # Encoder attempts to classify rare events (with some false negatives)
    encoder_rare_event_flag = (encoder_uncertainty > 0.90).astype(int)

    # 3. Apply Static Policy (e.g., purely random 1% retention)
    # This represents a naive approach to storage reduction.
    static_retention_mask = np.random.binomial(1, baseline_retention, num_samples)
    static_anomalies_caught = np.sum(ground_truth * static_retention_mask)

    # 4. Apply Adaptive HAS Policy (Paper's Proposed Logic)
    # HAS = Baseline + Uncertainty Trigger + Rare-Event Escalation
    baseline_mask = np.random.binomial(1, baseline_retention, num_samples)
    uncertainty_mask = (encoder_uncertainty > uncertainty_threshold).astype(int)
    rare_event_mask = encoder_rare_event_flag

    # Logical OR: Retain if it meets ANY of the HAS criteria
    has_retention_mask = np.logical_or.reduce((baseline_mask, uncertainty_mask, rare_event_mask))
    has_anomalies_caught = np.sum(ground_truth * has_retention_mask)

    # Calculate System Metrics
    total_anomalies = np.sum(ground_truth)
    static_recall = static_anomalies_caught / total_anomalies
    has_recall = has_anomalies_caught / total_anomalies

    static_storage_cost = np.mean(static_retention_mask)
    has_storage_cost = np.mean(has_retention_mask)

    return {
        "Total Samples": num_samples,
        "True Anomalies": total_anomalies,
        "Static Policy": {
            "Storage Retained": f"{static_storage_cost * 100:.2f}%",
            "Anomaly Recall": f"{static_recall * 100:.2f}%"
        },
        "HAS Policy": {
            "Storage Retained": f"{has_storage_cost * 100:.2f}%",
            "Anomaly Recall": f"{has_recall * 100:.2f}%"
        }
    }


# Execute the baseline simulation
results = simulate_has_policy()

print("--- Simulation Results ---")
for key, value in results.items():
    if isinstance(value, dict):
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")