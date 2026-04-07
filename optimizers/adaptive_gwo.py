import numpy as np

from core.fitness import evaluate_final_model, evaluate_solution


class AdaptiveGWO:
    def __init__(self, num_features, num_wolves=10, num_iterations=10):
        self.num_features = num_features
        self.num_wolves = num_wolves
        self.num_iterations = num_iterations
        self.dim = num_features + 3

    def optimize(self, X_train, y_train, X_val, y_val):
        print(
            f"\n--- Starting Adaptive GWO Optimization "
            f"({self.num_iterations} iterations, {self.num_wolves} wolves) ---"
        )

        positions = np.random.rand(self.num_wolves, self.dim)
        history = []

        alpha_pos = np.zeros(self.dim)
        alpha_score = float("inf")
        beta_pos = np.zeros(self.dim)
        beta_score = float("inf")
        delta_pos = np.zeros(self.dim)
        delta_score = float("inf")

        initial_diversity = np.mean(np.std(positions, axis=0)) + 1e-12

        for iteration in range(self.num_iterations):
            print(f"Adaptive GWO Iteration {iteration + 1}/{self.num_iterations}")

            for idx in range(self.num_wolves):
                positions[idx] = np.clip(positions[idx], 0.0, 1.0)
                fitness = evaluate_solution(positions[idx], X_train, y_train, X_val, y_val)

                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = positions[idx].copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = positions[idx].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = positions[idx].copy()

            progress = iteration / max(1, self.num_iterations - 1)
            base_a = 2.0 * (1.0 - progress)
            diversity = np.mean(np.std(positions, axis=0))
            diversity_ratio = diversity / initial_diversity
            exploration_boost = 0.35 if diversity_ratio < 0.25 else 0.0
            a = min(2.0, base_a + exploration_boost)

            # Keep more memory of prior position early, then increase exploitation.
            inertia = 0.75 - 0.50 * progress

            for idx in range(self.num_wolves):
                candidate = np.zeros(self.dim)
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[idx, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2.0 * a * r1 - a
                    C2 = 2.0 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[idx, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2.0 * a * r1 - a
                    C3 = 2.0 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[idx, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    candidate[j] = (X1 + X2 + X3) / 3.0

                positions[idx] = np.clip(
                    inertia * positions[idx] + (1.0 - inertia) * candidate,
                    0.0,
                    1.0,
                )

            history.append(alpha_score)
            print(
                f"  Alpha (Best) Cost: {alpha_score:.4f} | "
                f"a: {a:.3f} | diversity ratio: {diversity_ratio:.3f}"
            )

        print("Adaptive GWO Optimization Finished!")
        return alpha_pos, alpha_score, history


def run_adaptive_gwo(X_train, y_train, X_val, y_val, num_wolves=10, num_iterations=10):
    num_features = X_train.shape[1]
    agwo = AdaptiveGWO(
        num_features=num_features,
        num_wolves=num_wolves,
        num_iterations=num_iterations,
    )

    best_particle, best_cost, history = agwo.optimize(X_train, y_train, X_val, y_val)
    metrics, model, mask = evaluate_final_model(best_particle, X_train, y_train, X_val, y_val)
    return metrics, model, mask, history
