import numpy as np

from core.fitness import evaluate_final_model, evaluate_solution


class GAPSOHybrid:
    def __init__(
        self,
        num_features,
        num_particles=10,
        num_iterations=10,
        mutation_rate=0.08,
        crossover_rate=0.85,
        elite_ratio=0.25,
    ):
        self.num_features = num_features
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.dim = num_features + 3

        self.w = 0.6
        self.c1 = 1.7
        self.c2 = 1.7

    @staticmethod
    def _tournament_select(population, costs, k=3):
        idx = np.random.choice(len(population), k, replace=False)
        winner = idx[np.argmin(costs[idx])]
        return population[winner].copy()

    def _crossover(self, parent_a, parent_b):
        if np.random.rand() >= self.crossover_rate:
            return parent_a.copy(), parent_b.copy()

        point = np.random.randint(1, self.dim)
        child_a = np.concatenate([parent_a[:point], parent_b[point:]])
        child_b = np.concatenate([parent_b[:point], parent_a[point:]])
        return child_a, child_b

    def _mutate(self, candidate):
        mask = np.random.rand(self.dim) < self.mutation_rate
        if np.any(mask):
            candidate[mask] = np.random.rand(np.sum(mask))
        return np.clip(candidate, 0.0, 1.0)

    def optimize(self, X_train, y_train, X_val, y_val):
        print(
            f"\n--- Starting GA-PSO Hybrid Optimization "
            f"({self.num_iterations} iterations, {self.num_particles} particles) ---"
        )

        particles = np.random.rand(self.num_particles, self.dim)
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))

        pbest = particles.copy()
        pbest_costs = np.full(self.num_particles, np.inf)
        gbest = particles[0].copy()
        gbest_cost = np.inf
        history = []

        for iteration in range(self.num_iterations):
            print(f"GA-PSO Iteration {iteration + 1}/{self.num_iterations}")

            costs = np.array(
                [evaluate_solution(ind, X_train, y_train, X_val, y_val) for ind in particles]
            )

            improved = costs < pbest_costs
            pbest_costs[improved] = costs[improved]
            pbest[improved] = particles[improved]

            best_idx = int(np.argmin(costs))
            if costs[best_idx] < gbest_cost:
                gbest_cost = costs[best_idx]
                gbest = particles[best_idx].copy()

            progress = iteration / max(1, self.num_iterations - 1)
            inertia = 0.80 - 0.45 * progress

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (
                inertia * velocities
                + self.c1 * r1 * (pbest - particles)
                + self.c2 * r2 * (gbest - particles)
            )
            particles = np.clip(particles + velocities, 0.0, 1.0)

            elite_count = max(2, int(self.elite_ratio * self.num_particles))
            elite_idx = np.argsort(costs)[:elite_count]
            elites = particles[elite_idx].copy()

            offspring = []
            while len(offspring) < (self.num_particles - elite_count):
                parent_a = self._tournament_select(particles, costs)
                parent_b = self._tournament_select(particles, costs)
                child_a, child_b = self._crossover(parent_a, parent_b)
                offspring.append(self._mutate(child_a))
                if len(offspring) < (self.num_particles - elite_count):
                    offspring.append(self._mutate(child_b))

            particles = np.vstack([elites, np.array(offspring)])
            particles = particles[: self.num_particles]

            history.append(gbest_cost)
            print(f"  Global Best Cost so far: {gbest_cost:.4f}")

        print("GA-PSO Hybrid Optimization Finished!")
        return gbest, gbest_cost, history


def run_gapso_hybrid(
    X_train,
    y_train,
    X_val,
    y_val,
    num_particles=10,
    num_iterations=10,
):
    num_features = X_train.shape[1]
    gapso = GAPSOHybrid(
        num_features=num_features,
        num_particles=num_particles,
        num_iterations=num_iterations,
    )

    best_particle, best_cost, history = gapso.optimize(X_train, y_train, X_val, y_val)
    metrics, model, mask = evaluate_final_model(best_particle, X_train, y_train, X_val, y_val)
    return metrics, model, mask, history
