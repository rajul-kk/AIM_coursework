import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

from core.fitness import decode_particle, evaluate_final_model


def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def _non_dominated_sort(objectives):
    pop_size = objectives.shape[0]
    dominates_list = [[] for _ in range(pop_size)]
    domination_count = np.zeros(pop_size, dtype=int)
    fronts = [[]]

    for p in range(pop_size):
        for q in range(pop_size):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates_list[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [front for front in fronts if front]


def _generate_reference_points(num_objectives, divisions):
    points = []

    def _recurse(remaining, left, current):
        if remaining == 1:
            points.append(current + [left])
            return
        for i in range(left + 1):
            _recurse(remaining - 1, left - i, current + [i])

    _recurse(num_objectives, divisions, [])
    ref = np.array(points, dtype=float)
    ref /= divisions
    return ref


def _normalize_objectives(objs):
    ideal = np.min(objs, axis=0)
    nadir = np.max(objs, axis=0)
    scale = np.where((nadir - ideal) < 1e-12, 1.0, (nadir - ideal))
    return (objs - ideal) / scale


def _associate_to_reference_points(normalized_objs, ref_points):
    ref_norm = np.linalg.norm(ref_points, axis=1)
    ref_norm = np.where(ref_norm < 1e-12, 1e-12, ref_norm)
    ref_unit = ref_points / ref_norm[:, None]

    niche_idx = np.zeros(len(normalized_objs), dtype=int)
    distances = np.zeros(len(normalized_objs), dtype=float)

    for i, point in enumerate(normalized_objs):
        d1 = np.dot(ref_unit, point)
        proj = d1[:, None] * ref_unit
        perp = np.linalg.norm(point - proj, axis=1)
        best = int(np.argmin(perp))
        niche_idx[i] = best
        distances[i] = perp[best]

    return niche_idx, distances


class NSGAIII:
    def __init__(
        self,
        num_features,
        pop_size=30,
        num_iterations=30,
        divisions=6,
        crossover_rate=0.9,
        mutation_rate=0.1,
    ):
        self.num_features = num_features
        self.pop_size = pop_size
        self.num_iterations = num_iterations
        self.divisions = divisions
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.dim = num_features + 3
        self.ref_points = _generate_reference_points(4, divisions)

    def _evaluate_objectives(self, particle, X_train, y_train, X_val, y_val):
        feature_mask, n_est, m_depth, min_split = decode_particle(particle, self.num_features)

        X_train_filtered = X_train[:, feature_mask]
        X_val_filtered = X_val[:, feature_mask]

        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=m_depth,
            min_samples_split=min_split,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_filtered, y_train)
        y_pred = rf.predict(X_val_filtered)

        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            recall_attack = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            total_fp = (cm.sum(axis=0) - np.diag(cm)).sum()
            fp_rate = total_fp / cm.sum() if cm.sum() > 0 else 0.0
            recall_attack = recall_score(y_val, y_pred, average="weighted", zero_division=0)

        selected_ratio = float(np.sum(feature_mask)) / float(self.num_features)

        # Minimize all objectives.
        obj = np.array(
            [
                1.0 - recall_attack,
                fp_rate,
                1.0 - acc,
                selected_ratio,
            ],
            dtype=float,
        )
        return obj

    def _evaluate_population(self, population, X_train, y_train, X_val, y_val):
        return np.array(
            [
                self._evaluate_objectives(ind, X_train, y_train, X_val, y_val)
                for ind in population
            ]
        )

    def _weighted_cost(self, objectives):
        # Same priority profile used in scalar fitness for tie-breaking/reporting.
        return (
            0.35 * objectives[:, 0]
            + 0.30 * objectives[:, 1]
            + 0.20 * objectives[:, 2]
            + 0.15 * objectives[:, 3]
        )

    def _crossover(self, parent_a, parent_b):
        if np.random.rand() >= self.crossover_rate:
            return parent_a.copy(), parent_b.copy()
        alpha = np.random.rand(self.dim)
        child_a = alpha * parent_a + (1.0 - alpha) * parent_b
        child_b = alpha * parent_b + (1.0 - alpha) * parent_a
        return np.clip(child_a, 0.0, 1.0), np.clip(child_b, 0.0, 1.0)

    def _mutate(self, child):
        mask = np.random.rand(self.dim) < self.mutation_rate
        if np.any(mask):
            child[mask] = np.clip(child[mask] + np.random.normal(0, 0.1, np.sum(mask)), 0.0, 1.0)
        return child

    def _tournament(self, population, costs):
        i, j = np.random.choice(len(population), 2, replace=False)
        return population[i].copy() if costs[i] <= costs[j] else population[j].copy()

    def _niching_select(self, front_indices, all_objectives, selected_indices, remaining):
        chosen = []
        normalized = _normalize_objectives(all_objectives)

        all_niches, all_dist = _associate_to_reference_points(normalized, self.ref_points)

        niche_count = np.zeros(len(self.ref_points), dtype=int)
        for idx in selected_indices:
            niche_count[all_niches[idx]] += 1

        candidates = list(front_indices)
        while len(chosen) < remaining and candidates:
            candidate_niches = np.unique([all_niches[idx] for idx in candidates])
            min_count = min(niche_count[n] for n in candidate_niches)
            candidate_ref = [n for n in candidate_niches if niche_count[n] == min_count]
            ref_choice = int(np.random.choice(candidate_ref))

            niche_members = [idx for idx in candidates if all_niches[idx] == ref_choice]
            if not niche_members:
                continue

            if niche_count[ref_choice] == 0:
                pick = min(niche_members, key=lambda idx: all_dist[idx])
            else:
                pick = int(np.random.choice(niche_members))

            chosen.append(pick)
            candidates.remove(pick)
            niche_count[ref_choice] += 1

        if len(chosen) < remaining and candidates:
            extra = np.random.choice(candidates, remaining - len(chosen), replace=False).tolist()
            chosen.extend(extra)

        return chosen

    def optimize(self, X_train, y_train, X_val, y_val):
        print(
            f"\n--- Starting NSGA-III Optimization "
            f"({self.num_iterations} generations, {self.pop_size} population) ---"
        )

        population = np.random.rand(self.pop_size, self.dim)
        history = []

        for generation in range(self.num_iterations):
            print(f"NSGA-III Generation {generation + 1}/{self.num_iterations}")

            population_obj = self._evaluate_population(population, X_train, y_train, X_val, y_val)
            population_cost = self._weighted_cost(population_obj)

            offspring = []
            while len(offspring) < self.pop_size:
                parent_a = self._tournament(population, population_cost)
                parent_b = self._tournament(population, population_cost)
                child_a, child_b = self._crossover(parent_a, parent_b)
                offspring.append(self._mutate(child_a))
                if len(offspring) < self.pop_size:
                    offspring.append(self._mutate(child_b))

            offspring = np.array(offspring)
            combined = np.vstack([population, offspring])
            combined_obj = self._evaluate_population(combined, X_train, y_train, X_val, y_val)

            fronts = _non_dominated_sort(combined_obj)
            next_indices = []
            for front in fronts:
                if len(next_indices) + len(front) <= self.pop_size:
                    next_indices.extend(front)
                else:
                    remaining = self.pop_size - len(next_indices)
                    picked = self._niching_select(front, combined_obj, next_indices, remaining)
                    next_indices.extend(picked)
                    break

            population = combined[next_indices]
            pop_obj = combined_obj[next_indices]
            pop_cost = self._weighted_cost(pop_obj)
            best_cost = float(np.min(pop_cost))
            history.append(best_cost)
            print(f"  Best Weighted Cost in population: {best_cost:.4f}")

        final_obj = self._evaluate_population(population, X_train, y_train, X_val, y_val)
        fronts = _non_dominated_sort(final_obj)
        first_front = fronts[0]
        front_cost = self._weighted_cost(final_obj[first_front])
        best_local_idx = int(np.argmin(front_cost))
        best_particle = population[first_front[best_local_idx]].copy()
        best_cost = float(front_cost[best_local_idx])

        print("NSGA-III Optimization Finished!")
        return best_particle, best_cost, history


def run_nsga3(X_train, y_train, X_val, y_val, pop_size=30, num_iterations=30, divisions=6):
    num_features = X_train.shape[1]
    nsga3 = NSGAIII(
        num_features=num_features,
        pop_size=pop_size,
        num_iterations=num_iterations,
        divisions=divisions,
    )

    best_particle, best_cost, history = nsga3.optimize(X_train, y_train, X_val, y_val)
    metrics, model, mask = evaluate_final_model(best_particle, X_train, y_train, X_val, y_val)
    return metrics, model, mask, history
