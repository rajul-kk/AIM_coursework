import numpy as np
from core.fitness import evaluate_solution, evaluate_final_model

class GA:
    def __init__(self, num_features, pop_size=10, num_iterations=10, mutation_rate=0.1):
        self.num_features = num_features
        self.pop_size = pop_size
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.dim = num_features + 3 # Features + 3 Hyperparameters
        
    def optimize(self, X_train, y_train, X_val, y_val):
        print(f"\n--- Starting GA Optimization ({self.num_iterations} iterations, {self.pop_size} population) ---")
        
        # Initialize population [0, 1]
        population = np.random.rand(self.pop_size, self.dim)
        best_sol = None
        best_cost = np.inf
        history = []
        
        for i in range(self.num_iterations):
            print(f"GA Iteration {i+1}/{self.num_iterations}")
            
            # Evaluate fitness
            costs = np.array([evaluate_solution(ind, X_train, y_train, X_val, y_val) for ind in population])
            
            # Update best
            min_idx = np.argmin(costs)
            if costs[min_idx] < best_cost:
                best_cost = costs[min_idx]
                best_sol = population[min_idx].copy()
                
            print(f"  Best Cost so far: {best_cost:.4f}")
            
            # Selection (Tournament of size 3)
            new_population = np.zeros_like(population)
            for j in range(self.pop_size):
                idx = np.random.choice(self.pop_size, 3, replace=False)
                winner = idx[np.argmin(costs[idx])]
                new_population[j] = population[winner].copy()
                
            # Crossover (Single point)
            for j in range(0, self.pop_size, 2):
                if j+1 < self.pop_size and np.random.rand() < 0.8: # 80% crossover rate
                    crossover_pt = np.random.randint(1, self.dim)
                    temp = new_population[j].copy()
                    new_population[j, crossover_pt:] = new_population[j+1, crossover_pt:]
                    new_population[j+1, crossover_pt:] = temp[crossover_pt:]
                    
            # Mutation
            mutation_mask = np.random.rand(self.pop_size, self.dim) < self.mutation_rate
            random_values = np.random.rand(self.pop_size, self.dim)
            new_population = np.where(mutation_mask, random_values, new_population)
            
            population = new_population
            history.append(best_cost)
            
        print("GA Optimization Finished!")
        return best_sol, best_cost, history

def run_ga(X_train, y_train, X_val, y_val, pop_size=10, num_iterations=10):
    """
    Helper function to run GA and evaluate the final model.
    """
    num_features = X_train.shape[1]
    ga = GA(num_features=num_features, pop_size=pop_size, num_iterations=num_iterations)
    
    best_particle, best_cost, history = ga.optimize(X_train, y_train, X_val, y_val)
    
    # After optimization, evaluate on validation set (or a separate test set if provided)
    metrics, model, mask = evaluate_final_model(best_particle, X_train, y_train, X_val, y_val)
    return metrics, model, mask, history
