import numpy as np
from core.fitness import evaluate_solution, evaluate_final_model

class GWO:
    def __init__(self, num_features, num_wolves=10, num_iterations=10):
        self.num_features = num_features
        self.num_wolves = num_wolves
        self.num_iterations = num_iterations
        self.dim = num_features + 3 # Features + 3 Hyperparameters
        
    def optimize(self, X_train, y_train, X_val, y_val):
        print(f"\n--- Starting GWO Optimization ({self.num_iterations} iterations, {self.num_wolves} wolves) ---")
        
        # Initialize alpha, beta, and delta positions and their fitness scores
        alpha_pos = np.zeros(self.dim)
        alpha_score = float("inf")
        
        beta_pos = np.zeros(self.dim)
        beta_score = float("inf")
        
        delta_pos = np.zeros(self.dim)
        delta_score = float("inf")
        
        # Initialize wolf positions [0, 1]
        positions = np.random.rand(self.num_wolves, self.dim)
        history = []
        
        for iteration in range(self.num_iterations):
            print(f"GWO Iteration {iteration+1}/{self.num_iterations}")
            
            # Calculate fitness for each wolf
            for idx in range(self.num_wolves):
                # Ensure boundary
                positions[idx] = np.clip(positions[idx], 0.0, 1.0)
                
                fitness = evaluate_solution(positions[idx], X_train, y_train, X_val, y_val)
                
                # Update Alpha, Beta, Delta
                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = positions[idx].copy()
                elif fitness > alpha_score and fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = positions[idx].copy()
                elif fitness > alpha_score and fitness > beta_score and fitness < delta_score:
                    delta_score = fitness
                    delta_pos = positions[idx].copy()
            
            print(f"  Alpha (Best) Score so far: {alpha_score:.4f}")
            
            # a decreases linearly fron 2 to 0
            a = 2 - iteration * ((2) / self.num_iterations)
            
            # Update position of search agents including omegas
            for idx in range(self.num_wolves):
                for j in range(self.dim):
                    
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[idx, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[idx, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[idx, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    positions[idx, j] = (X1 + X2 + X3) / 3
            history.append(alpha_score)
        
        print("GWO Optimization Finished!")
        return alpha_pos, alpha_score, history

def run_gwo(X_train, y_train, X_val, y_val, num_wolves=10, num_iterations=10):
    """
    Helper function to run GWO and evaluate the final model.
    """
    num_features = X_train.shape[1]
    gwo = GWO(num_features=num_features, num_wolves=num_wolves, num_iterations=num_iterations)
    
    best_particle, best_cost, history = gwo.optimize(X_train, y_train, X_val, y_val)
    
    # After optimization, evaluate on validation set (or a separate test set if provided)
    metrics, model, mask = evaluate_final_model(best_particle, X_train, y_train, X_val, y_val)
    return metrics, model, mask, history
