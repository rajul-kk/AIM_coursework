import numpy as np
from core.fitness import evaluate_solution, evaluate_final_model

class PSO:
    def __init__(self, num_features, num_particles=10, num_iterations=10):
        self.num_features = num_features
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = num_features + 3 # Features + 3 Hyperparameters
        
        # Hyperparameters for PSO
        self.w = 0.5  # Inertia
        self.c1 = 1.5 # Cognitive (particle best)
        self.c2 = 1.5 # Social (global best)
        
    def optimize(self, X_train, y_train, X_val, y_val):
        print(f"\n--- Starting PSO Optimization ({self.num_iterations} iterations, {self.num_particles} particles) ---")
        
        # Initialize particles [0, 1]
        particles = np.random.rand(self.num_particles, self.dim)
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        
        pbest = particles.copy()
        pbest_costs = np.full(self.num_particles, np.inf)
        
        gbest = None
        gbest_cost = np.inf
        history = []
        
        for i in range(self.num_iterations):
            print(f"PSO Iteration {i+1}/{self.num_iterations}")
            for p in range(self.num_particles):
                cost = evaluate_solution(particles[p], X_train, y_train, X_val, y_val)
                
                if cost < pbest_costs[p]:
                    pbest_costs[p] = cost
                    pbest[p] = particles[p].copy()
                    
                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest = particles[p].copy()
            
            # Update velocities and positions
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (pbest - particles) + 
                          self.c2 * r2 * (gbest - particles))
            
            particles = particles + velocities
            # Clip between 0 and 1
            particles = np.clip(particles, 0.0, 1.0)
            
            print(f"  Best Cost so far: {gbest_cost:.4f}")
            history.append(gbest_cost)
            
        print("PSO Optimization Finished!")
        return gbest, gbest_cost, history

def run_pso(X_train, y_train, X_val, y_val, num_particles=10, num_iterations=10):
    """
    Helper function to run PSO and evaluate the final model.
    """
    num_features = X_train.shape[1]
    pso = PSO(num_features=num_features, num_particles=num_particles, num_iterations=num_iterations)
    
    best_particle, best_cost, history = pso.optimize(X_train, y_train, X_val, y_val)
    
    # After optimization, evaluate on validation set (or a separate test set if provided)
    metrics, model, mask = evaluate_final_model(best_particle, X_train, y_train, X_val, y_val)
    return metrics, model, mask, history
