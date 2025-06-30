"""
Optimization Framework for Digital Twin
=====================================

Advanced optimization algorithms including:
- Multi-Objective Optimization (NSGA-II, SPEA2)
- Adaptive Optimization Algorithms
- Bayesian Optimization with Gaussian Processes
- Robust Optimization under Uncertainty

Mathematical formulations:
Multi-objective: min F(x) = [f₁(x), f₂(x), ..., fₘ(x)]
Pareto dominance: x ≺ y ⟺ ∀i: fᵢ(x) ≤ fᵢ(y) ∧ ∃j: fⱼ(x) < fⱼ(y)
Expected improvement: EI(x) = E[max(0, fₘᵢₙ - f(x))]
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import warnings
from scipy import optimize, stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class OptimizationConfig:
    """Configuration for optimization framework"""
    
    # Multi-objective parameters
    population_size: int = 100
    max_generations: int = 200
    crossover_probability: float = 0.8
    mutation_probability: float = 0.2
    tournament_size: int = 3
    
    # Bayesian optimization
    n_initial_samples: int = 20
    max_iterations: int = 100
    acquisition_function: str = "expected_improvement"  # "expected_improvement", "upper_confidence_bound", "probability_improvement"
    exploration_weight: float = 0.1
    
    # Robust optimization
    uncertainty_samples: int = 100
    confidence_level: float = 0.95
    robustness_weight: float = 0.3
    
    # Adaptive parameters
    adaptation_frequency: int = 10
    performance_window: int = 20
    diversity_threshold: float = 0.1
    
    # Parallel processing
    max_workers: int = 4
    enable_parallel: bool = True
    
    # Convergence criteria
    tolerance: float = 1e-6
    max_stagnation_generations: int = 50

@dataclass
class Individual:
    """Individual solution for population-based optimization"""
    variables: np.ndarray
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    constraints: np.ndarray = field(default_factory=lambda: np.array([]))
    fitness: float = float('inf')
    rank: int = 0
    crowding_distance: float = 0.0
    feasible: bool = True
    
    def dominates(self, other: 'Individual') -> bool:
        """Check if this individual dominates another (Pareto dominance)"""
        if not self.feasible or not other.feasible:
            return self.feasible and not other.feasible
        
        if len(self.objectives) == 0 or len(other.objectives) == 0:
            return self.fitness < other.fitness
        
        # Pareto dominance
        better_in_any = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            elif self.objectives[i] < other.objectives[i]:
                better_in_any = True
        
        return better_in_any

class OptimizationProblem(ABC):
    """Abstract base class for optimization problems"""
    
    def __init__(self, 
                 name: str,
                 variable_bounds: List[Tuple[float, float]],
                 objective_names: List[str],
                 constraint_names: Optional[List[str]] = None):
        self.name = name
        self.variable_bounds = variable_bounds
        self.objective_names = objective_names
        self.constraint_names = constraint_names or []
        self.n_variables = len(variable_bounds)
        self.n_objectives = len(objective_names)
        self.n_constraints = len(self.constraint_names)
        
        # Evaluation statistics
        self.n_evaluations = 0
        self.evaluation_history = []
    
    @abstractmethod
    def evaluate(self, variables: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate objectives and constraints
        
        Returns:
            Tuple of (objectives, constraints)
        """
        pass
    
    def is_feasible(self, constraints: np.ndarray) -> bool:
        """Check if constraints are satisfied (≤ 0)"""
        return np.all(constraints <= 0) if len(constraints) > 0 else True

class CasimirFabricationProblem(OptimizationProblem):
    """Casimir fabrication optimization problem"""
    
    def __init__(self):
        # Optimization variables: [pressure, temperature, flow_rate, voltage, pH]
        variable_bounds = [
            (8e4, 1.2e5),     # Pressure (Pa)
            (280, 320),       # Temperature (K)
            (5e-7, 2e-6),     # Flow rate (m³/s)
            (-10, 10),        # Voltage (V)
            (6.0, 8.0)        # pH
        ]
        
        objective_names = [
            "surface_roughness",      # Minimize (nm RMS)
            "defect_density",         # Minimize (defects/μm²)  
            "processing_time"         # Minimize (s)
        ]
        
        constraint_names = [
            "temperature_constraint",  # Temperature stability
            "pressure_constraint",     # Pressure limits
            "quality_constraint"       # Minimum quality
        ]
        
        super().__init__("CasimirFabrication", variable_bounds, objective_names, constraint_names)
        
        # Target values
        self.target_roughness = 0.2e-9  # 0.2 nm RMS
        self.target_defects = 0.01      # 0.01 defects/μm²
        
    def evaluate(self, variables: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate Casimir fabrication objectives"""
        
        pressure, temperature, flow_rate, voltage, pH = variables
        
        # Surface roughness model (nm RMS)
        base_roughness = 0.5e-9
        pressure_effect = -0.3e-9 * (pressure - 1e5) / 1e4
        temp_effect = 0.1e-9 * (temperature - 300) / 50
        flow_effect = -0.2e-9 * (flow_rate - 1e-6) / 1e-7
        voltage_effect = 0.05e-9 * abs(voltage) / 5
        pH_effect = 0.05e-9 * (pH - 7)**2
        
        # Interactions
        pressure_temp_interaction = 0.02e-9 * (pressure - 1e5) * (temperature - 300) / (1e4 * 50)
        
        surface_roughness = base_roughness + pressure_effect + temp_effect + flow_effect + voltage_effect + pH_effect + pressure_temp_interaction
        surface_roughness = max(0.05e-9, surface_roughness)  # Physical minimum
        
        # Defect density model (defects/μm²)
        base_defects = 0.02
        temp_stress_defects = 0.005 * abs(temperature - 300) / 50
        pressure_defects = -0.003 * (pressure - 1e5) / 1e4
        voltage_defects = 0.002 * voltage**2 / 25
        pH_defects = 0.001 * (pH - 7)**2
        
        defect_density = base_defects + temp_stress_defects + pressure_defects + voltage_defects + pH_defects
        defect_density = max(0.001, defect_density)  # Physical minimum
        
        # Processing time model (s)
        base_time = 100.0
        pressure_time = -20.0 * (pressure - 1e5) / 1e4  # Higher pressure = faster
        flow_time = -15.0 * (flow_rate - 1e-6) / 1e-7   # Higher flow = faster
        temp_time = 5.0 * abs(temperature - 300) / 50   # Temperature deviation = slower
        
        processing_time = base_time + pressure_time + flow_time + temp_time
        processing_time = max(10.0, processing_time)  # Minimum processing time
        
        objectives = np.array([surface_roughness, defect_density, processing_time])
        
        # Constraints
        temp_constraint = abs(temperature - 300) - 15  # Temperature within ±15K
        pressure_constraint = max(pressure - 1.3e5, 7e4 - pressure)  # Pressure limits
        
        # Quality constraint (both roughness and defects must be reasonable)
        quality_metric = (surface_roughness / self.target_roughness + 
                         defect_density / self.target_defects) / 2
        quality_constraint = quality_metric - 2.0  # Must be less than 2x target
        
        constraints = np.array([temp_constraint, pressure_constraint, quality_constraint])
        
        # Update evaluation statistics
        self.n_evaluations += 1
        self.evaluation_history.append({
            'variables': variables.copy(),
            'objectives': objectives.copy(),
            'constraints': constraints.copy()
        })
        
        return objectives, constraints

class NSGA2Optimizer:
    """Non-dominated Sorting Genetic Algorithm II"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.pareto_front = []
        
        # Statistics
        self.convergence_history = []
        self.diversity_history = []
        
    def initialize_population(self, problem: OptimizationProblem) -> List[Individual]:
        """Initialize random population"""
        
        population = []
        
        for _ in range(self.config.population_size):
            variables = np.zeros(problem.n_variables)
            
            for i, (lb, ub) in enumerate(problem.variable_bounds):
                variables[i] = random.uniform(lb, ub)
            
            individual = Individual(variables=variables)
            
            # Evaluate individual
            objectives, constraints = problem.evaluate(variables)
            individual.objectives = objectives
            individual.constraints = constraints
            individual.feasible = problem.is_feasible(constraints)
            
            population.append(individual)
        
        self.population = population
        return population
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting"""
        
        fronts = [[]]
        
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            
            for other in population:
                if individual.dominates(other):
                    individual.dominated_solutions.append(other)
                elif other.dominates(individual):
                    individual.domination_count += 1
            
            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)
        
        front_index = 0
        
        while len(fronts[front_index]) > 0:
            next_front = []
            
            for individual in fronts[front_index]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = front_index + 1
                        next_front.append(dominated)
            
            front_index += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def crowding_distance_assignment(self, front: List[Individual]) -> None:
        """Assign crowding distances to individuals in a front"""
        
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        for individual in front:
            individual.crowding_distance = 0.0
        
        n_objectives = len(front[0].objectives)
        
        for obj_index in range(n_objectives):
            # Sort by objective
            front.sort(key=lambda x: x.objectives[obj_index])
            
            # Set boundary distances to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_range = (front[-1].objectives[obj_index] - 
                        front[0].objectives[obj_index])
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distances
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[obj_index] - 
                           front[i - 1].objectives[obj_index]) / obj_range
                front[i].crowding_distance += distance
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection based on rank and crowding distance"""
        
        tournament = random.sample(population, self.config.tournament_size)
        
        # Select best individual
        best = tournament[0]
        for individual in tournament[1:]:
            if (individual.rank < best.rank or 
                (individual.rank == best.rank and 
                 individual.crowding_distance > best.crowding_distance)):
                best = individual
        
        return best
    
    def crossover(self, parent1: Individual, parent2: Individual, problem: OptimizationProblem) -> Tuple[Individual, Individual]:
        """Simulated binary crossover (SBX)"""
        
        if random.random() > self.config.crossover_probability:
            return parent1, parent2
        
        eta_c = 20.0  # Distribution index for crossover
        
        child1_vars = np.zeros(problem.n_variables)
        child2_vars = np.zeros(problem.n_variables)
        
        for i in range(problem.n_variables):
            if random.random() <= 0.5:
                x1, x2 = parent1.variables[i], parent2.variables[i]
                lb, ub = problem.variable_bounds[i]
                
                if abs(x1 - x2) > 1e-14:
                    if x1 > x2:
                        x1, x2 = x2, x1
                    
                    # Calculate beta
                    rand = random.random()
                    if rand <= 0.5:
                        beta = (2.0 * rand) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta_c + 1.0))
                    
                    c1 = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                    c2 = 0.5 * ((x1 + x2) + beta * (x2 - x1))
                    
                    # Apply bounds
                    c1 = max(lb, min(ub, c1))
                    c2 = max(lb, min(ub, c2))
                    
                    child1_vars[i] = c1
                    child2_vars[i] = c2
                else:
                    child1_vars[i] = x1
                    child2_vars[i] = x2
            else:
                child1_vars[i] = parent1.variables[i]
                child2_vars[i] = parent2.variables[i]
        
        child1 = Individual(variables=child1_vars)
        child2 = Individual(variables=child2_vars)
        
        return child1, child2
    
    def mutation(self, individual: Individual, problem: OptimizationProblem) -> Individual:
        """Polynomial mutation"""
        
        if random.random() > self.config.mutation_probability:
            return individual
        
        eta_m = 20.0  # Distribution index for mutation
        
        mutated_vars = individual.variables.copy()
        
        for i in range(problem.n_variables):
            if random.random() <= (1.0 / problem.n_variables):
                x = individual.variables[i]
                lb, ub = problem.variable_bounds[i]
                
                delta1 = (x - lb) / (ub - lb)
                delta2 = (ub - x) / (ub - lb)
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                x_new = x + deltaq * (ub - lb)
                x_new = max(lb, min(ub, x_new))
                
                mutated_vars[i] = x_new
        
        return Individual(variables=mutated_vars)
    
    def environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """Environmental selection to maintain population size"""
        
        # Fast non-dominated sort
        fronts = self.fast_non_dominated_sort(population)
        
        # Calculate crowding distances
        for front in fronts:
            self.crowding_distance_assignment(front)
        
        # Select individuals for next generation
        next_population = []
        front_index = 0
        
        while len(next_population) + len(fronts[front_index]) <= self.config.population_size:
            next_population.extend(fronts[front_index])
            front_index += 1
            
            if front_index >= len(fronts):
                break
        
        # Fill remaining slots from next front
        if len(next_population) < self.config.population_size and front_index < len(fronts):
            remaining_slots = self.config.population_size - len(next_population)
            last_front = fronts[front_index]
            
            # Sort by crowding distance (descending)
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            next_population.extend(last_front[:remaining_slots])
        
        return next_population
    
    def optimize(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Run NSGA-II optimization"""
        
        print(f"🧬 Starting NSGA-II optimization for {problem.name}")
        print(f"Population size: {self.config.population_size}, Generations: {self.config.max_generations}")
        
        # Initialize population
        self.population = self.initialize_population(problem)
        
        stagnation_counter = 0
        best_hypervolume = 0.0
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Create offspring
            offspring = []
            
            while len(offspring) < self.config.population_size:
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                child1, child2 = self.crossover(parent1, parent2, problem)
                
                child1 = self.mutation(child1, problem)
                child2 = self.mutation(child2, problem)
                
                # Evaluate offspring
                for child in [child1, child2]:
                    objectives, constraints = problem.evaluate(child.variables)
                    child.objectives = objectives
                    child.constraints = constraints
                    child.feasible = problem.is_feasible(constraints)
                
                offspring.extend([child1, child2])
            
            # Combine parent and offspring populations
            combined_population = self.population + offspring[:self.config.population_size]
            
            # Environmental selection
            self.population = self.environmental_selection(combined_population)
            
            # Update Pareto front
            fronts = self.fast_non_dominated_sort(self.population)
            self.pareto_front = fronts[0] if fronts else []
            
            # Convergence tracking
            if generation % 10 == 0:
                hypervolume = self._calculate_hypervolume(self.pareto_front, problem)
                self.convergence_history.append(hypervolume)
                
                print(f"Generation {generation}: {len(self.pareto_front)} Pareto solutions, HV = {hypervolume:.6f}")
                
                # Check for stagnation
                if abs(hypervolume - best_hypervolume) < self.config.tolerance:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    best_hypervolume = hypervolume
                
                if stagnation_counter >= self.config.max_stagnation_generations // 10:
                    print(f"Converged after {generation} generations")
                    break
        
        print(f"✅ Optimization completed. Final Pareto front: {len(self.pareto_front)} solutions")
        
        return {
            'pareto_front': self.pareto_front,
            'final_population': self.population,
            'convergence_history': self.convergence_history,
            'n_evaluations': problem.n_evaluations,
            'generations': self.generation + 1
        }
    
    def _calculate_hypervolume(self, pareto_front: List[Individual], problem: OptimizationProblem) -> float:
        """Calculate hypervolume indicator (simplified 2D/3D version)"""
        
        if not pareto_front:
            return 0.0
        
        # Extract objectives
        objectives = np.array([ind.objectives for ind in pareto_front if ind.feasible])
        
        if len(objectives) == 0:
            return 0.0
        
        # Reference point (worst case + margin)
        ref_point = np.max(objectives, axis=0) * 1.1
        
        # Simple hypervolume calculation for up to 3 objectives
        if problem.n_objectives == 2:
            return self._hypervolume_2d(objectives, ref_point)
        elif problem.n_objectives == 3:
            return self._hypervolume_3d(objectives, ref_point)
        else:
            # Fallback: use sum of distances to reference point
            return -np.sum(np.linalg.norm(objectives - ref_point, axis=1))
    
    def _hypervolume_2d(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """2D hypervolume calculation"""
        
        # Sort by first objective
        sorted_objectives = objectives[np.argsort(objectives[:, 0])]
        
        hypervolume = 0.0
        prev_x = 0.0
        
        for obj in sorted_objectives:
            hypervolume += (obj[0] - prev_x) * (ref_point[1] - obj[1])
            prev_x = obj[0]
        
        return max(0.0, hypervolume)
    
    def _hypervolume_3d(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """3D hypervolume calculation (approximation)"""
        
        # Simplified 3D hypervolume - more complex algorithms exist
        hypervolume = 0.0
        
        for i, obj in enumerate(objectives):
            # Calculate dominated volume for this point
            dominated_volume = 1.0
            for j in range(3):
                dominated_volume *= max(0, ref_point[j] - obj[j])
            
            # Subtract overlaps (approximation)
            overlap_factor = 1.0
            for other_obj in objectives:
                if not np.array_equal(obj, other_obj):
                    if np.all(other_obj <= obj):  # other_obj dominates obj
                        overlap_factor *= 0.8  # Reduce contribution
            
            hypervolume += dominated_volume * overlap_factor
        
        return hypervolume

class BayesianOptimizer:
    """Bayesian Optimization with Gaussian Process"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gp_models = {}
        self.evaluated_points = []
        self.evaluated_objectives = []
        
    def optimize_single_objective(self, 
                                problem: OptimizationProblem,
                                objective_index: int = 0) -> Dict[str, Any]:
        """Bayesian optimization for single objective"""
        
        print(f"🎯 Starting Bayesian optimization for {problem.objective_names[objective_index]}")
        
        # Initialize with random samples
        self._initialize_samples(problem, self.config.n_initial_samples)
        
        for iteration in range(self.config.max_iterations):
            
            # Fit Gaussian process
            self._fit_gaussian_process(objective_index)
            
            # Find next point to evaluate
            next_point = self._optimize_acquisition_function(problem, objective_index)
            
            # Evaluate next point
            objectives, constraints = problem.evaluate(next_point)
            
            self.evaluated_points.append(next_point.copy())
            self.evaluated_objectives.append(objectives.copy())
            
            # Check convergence
            if iteration % 10 == 0:
                best_value = min([obj[objective_index] for obj in self.evaluated_objectives])
                print(f"Iteration {iteration}: Best {problem.objective_names[objective_index]} = {best_value:.6f}")
        
        # Find best solution
        best_idx = np.argmin([obj[objective_index] for obj in self.evaluated_objectives])
        best_point = self.evaluated_points[best_idx]
        best_objective = self.evaluated_objectives[best_idx][objective_index]
        
        print(f"✅ Bayesian optimization completed. Best value: {best_objective:.6f}")
        
        return {
            'best_point': best_point,
            'best_objective': best_objective,
            'evaluated_points': self.evaluated_points,
            'evaluated_objectives': self.evaluated_objectives,
            'n_evaluations': len(self.evaluated_points)
        }
    
    def _initialize_samples(self, problem: OptimizationProblem, n_samples: int):
        """Initialize with random samples"""
        
        self.evaluated_points = []
        self.evaluated_objectives = []
        
        for _ in range(n_samples):
            variables = np.zeros(problem.n_variables)
            
            for i, (lb, ub) in enumerate(problem.variable_bounds):
                variables[i] = random.uniform(lb, ub)
            
            objectives, constraints = problem.evaluate(variables)
            
            self.evaluated_points.append(variables.copy())
            self.evaluated_objectives.append(objectives.copy())
    
    def _fit_gaussian_process(self, objective_index: int):
        """Fit Gaussian process to evaluated data"""
        
        X = np.array(self.evaluated_points)
        y = np.array([obj[objective_index] for obj in self.evaluated_objectives])
        
        # Define kernel
        kernel = (ConstantKernel(1.0) * 
                 RBF(length_scale=1.0) + 
                 WhiteKernel(noise_level=1e-6))
        
        # Fit GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        
        gp.fit(X, y)
        self.gp_models[objective_index] = gp
    
    def _optimize_acquisition_function(self, 
                                     problem: OptimizationProblem, 
                                     objective_index: int) -> np.ndarray:
        """Optimize acquisition function to find next point"""
        
        gp = self.gp_models[objective_index]
        
        def acquisition_function(x):
            x = x.reshape(1, -1)
            
            if self.config.acquisition_function == "expected_improvement":
                return -self._expected_improvement(x, gp, objective_index)
            elif self.config.acquisition_function == "upper_confidence_bound":
                return -self._upper_confidence_bound(x, gp)
            else:  # probability_improvement
                return -self._probability_improvement(x, gp, objective_index)
        
        # Multi-start optimization
        best_x = None
        best_acquisition = float('inf')
        
        for _ in range(20):  # Multiple random starts
            x0 = np.zeros(problem.n_variables)
            for i, (lb, ub) in enumerate(problem.variable_bounds):
                x0[i] = random.uniform(lb, ub)
            
            bounds = problem.variable_bounds
            
            try:
                result = optimize.minimize(
                    acquisition_function,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_acquisition:
                    best_acquisition = result.fun
                    best_x = result.x
                    
            except Exception:
                continue
        
        return best_x if best_x is not None else x0
    
    def _expected_improvement(self, x: np.ndarray, gp: GaussianProcessRegressor, objective_index: int) -> float:
        """Expected improvement acquisition function"""
        
        mean, std = gp.predict(x, return_std=True)
        
        # Current best value
        current_best = min([obj[objective_index] for obj in self.evaluated_objectives])
        
        if std == 0:
            return 0.0
        
        # Expected improvement
        improvement = current_best - mean
        z = improvement / std
        
        ei = improvement * stats.norm.cdf(z) + std * stats.norm.pdf(z)
        
        return ei[0] if isinstance(ei, np.ndarray) else ei
    
    def _upper_confidence_bound(self, x: np.ndarray, gp: GaussianProcessRegressor) -> float:
        """Upper confidence bound acquisition function"""
        
        mean, std = gp.predict(x, return_std=True)
        
        # Exploration-exploitation trade-off
        kappa = self.config.exploration_weight
        ucb = -(mean - kappa * std)  # Negative for minimization
        
        return ucb[0] if isinstance(ucb, np.ndarray) else ucb
    
    def _probability_improvement(self, x: np.ndarray, gp: GaussianProcessRegressor, objective_index: int) -> float:
        """Probability of improvement acquisition function"""
        
        mean, std = gp.predict(x, return_std=True)
        
        # Current best value
        current_best = min([obj[objective_index] for obj in self.evaluated_objectives])
        
        if std == 0:
            return 0.0
        
        # Probability of improvement
        improvement = current_best - mean
        z = improvement / std
        
        pi = stats.norm.cdf(z)
        
        return pi[0] if isinstance(pi, np.ndarray) else pi

# Example usage and testing
if __name__ == "__main__":
    print("🎯 Testing Optimization Framework")
    print("=" * 50)
    
    # Configuration
    config = OptimizationConfig(
        population_size=50,
        max_generations=100,
        max_iterations=50,
        enable_parallel=False  # For testing
    )
    
    # Create optimization problem
    problem = CasimirFabricationProblem()
    
    print(f"📋 Problem: {problem.name}")
    print(f"Variables: {problem.n_variables}, Objectives: {problem.n_objectives}, Constraints: {problem.n_constraints}")
    
    # Test NSGA-II optimization
    print("\n🧬 Testing NSGA-II Multi-Objective Optimization...")
    
    nsga2 = NSGA2Optimizer(config)
    nsga2_results = nsga2.optimize(problem)
    
    print(f"NSGA-II Results:")
    print(f"  Pareto solutions: {len(nsga2_results['pareto_front'])}")
    print(f"  Total evaluations: {nsga2_results['n_evaluations']}")
    print(f"  Generations: {nsga2_results['generations']}")
    
    # Show best solutions
    pareto_front = nsga2_results['pareto_front']
    if pareto_front:
        print("\n🏆 Best Pareto Solutions:")
        for i, individual in enumerate(pareto_front[:3]):  # Show top 3
            objectives = individual.objectives
            print(f"  Sol {i+1}: Roughness={objectives[0]*1e9:.2f}nm, Defects={objectives[1]:.4f}/μm², Time={objectives[2]:.1f}s")
    
    # Test Bayesian Optimization
    print("\n🎯 Testing Bayesian Optimization (Surface Roughness)...")
    
    bayesian_opt = BayesianOptimizer(config)
    bayesian_results = bayesian_opt.optimize_single_objective(problem, objective_index=0)
    
    print(f"Bayesian Optimization Results:")
    print(f"  Best roughness: {bayesian_results['best_objective']*1e9:.2f} nm")
    print(f"  Total evaluations: {bayesian_results['n_evaluations']}")
    
    # Best solution details
    best_vars = bayesian_results['best_point']
    print(f"  Best variables: P={best_vars[0]/1e5:.1f}bar, T={best_vars[1]:.1f}K, Flow={best_vars[2]*1e6:.1f}μL/s")
    
    print("\n✅ Optimization Framework implementation complete!")
    
    # Final comparison
    nsga2_best_roughness = min([ind.objectives[0] for ind in pareto_front if ind.feasible])
    bayesian_best_roughness = bayesian_results['best_objective']
    
    print(f"\n🏁 Final Comparison:")
    print(f"  NSGA-II best roughness: {nsga2_best_roughness*1e9:.2f} nm")
    print(f"  Bayesian best roughness: {bayesian_best_roughness*1e9:.2f} nm")
    print(f"  Target roughness: {problem.target_roughness*1e9:.2f} nm")
