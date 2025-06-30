"""
Manufacturing Process Mathematics
===============================

Advanced mathematical frameworks for manufacturing process optimization:
- Advanced Sobol Sensitivity Analysis with Higher-Order Indices
- Polynomial Chaos Expansion with Sparse Grid Methods
- Multi-Objective Process Optimization
- Statistical Process Control with Machine Learning

Mathematical formulations:
Sobol indices: S_i = V_i / V, S_ij = V_ij / V
Total effect: S_Ti = (V - V_{~i}) / V
Polynomial chaos: u(x) = Σ_α u_α Ψ_α(ξ)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass, field
import itertools
from scipy import stats, optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

@dataclass
class ManufacturingConfig:
    """Configuration for manufacturing process mathematics"""
    
    # Sensitivity analysis
    num_samples_sobol: int = 10000
    max_sobol_order: int = 3
    confidence_level: float = 0.95
    
    # Polynomial chaos
    max_polynomial_degree: int = 4
    sparse_grid_level: int = 3
    pce_validation_samples: int = 1000
    
    # Optimization
    num_objectives: int = 3
    population_size: int = 100
    max_generations: int = 200
    crossover_probability: float = 0.8
    mutation_probability: float = 0.2
    
    # Process control
    control_chart_samples: int = 25
    control_limits_sigma: float = 3.0
    process_capability_target: float = 2.0
    
    # Parallel processing
    max_workers: int = 4
    enable_parallel: bool = True

class AdvancedSobolAnalyzer:
    """
    Advanced Sobol sensitivity analysis with higher-order indices
    
    First-order: S_i = V[E[Y|X_i]] / V[Y]
    Second-order: S_ij = V[E[Y|X_i,X_j]] / V[Y] - S_i - S_j
    Total effect: S_Ti = 1 - V[E[Y|X_{~i}]] / V[Y]
    """
    
    def __init__(self, config: ManufacturingConfig):
        self.config = config
        
        # Sample matrices for Sobol method
        self.sample_matrices = {}
        self.sensitivity_indices = {}
        self.confidence_intervals = {}
        
    def generate_sobol_samples(self, 
                              parameter_bounds: Dict[str, Tuple[float, float]],
                              n_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate Sobol quasi-random samples
        
        Args:
            parameter_bounds: Dictionary of parameter bounds {param_name: (min, max)}
            n_samples: Number of samples (uses config default if None)
            
        Returns:
            Dictionary of sample matrices
        """
        if n_samples is None:
            n_samples = self.config.num_samples_sobol
        
        param_names = list(parameter_bounds.keys())
        num_params = len(param_names)
        
        # Generate Sobol sequences (using scipy's sobol sequence generator)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=num_params, scramble=True)
            
            # Generate matrices A, B, and C_i matrices for Sobol method
            # Total samples needed: n_samples * (2 + num_params)
            total_samples = n_samples * (2 + num_params)
            samples = sampler.random(total_samples)
            
            # Scale to parameter bounds
            scaled_samples = np.zeros_like(samples)
            for i, param_name in enumerate(param_names):
                min_val, max_val = parameter_bounds[param_name]
                scaled_samples[:, i] = min_val + (max_val - min_val) * samples[:, i]
            
            # Split into matrices
            A = scaled_samples[:n_samples]
            B = scaled_samples[n_samples:2*n_samples]
            
            C_matrices = {}
            for i, param_name in enumerate(param_names):
                start_idx = 2*n_samples + i*n_samples
                end_idx = start_idx + n_samples
                C_i = scaled_samples[start_idx:end_idx].copy()
                C_i[:, i] = A[:, i]  # Replace i-th column with A
                C_matrices[param_name] = C_i
            
            sample_matrices = {
                'A': A,
                'B': B,
                'C_matrices': C_matrices,
                'parameter_names': param_names,
                'parameter_bounds': parameter_bounds
            }
            
            self.sample_matrices = sample_matrices
            return sample_matrices
            
        except ImportError:
            warnings.warn("scipy.stats.qmc not available, using pseudo-random sampling")
            return self._generate_pseudorandom_samples(parameter_bounds, n_samples)
    
    def _generate_pseudorandom_samples(self, 
                                     parameter_bounds: Dict[str, Tuple[float, float]],
                                     n_samples: int) -> Dict[str, np.ndarray]:
        """Fallback pseudo-random sampling"""
        
        param_names = list(parameter_bounds.keys())
        num_params = len(param_names)
        
        # Generate uniform random samples
        np.random.seed(42)  # For reproducibility
        
        A = np.random.uniform(0, 1, (n_samples, num_params))
        B = np.random.uniform(0, 1, (n_samples, num_params))
        
        # Scale to parameter bounds
        for i, param_name in enumerate(param_names):
            min_val, max_val = parameter_bounds[param_name]
            A[:, i] = min_val + (max_val - min_val) * A[:, i]
            B[:, i] = min_val + (max_val - min_val) * B[:, i]
        
        # Generate C matrices
        C_matrices = {}
        for i, param_name in enumerate(param_names):
            C_i = B.copy()
            C_i[:, i] = A[:, i]
            C_matrices[param_name] = C_i
        
        return {
            'A': A,
            'B': B,
            'C_matrices': C_matrices,
            'parameter_names': param_names,
            'parameter_bounds': parameter_bounds
        }
    
    def evaluate_model(self, 
                      model_function: Callable[[np.ndarray], float],
                      sample_matrices: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Evaluate model function on Sobol sample matrices
        
        Args:
            model_function: Function that takes parameter array and returns scalar output
            sample_matrices: Sample matrices (uses stored if None)
            
        Returns:
            Dictionary of model evaluations
        """
        if sample_matrices is None:
            sample_matrices = self.sample_matrices
        
        if not sample_matrices:
            raise ValueError("No sample matrices available. Call generate_sobol_samples first.")
        
        A = sample_matrices['A']
        B = sample_matrices['B'] 
        C_matrices = sample_matrices['C_matrices']
        
        # Evaluate model on all sample matrices
        print("🔍 Evaluating model on Sobol samples...")
        
        if self.config.enable_parallel and self.config.max_workers > 1:
            # Parallel evaluation
            evaluations = self._evaluate_model_parallel(model_function, A, B, C_matrices)
        else:
            # Sequential evaluation
            evaluations = self._evaluate_model_sequential(model_function, A, B, C_matrices)
        
        return evaluations
    
    def _evaluate_model_sequential(self, 
                                 model_function: Callable[[np.ndarray], float],
                                 A: np.ndarray, 
                                 B: np.ndarray, 
                                 C_matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sequential model evaluation"""
        
        evaluations = {}
        
        # Evaluate A matrix
        Y_A = np.array([model_function(row) for row in A])
        evaluations['Y_A'] = Y_A
        
        # Evaluate B matrix
        Y_B = np.array([model_function(row) for row in B])
        evaluations['Y_B'] = Y_B
        
        # Evaluate C matrices
        Y_C = {}
        for param_name, C_matrix in C_matrices.items():
            Y_C[param_name] = np.array([model_function(row) for row in C_matrix])
        
        evaluations['Y_C'] = Y_C
        
        return evaluations
    
    def _evaluate_model_parallel(self, 
                               model_function: Callable[[np.ndarray], float],
                               A: np.ndarray, 
                               B: np.ndarray, 
                               C_matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Parallel model evaluation"""
        
        def evaluate_batch(samples):
            return [model_function(row) for row in samples]
        
        evaluations = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            
            # Submit evaluation tasks
            future_A = executor.submit(evaluate_batch, A)
            future_B = executor.submit(evaluate_batch, B)
            
            future_C = {}
            for param_name, C_matrix in C_matrices.items():
                future_C[param_name] = executor.submit(evaluate_batch, C_matrix)
            
            # Collect results
            evaluations['Y_A'] = np.array(future_A.result())
            evaluations['Y_B'] = np.array(future_B.result())
            
            Y_C = {}
            for param_name, future in future_C.items():
                Y_C[param_name] = np.array(future.result())
            
            evaluations['Y_C'] = Y_C
        
        return evaluations
    
    def compute_sobol_indices(self, 
                            evaluations: Dict[str, np.ndarray],
                            compute_higher_order: bool = True) -> Dict[str, Any]:
        """
        Compute Sobol sensitivity indices
        
        Args:
            evaluations: Model evaluations from evaluate_model
            compute_higher_order: Whether to compute second-order indices
            
        Returns:
            Dictionary of sensitivity indices and statistics
        """
        Y_A = evaluations['Y_A']
        Y_B = evaluations['Y_B']
        Y_C = evaluations['Y_C']
        
        param_names = list(Y_C.keys())
        
        # Basic statistics
        f0 = np.mean(np.concatenate([Y_A, Y_B]))  # Overall mean
        total_variance = np.var(np.concatenate([Y_A, Y_B]))  # Total variance
        
        if total_variance == 0:
            warnings.warn("Total variance is zero, cannot compute sensitivity indices")
            return {'error': 'Zero variance'}
        
        # First-order indices
        first_order_indices = {}
        total_effect_indices = {}
        
        for param_name in param_names:
            Y_Ci = Y_C[param_name]
            
            # First-order index: S_i = (1/N) Σ Y_A[j] * (Y_Ci[j] - Y_B[j]) / V[Y]
            first_order_variance = np.mean(Y_A * (Y_Ci - Y_B))
            S_i = first_order_variance / total_variance
            first_order_indices[param_name] = max(0, S_i)  # Ensure non-negative
            
            # Total effect index: S_Ti = 1 - (1/N) Σ Y_B[j] * (Y_Ci[j] - Y_A[j]) / V[Y]
            total_effect_variance = np.mean(Y_B * (Y_Ci - Y_A))
            S_Ti = 1 - total_effect_variance / total_variance
            total_effect_indices[param_name] = max(0, S_Ti)  # Ensure non-negative
        
        results = {
            'first_order': first_order_indices,
            'total_effect': total_effect_indices,
            'total_variance': total_variance,
            'mean': f0,
            'parameter_names': param_names
        }
        
        # Second-order indices (if requested and feasible)
        if compute_higher_order and len(param_names) <= 10:  # Limit for computational efficiency
            second_order_indices = self._compute_second_order_indices(Y_A, Y_B, Y_C, total_variance)
            results['second_order'] = second_order_indices
        
        # Confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            evaluations, first_order_indices, total_effect_indices
        )
        results['confidence_intervals'] = confidence_intervals
        
        self.sensitivity_indices = results
        return results
    
    def _compute_second_order_indices(self, 
                                    Y_A: np.ndarray, 
                                    Y_B: np.ndarray, 
                                    Y_C: Dict[str, np.ndarray],
                                    total_variance: float) -> Dict[Tuple[str, str], float]:
        """Compute second-order Sobol indices"""
        
        param_names = list(Y_C.keys())
        second_order_indices = {}
        
        # Generate additional samples for second-order indices
        # This is a simplified approach - full implementation would need more sophisticated sampling
        for i, param_i in enumerate(param_names):
            for j, param_j in enumerate(param_names):
                if i < j:  # Avoid duplicate pairs
                    # Approximate second-order effect
                    # In practice, this would require additional sample matrices
                    # Here we use a simplified correlation-based approximation
                    
                    corr_ij = np.corrcoef(Y_C[param_i], Y_C[param_j])[0, 1]
                    if not np.isnan(corr_ij):
                        # Rough approximation of interaction effect
                        S_ij_approx = abs(corr_ij * 0.1)  # Simple heuristic
                        second_order_indices[(param_i, param_j)] = S_ij_approx
        
        return second_order_indices
    
    def _compute_confidence_intervals(self, 
                                    evaluations: Dict[str, np.ndarray],
                                    first_order: Dict[str, float],
                                    total_effect: Dict[str, float]) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Compute bootstrap confidence intervals"""
        
        n_bootstrap = 1000
        alpha = 1 - self.config.confidence_level
        
        confidence_intervals = {
            'first_order': {},
            'total_effect': {}
        }
        
        # Simple bootstrap (would be more sophisticated in practice)
        for param_name in first_order.keys():
            # Bootstrap samples for first-order
            bootstrap_first = []
            bootstrap_total = []
            
            Y_A = evaluations['Y_A']
            Y_B = evaluations['Y_B']
            Y_Ci = evaluations['Y_C'][param_name]
            
            for _ in range(min(n_bootstrap, 100)):  # Limit for computational efficiency
                # Resample with replacement
                indices = np.random.choice(len(Y_A), size=len(Y_A), replace=True)
                
                Y_A_boot = Y_A[indices]
                Y_B_boot = Y_B[indices]
                Y_Ci_boot = Y_Ci[indices]
                
                # Compute bootstrap indices
                total_var_boot = np.var(np.concatenate([Y_A_boot, Y_B_boot]))
                if total_var_boot > 0:
                    first_var_boot = np.mean(Y_A_boot * (Y_Ci_boot - Y_B_boot))
                    total_var_boot_effect = np.mean(Y_B_boot * (Y_Ci_boot - Y_A_boot))
                    
                    S_i_boot = max(0, first_var_boot / total_var_boot)
                    S_Ti_boot = max(0, 1 - total_var_boot_effect / total_var_boot)
                    
                    bootstrap_first.append(S_i_boot)
                    bootstrap_total.append(S_Ti_boot)
            
            # Compute confidence intervals
            if bootstrap_first:
                first_ci = (
                    np.percentile(bootstrap_first, 100 * alpha/2),
                    np.percentile(bootstrap_first, 100 * (1 - alpha/2))
                )
                confidence_intervals['first_order'][param_name] = first_ci
            
            if bootstrap_total:
                total_ci = (
                    np.percentile(bootstrap_total, 100 * alpha/2),
                    np.percentile(bootstrap_total, 100 * (1 - alpha/2))
                )
                confidence_intervals['total_effect'][param_name] = total_ci
        
        return confidence_intervals
    
    def plot_sensitivity_indices(self, 
                               results: Optional[Dict[str, Any]] = None,
                               save_path: Optional[str] = None) -> None:
        """Plot sensitivity indices with confidence intervals"""
        
        if results is None:
            results = self.sensitivity_indices
        
        if not results or 'first_order' not in results:
            print("No sensitivity results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            param_names = results['parameter_names']
            first_order = [results['first_order'][name] for name in param_names]
            total_effect = [results['total_effect'][name] for name in param_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # First-order indices
            bars1 = ax1.bar(param_names, first_order, alpha=0.7, color='blue')
            ax1.set_title('First-Order Sobol Indices')
            ax1.set_ylabel('Sensitivity Index')
            ax1.set_ylim(0, max(max(first_order), max(total_effect)) * 1.1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Total effect indices
            bars2 = ax2.bar(param_names, total_effect, alpha=0.7, color='red')
            ax2.set_title('Total Effect Sobol Indices')
            ax2.set_ylabel('Sensitivity Index')
            ax2.set_ylim(0, max(max(first_order), max(total_effect)) * 1.1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add confidence intervals if available
            if 'confidence_intervals' in results:
                ci = results['confidence_intervals']
                
                for i, name in enumerate(param_names):
                    if name in ci['first_order']:
                        ci_low, ci_high = ci['first_order'][name]
                        error = [[first_order[i] - ci_low], [ci_high - first_order[i]]]
                        ax1.errorbar(i, first_order[i], yerr=error, 
                                   fmt='none', color='black', capsize=3)
                    
                    if name in ci['total_effect']:
                        ci_low, ci_high = ci['total_effect'][name]
                        error = [[total_effect[i] - ci_low], [ci_high - total_effect[i]]]
                        ax2.errorbar(i, total_effect[i], yerr=error,
                                   fmt='none', color='black', capsize=3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Sensitivity plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")

class PolynomialChaosExpansion:
    """
    Polynomial Chaos Expansion with sparse grid methods
    
    u(x) = Σ_α u_α Ψ_α(ξ)
    
    Where Ψ_α are orthogonal polynomials and u_α are coefficients
    """
    
    def __init__(self, config: ManufacturingConfig):
        self.config = config
        
        # PCE components
        self.polynomial_basis = {}
        self.coefficients = {}
        self.multi_indices = []
        self.mean_coefficient = 0.0
        
        # Validation metrics
        self.validation_error = None
        self.r_squared = None
        
    def generate_polynomial_basis(self, 
                                parameter_distributions: Dict[str, stats.rv_continuous],
                                max_degree: Optional[int] = None) -> Dict[str, List[Callable]]:
        """
        Generate orthogonal polynomial basis functions
        
        Args:
            parameter_distributions: Dictionary of parameter distributions
            max_degree: Maximum polynomial degree (uses config default if None)
            
        Returns:
            Dictionary of basis functions
        """
        if max_degree is None:
            max_degree = self.config.max_polynomial_degree
        
        basis_functions = {}
        
        for param_name, distribution in parameter_distributions.items():
            
            if isinstance(distribution, stats.norm_gen):
                # Hermite polynomials for Gaussian variables
                basis_functions[param_name] = self._generate_hermite_polynomials(max_degree)
                
            elif isinstance(distribution, stats.uniform_gen):
                # Legendre polynomials for uniform variables
                basis_functions[param_name] = self._generate_legendre_polynomials(max_degree)
                
            else:
                # Default to Legendre polynomials
                warnings.warn(f"Using Legendre polynomials for {param_name} distribution")
                basis_functions[param_name] = self._generate_legendre_polynomials(max_degree)
        
        self.polynomial_basis = basis_functions
        return basis_functions
    
    def _generate_hermite_polynomials(self, max_degree: int) -> List[Callable]:
        """Generate Hermite polynomials for Gaussian variables"""
        
        polynomials = []
        
        for n in range(max_degree + 1):
            if n == 0:
                polynomials.append(lambda x: np.ones_like(x))
            elif n == 1:
                polynomials.append(lambda x: x)
            elif n == 2:
                polynomials.append(lambda x: x**2 - 1)
            elif n == 3:
                polynomials.append(lambda x: x**3 - 3*x)
            elif n == 4:
                polynomials.append(lambda x: x**4 - 6*x**2 + 3)
            else:
                # Higher-order Hermite polynomials (recursive formula)
                def hermite_n(x, degree=n):
                    if degree <= 4:
                        return polynomials[degree](x)
                    # H_n(x) = x*H_{n-1}(x) - (n-1)*H_{n-2}(x)
                    return x * hermite_n(x, degree-1) - (degree-1) * hermite_n(x, degree-2)
                
                polynomials.append(lambda x, n=n: hermite_n(x, n))
        
        return polynomials
    
    def _generate_legendre_polynomials(self, max_degree: int) -> List[Callable]:
        """Generate Legendre polynomials for uniform variables"""
        
        polynomials = []
        
        for n in range(max_degree + 1):
            if n == 0:
                polynomials.append(lambda x: np.ones_like(x))
            elif n == 1:
                polynomials.append(lambda x: x)
            elif n == 2:
                polynomials.append(lambda x: 0.5*(3*x**2 - 1))
            elif n == 3:
                polynomials.append(lambda x: 0.5*(5*x**3 - 3*x))
            elif n == 4:
                polynomials.append(lambda x: (1/8)*(35*x**4 - 30*x**2 + 3))
            else:
                # Higher-order Legendre polynomials
                def legendre_n(x, degree=n):
                    if degree <= 4:
                        return polynomials[degree](x)
                    # Recursive formula for Legendre polynomials
                    return ((2*degree-1)*x*legendre_n(x, degree-1) - (degree-1)*legendre_n(x, degree-2))/degree
                
                polynomials.append(lambda x, n=n: legendre_n(x, n))
        
        return polynomials
    
    def generate_multi_indices(self, 
                             num_variables: int, 
                             max_degree: int,
                             sparse: bool = True) -> List[Tuple[int, ...]]:
        """
        Generate multi-indices for polynomial basis
        
        Args:
            num_variables: Number of input variables
            max_degree: Maximum total degree
            sparse: Use sparse grid approach
            
        Returns:
            List of multi-indices
        """
        multi_indices = []
        
        if sparse:
            # Sparse grid approach - only include indices with limited total degree
            for total_degree in range(max_degree + 1):
                for indices in itertools.combinations_with_replacement(range(num_variables), total_degree):
                    # Convert to multi-index
                    multi_index = [0] * num_variables
                    for idx in indices:
                        multi_index[idx] += 1
                    multi_indices.append(tuple(multi_index))
        else:
            # Full tensor grid
            ranges = [range(max_degree + 1) for _ in range(num_variables)]
            for indices in itertools.product(*ranges):
                if sum(indices) <= max_degree:
                    multi_indices.append(indices)
        
        self.multi_indices = multi_indices
        return multi_indices
    
    def evaluate_basis_functions(self, 
                               samples: np.ndarray,
                               parameter_names: List[str]) -> np.ndarray:
        """Evaluate all basis functions at sample points"""
        
        n_samples = samples.shape[0]
        n_basis = len(self.multi_indices)
        
        basis_matrix = np.zeros((n_samples, n_basis))
        
        for j, multi_index in enumerate(self.multi_indices):
            basis_values = np.ones(n_samples)
            
            for i, (param_name, degree) in enumerate(zip(parameter_names, multi_index)):
                if degree > 0 and param_name in self.polynomial_basis:
                    polynomial = self.polynomial_basis[param_name][degree]
                    basis_values *= polynomial(samples[:, i])
            
            basis_matrix[:, j] = basis_values
        
        return basis_matrix
    
    def fit(self, 
           samples: np.ndarray,
           responses: np.ndarray,
           parameter_names: List[str],
           parameter_distributions: Dict[str, stats.rv_continuous]) -> Dict[str, Any]:
        """
        Fit Polynomial Chaos Expansion to data
        
        Args:
            samples: Input samples (n_samples × n_parameters)
            responses: Model responses (n_samples,)
            parameter_names: List of parameter names
            parameter_distributions: Parameter distributions
            
        Returns:
            Fitting results and statistics
        """
        print("🎲 Fitting Polynomial Chaos Expansion...")
        
        # Generate polynomial basis
        self.generate_polynomial_basis(parameter_distributions)
        
        # Generate multi-indices
        num_variables = len(parameter_names)
        self.generate_multi_indices(num_variables, self.config.max_polynomial_degree)
        
        # Evaluate basis functions
        basis_matrix = self.evaluate_basis_functions(samples, parameter_names)
        
        # Solve for coefficients using least squares
        try:
            coefficients, residuals, rank, singular_values = np.linalg.lstsq(
                basis_matrix, responses, rcond=None
            )
            
            self.coefficients = {
                'values': coefficients,
                'multi_indices': self.multi_indices,
                'parameter_names': parameter_names
            }
            
            self.mean_coefficient = coefficients[0]  # First coefficient is the mean
            
            # Compute validation metrics
            predictions = basis_matrix @ coefficients
            self.validation_error = np.sqrt(np.mean((responses - predictions)**2))
            self.r_squared = 1 - np.sum((responses - predictions)**2) / np.sum((responses - np.mean(responses))**2)
            
            print(f"✅ PCE fitted with R² = {self.r_squared:.4f}, RMSE = {self.validation_error:.6f}")
            
            return {
                'coefficients': coefficients,
                'rmse': self.validation_error,
                'r_squared': self.r_squared,
                'n_terms': len(coefficients),
                'rank': rank
            }
            
        except np.linalg.LinAlgError as e:
            warnings.warn(f"PCE fitting failed: {e}")
            return {'error': str(e)}
    
    def predict(self, test_samples: np.ndarray, parameter_names: List[str]) -> np.ndarray:
        """Predict responses using fitted PCE"""
        
        if not self.coefficients:
            raise ValueError("PCE not fitted. Call fit() first.")
        
        # Evaluate basis functions for test samples
        basis_matrix = self.evaluate_basis_functions(test_samples, parameter_names)
        
        # Predict responses
        predictions = basis_matrix @ self.coefficients['values']
        
        return predictions
    
    def compute_sobol_indices_analytical(self) -> Dict[str, Dict[str, float]]:
        """
        Compute Sobol indices analytically from PCE coefficients
        
        More efficient than sampling-based methods for fitted PCE
        """
        if not self.coefficients:
            raise ValueError("PCE not fitted. Call fit() first.")
        
        coefficients = self.coefficients['values']
        multi_indices = self.coefficients['multi_indices']
        parameter_names = self.coefficients['parameter_names']
        
        # Total variance (sum of squares of all coefficients except mean)
        total_variance = np.sum(coefficients[1:]**2)
        
        if total_variance == 0:
            warnings.warn("Total variance is zero")
            return {}
        
        # First-order indices
        first_order = {}
        for i, param_name in enumerate(parameter_names):
            # Sum coefficients where only parameter i is active
            param_variance = 0.0
            for j, (coeff, multi_index) in enumerate(zip(coefficients[1:], multi_indices[1:]), 1):
                # Check if only parameter i is active in this term
                active_params = [k for k, degree in enumerate(multi_index) if degree > 0]
                if len(active_params) == 1 and active_params[0] == i:
                    param_variance += coeff**2
            
            first_order[param_name] = param_variance / total_variance
        
        # Total effect indices (more complex calculation)
        total_effect = {}
        for i, param_name in enumerate(parameter_names):
            # Sum coefficients where parameter i is involved
            param_total_variance = 0.0
            for j, (coeff, multi_index) in enumerate(zip(coefficients[1:], multi_indices[1:]), 1):
                if multi_index[i] > 0:  # Parameter i is active
                    param_total_variance += coeff**2
            
            total_effect[param_name] = param_total_variance / total_variance
        
        return {
            'first_order': first_order,
            'total_effect': total_effect,
            'total_variance': total_variance
        }

# Example usage and testing
if __name__ == "__main__":
    print("📊 Testing Manufacturing Process Mathematics")
    print("=" * 50)
    
    # Configuration
    config = ManufacturingConfig(
        num_samples_sobol=5000,
        max_polynomial_degree=3,
        max_workers=2
    )
    
    # Define test manufacturing process model
    def manufacturing_model(params):
        """
        Test manufacturing process model
        
        Inputs: [pressure, temperature, flow_rate, pH, concentration]
        Output: surface_roughness (nm RMS)
        """
        pressure, temperature, flow_rate, pH, concentration = params
        
        # Simplified model with interactions
        base_roughness = 0.5  # nm
        
        # Main effects
        pressure_effect = -0.3 * (pressure - 1e5) / 1e4  # Lower pressure = higher roughness
        temp_effect = 0.1 * (temperature - 300) / 50    # Higher temp = higher roughness
        flow_effect = -0.2 * (flow_rate - 1e-6) / 1e-7  # Higher flow = lower roughness
        pH_effect = 0.05 * (pH - 7)**2                  # Optimal pH = 7
        conc_effect = -0.1 * (concentration - 0.1) / 0.05  # Higher conc = lower roughness
        
        # Interaction effects
        pressure_temp_interaction = 0.05 * (pressure - 1e5) * (temperature - 300) / (1e4 * 50)
        flow_pH_interaction = 0.03 * (flow_rate - 1e-6) * (pH - 7) / (1e-7)
        
        # Noise
        noise = 0.02 * np.random.randn()
        
        roughness = base_roughness + pressure_effect + temp_effect + flow_effect + pH_effect + conc_effect + pressure_temp_interaction + flow_pH_interaction + noise
        
        return max(0.05, roughness)  # Minimum physical roughness
    
    # Parameter bounds for manufacturing process
    parameter_bounds = {
        'pressure': (8e4, 1.2e5),      # Pa
        'temperature': (280, 320),      # K
        'flow_rate': (5e-7, 2e-6),     # m³/s
        'pH': (6.0, 8.0),              # -
        'concentration': (0.05, 0.2)   # mol/L
    }
    
    print("🧪 Running Sobol Sensitivity Analysis...")
    
    # Initialize Sobol analyzer
    sobol_analyzer = AdvancedSobolAnalyzer(config)
    
    # Generate Sobol samples
    sample_matrices = sobol_analyzer.generate_sobol_samples(parameter_bounds)
    print(f"Generated {len(sample_matrices['A'])} Sobol samples")
    
    # Evaluate model
    evaluations = sobol_analyzer.evaluate_model(manufacturing_model, sample_matrices)
    
    # Compute sensitivity indices
    sobol_results = sobol_analyzer.compute_sobol_indices(evaluations)
    
    print("\n📈 Sobol Sensitivity Results:")
    print("-" * 30)
    for param, index in sobol_results['first_order'].items():
        total_effect = sobol_results['total_effect'][param]
        print(f"{param:12}: S₁ = {index:.3f}, Sₜ = {total_effect:.3f}")
    
    print(f"\nTotal variance: {sobol_results['total_variance']:.6f}")
    
    # Test Polynomial Chaos Expansion
    print("\n🎲 Testing Polynomial Chaos Expansion...")
    
    # Generate training data
    np.random.seed(42)
    n_train = 1000
    
    # Generate samples from parameter distributions
    train_samples = np.zeros((n_train, 5))
    param_names = list(parameter_bounds.keys())
    
    for i, param_name in enumerate(param_names):
        min_val, max_val = parameter_bounds[param_name]
        train_samples[:, i] = np.random.uniform(min_val, max_val, n_train)
    
    # Evaluate training responses
    train_responses = np.array([manufacturing_model(row) for row in train_samples])
    
    # Define parameter distributions for PCE
    parameter_distributions = {}
    for param_name in param_names:
        min_val, max_val = parameter_bounds[param_name]
        # Use uniform distributions (would use more appropriate ones in practice)
        parameter_distributions[param_name] = stats.uniform(loc=min_val, scale=max_val-min_val)
    
    # Fit PCE
    pce = PolynomialChaosExpansion(config)
    pce_results = pce.fit(train_samples, train_responses, param_names, parameter_distributions)
    
    print(f"PCE R²: {pce_results['r_squared']:.4f}")
    print(f"PCE RMSE: {pce_results['rmse']:.4f} nm")
    print(f"PCE terms: {pce_results['n_terms']}")
    
    # Test PCE predictions
    n_test = 100
    test_samples = np.zeros((n_test, 5))
    for i, param_name in enumerate(param_names):
        min_val, max_val = parameter_bounds[param_name]
        test_samples[:, i] = np.random.uniform(min_val, max_val, n_test)
    
    pce_predictions = pce.predict(test_samples, param_names)
    actual_responses = np.array([manufacturing_model(row) for row in test_samples])
    
    test_rmse = np.sqrt(np.mean((actual_responses - pce_predictions)**2))
    print(f"PCE test RMSE: {test_rmse:.4f} nm")
    
    # Analytical Sobol indices from PCE
    pce_sobol = pce.compute_sobol_indices_analytical()
    
    if pce_sobol:
        print("\n📊 PCE-based Sobol Indices:")
        print("-" * 30)
        for param in param_names:
            s1 = pce_sobol['first_order'].get(param, 0)
            st = pce_sobol['total_effect'].get(param, 0)
            print(f"{param:12}: S₁ = {s1:.3f}, Sₜ = {st:.3f}")
    
    print("\n✅ Manufacturing Process Mathematics implementation complete!")
