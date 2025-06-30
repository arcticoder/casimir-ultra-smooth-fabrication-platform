"""
Advanced Uncertainty Propagation for Digital Twin
================================================

Enhanced uncertainty quantification including:
- Monte Carlo with Auto-Convergence (Gelman-Rubin)
- Advanced Sobol Sensitivity Analysis
- Polynomial Chaos Expansion
- Comprehensive Numerical Stability Framework

Mathematical formulations:
R̂ = √((N-1)/N + (1/N)(B/W))
Converged if R̂ < 1.1

Sᵢ = Var[E[Y|Xᵢ]]/Var[Y]
Sᵢ,ⱼ = (Var[E[Y|Xᵢ,Xⱼ]] - Var[E[Y|Xᵢ]] - Var[E[Y|Xⱼ]])/Var[Y]
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.special import eval_hermite
import itertools

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty propagation"""
    num_samples: int = 50000
    max_samples: int = 100000
    convergence_threshold: float = 1.1
    confidence_level: float = 0.95
    
    # Sobol analysis
    num_sobol_samples: int = 10000
    max_order: int = 2  # Maximum interaction order
    
    # PCE parameters
    pce_order: int = 3
    pce_basis: str = "hermite"  # hermite, legendre, laguerre
    
    # Numerical stability
    condition_threshold: float = 1e12
    valid_fraction_threshold: float = 0.9
    overflow_threshold: float = 1e100

class MonteCarloWithConvergence:
    """
    Enhanced Monte Carlo with Auto-Convergence using Gelman-Rubin diagnostic
    
    R̂ = √((N-1)/N + (1/N)(B/W))
    Converged if R̂ < 1.1
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize Monte Carlo sampler"""
        self.config = config
        self.samples = []
        self.converged = False
        
    def gelman_rubin_diagnostic(self, chains: List[np.ndarray]) -> float:
        """
        Compute Gelman-Rubin convergence diagnostic
        
        R̂ = √((N-1)/N + (1/N)(B/W))
        
        Args:
            chains: List of MCMC chains
            
        Returns:
            R̂ convergence statistic
        """
        num_chains = len(chains)
        if num_chains < 2:
            return float('inf')  # Need at least 2 chains
        
        chain_length = min(len(chain) for chain in chains)
        if chain_length < 2:
            return float('inf')
        
        # Truncate chains to same length
        chains = [chain[:chain_length] for chain in chains]
        
        # Convert to array: (num_chains, chain_length, num_variables)
        chains_array = np.array(chains)
        
        # Handle multi-dimensional outputs
        if chains_array.ndim == 2:
            chains_array = chains_array[:, :, np.newaxis]
        
        num_variables = chains_array.shape[2]
        R_hat_values = []
        
        for var_idx in range(num_variables):
            var_chains = chains_array[:, :, var_idx]
            
            # Chain means
            chain_means = np.mean(var_chains, axis=1)
            overall_mean = np.mean(chain_means)
            
            # Between-chain variance B
            B = chain_length * np.var(chain_means, ddof=1)
            
            # Within-chain variance W
            chain_variances = np.var(var_chains, axis=1, ddof=1)
            W = np.mean(chain_variances)
            
            # Avoid division by zero
            if W == 0:
                R_hat_values.append(1.0)
                continue
            
            # Marginal posterior variance estimate
            var_plus = ((chain_length - 1) / chain_length) * W + (1 / chain_length) * B
            
            # R̂ statistic
            R_hat = np.sqrt(var_plus / W) if W > 0 else float('inf')
            R_hat_values.append(R_hat)
        
        return max(R_hat_values)  # Return worst-case R̂
    
    def adaptive_sampling(self,
                         model_func: Callable[[np.ndarray], np.ndarray],
                         parameter_distributions: Dict[str, stats.rv_continuous],
                         initial_samples: int = 1000) -> Dict[str, Any]:
        """
        Adaptive Monte Carlo sampling with convergence monitoring
        
        Args:
            model_func: Function to evaluate Y = f(X)
            parameter_distributions: Dictionary of parameter distributions
            initial_samples: Initial number of samples per chain
            
        Returns:
            Dictionary with uncertainty propagation results
        """
        param_names = list(parameter_distributions.keys())
        num_params = len(param_names)
        num_chains = 4  # Multiple chains for convergence diagnostics
        
        # Initialize chains
        chains_samples = []
        chains_outputs = []
        
        for chain_idx in range(num_chains):
            chain_samples = []
            chain_outputs = []
            chains_samples.append(chain_samples)
            chains_outputs.append(chain_outputs)
        
        converged = False
        total_samples = 0
        R_hat_history = []
        
        # Iterative sampling
        while not converged and total_samples < self.config.max_samples:
            batch_size = min(initial_samples, self.config.max_samples - total_samples)
            
            # Sample each chain
            for chain_idx in range(num_chains):
                # Generate samples for this chain
                samples_batch = np.zeros((batch_size, num_params))
                
                for i, param_name in enumerate(param_names):
                    dist = parameter_distributions[param_name]
                    samples_batch[:, i] = dist.rvs(size=batch_size, random_state=chain_idx*1000 + total_samples)
                
                # Evaluate model
                outputs_batch = []
                valid_count = 0
                
                for sample in samples_batch:
                    try:
                        output = model_func(sample)
                        if np.all(np.isfinite(output)) and np.all(np.abs(output) < self.config.overflow_threshold):
                            outputs_batch.append(output)
                            valid_count += 1
                        else:
                            outputs_batch.append(np.full_like(output, np.nan))
                    except Exception as e:
                        warnings.warn(f"Model evaluation failed: {e}")
                        # Use previous valid output or zeros as fallback
                        if len(outputs_batch) > 0:
                            outputs_batch.append(outputs_batch[-1])
                        else:
                            outputs_batch.append(np.zeros(1))
                
                # Store results
                chains_samples[chain_idx].extend(samples_batch)
                chains_outputs[chain_idx].extend(outputs_batch)
            
            total_samples += batch_size
            
            # Check convergence if we have enough samples
            if total_samples >= 2000:  # Minimum samples for meaningful convergence check
                # Convert outputs to arrays for convergence checking
                output_arrays = []
                for chain_outputs in chains_outputs:
                    if len(chain_outputs) > 0:
                        chain_array = np.array(chain_outputs)
                        # Handle NaN values
                        valid_mask = np.all(np.isfinite(chain_array), axis=1)
                        if np.sum(valid_mask) > 0:
                            output_arrays.append(chain_array[valid_mask])
                
                if len(output_arrays) >= 2:
                    R_hat = self.gelman_rubin_diagnostic(output_arrays)
                    R_hat_history.append(R_hat)
                    
                    print(f"Samples: {total_samples}, R̂: {R_hat:.4f}")
                    
                    if R_hat < self.config.convergence_threshold:
                        converged = True
                        print(f"✅ Converged at {total_samples} samples (R̂ = {R_hat:.4f})")
        
        # Combine all chains
        all_samples = []
        all_outputs = []
        
        for chain_idx in range(num_chains):
            all_samples.extend(chains_samples[chain_idx])
            all_outputs.extend(chains_outputs[chain_idx])
        
        # Convert to arrays
        samples_array = np.array(all_samples)
        outputs_array = np.array(all_outputs)
        
        # Remove invalid outputs
        valid_mask = np.all(np.isfinite(outputs_array), axis=1)
        valid_fraction = np.sum(valid_mask) / len(valid_mask)
        
        if valid_fraction < self.config.valid_fraction_threshold:
            warnings.warn(f"Only {valid_fraction:.1%} of samples are valid")
        
        samples_valid = samples_array[valid_mask]
        outputs_valid = outputs_array[valid_mask]
        
        # Compute statistics
        statistics = self._compute_statistics(outputs_valid)
        
        return {
            'samples': samples_valid,
            'outputs': outputs_valid,
            'statistics': statistics,
            'convergence': {
                'converged': converged,
                'total_samples': total_samples,
                'R_hat_final': R_hat_history[-1] if R_hat_history else float('inf'),
                'R_hat_history': R_hat_history,
                'valid_fraction': valid_fraction
            },
            'parameter_names': param_names
        }
    
    def _compute_statistics(self, outputs: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive statistics"""
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]
        
        num_outputs = outputs.shape[1]
        stats_dict = {}
        
        for i in range(num_outputs):
            output_col = outputs[:, i]
            
            stats_dict[f'output_{i}'] = {
                'mean': np.mean(output_col),
                'std': np.std(output_col),
                'var': np.var(output_col),
                'min': np.min(output_col),
                'max': np.max(output_col),
                'median': np.median(output_col),
                'percentiles': {
                    '5%': np.percentile(output_col, 5),
                    '25%': np.percentile(output_col, 25),
                    '75%': np.percentile(output_col, 75),
                    '95%': np.percentile(output_col, 95)
                }
            }
        
        return stats_dict

class SobolSensitivityAnalysis:
    """
    Advanced Sobol Sensitivity Analysis
    
    Sᵢ = Var[E[Y|Xᵢ]]/Var[Y]
    Sᵢ,ⱼ = (Var[E[Y|Xᵢ,Xⱼ]] - Var[E[Y|Xᵢ]] - Var[E[Y|Xⱼ]])/Var[Y]
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize Sobol analyzer"""
        self.config = config
    
    def generate_sobol_samples(self, num_params: int, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Sobol sample matrices A and B
        
        Returns:
            A, B sample matrices for Sobol analysis
        """
        # Use Sobol sequence if available, otherwise use random
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=num_params, scramble=True)
            A = sampler.random(num_samples)
            B = sampler.random(num_samples)
        except ImportError:
            # Fallback to random sampling
            A = np.random.random((num_samples, num_params))
            B = np.random.random((num_samples, num_params))
        
        return A, B
    
    def compute_sobol_indices(self,
                            model_func: Callable[[np.ndarray], np.ndarray],
                            parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Compute first and second-order Sobol sensitivity indices
        
        Args:
            model_func: Model function Y = f(X)
            parameter_bounds: Dictionary of parameter bounds {name: (min, max)}
            
        Returns:
            Dictionary with Sobol indices
        """
        param_names = list(parameter_bounds.keys())
        num_params = len(param_names)
        num_samples = self.config.num_sobol_samples
        
        # Generate Sobol samples
        A, B = self.generate_sobol_samples(num_params, num_samples)
        
        # Scale samples to parameter bounds
        for i, param_name in enumerate(param_names):
            min_val, max_val = parameter_bounds[param_name]
            A[:, i] = min_val + A[:, i] * (max_val - min_val)
            B[:, i] = min_val + B[:, i] * (max_val - min_val)
        
        # Evaluate model at sample points
        Y_A = self._evaluate_model_batch(model_func, A)
        Y_B = self._evaluate_model_batch(model_func, B)
        
        # Compute variance estimates
        Y_all = np.concatenate([Y_A, Y_B])
        total_variance = np.var(Y_all, ddof=1)
        
        if total_variance == 0:
            warnings.warn("Total variance is zero - cannot compute Sobol indices")
            return {'error': 'Zero variance'}
        
        # First-order indices
        first_order = {}
        
        for i, param_name in enumerate(param_names):
            # Create matrix C_i where column i is from B, others from A
            C_i = A.copy()
            C_i[:, i] = B[:, i]
            
            Y_C_i = self._evaluate_model_batch(model_func, C_i)
            
            # First-order index: Sᵢ = Var[E[Y|Xᵢ]]/Var[Y]
            # Approximated as: (1/N) * Σ(Y_A * Y_C_i) - f₀²
            f0_squared = np.mean(Y_A) * np.mean(Y_B)
            
            numerator = np.mean(Y_A * Y_C_i) - f0_squared
            S_i = numerator / total_variance
            
            first_order[param_name] = max(0.0, S_i)  # Ensure non-negative
        
        # Second-order indices (if requested)
        second_order = {}
        
        if self.config.max_order >= 2:
            for i, param_i in enumerate(param_names):
                for j, param_j in enumerate(param_names):
                    if i < j:  # Avoid duplicates
                        # Create matrix C_ij where columns i,j are from B, others from A
                        C_ij = A.copy()
                        C_ij[:, i] = B[:, i]
                        C_ij[:, j] = B[:, j]
                        
                        Y_C_ij = self._evaluate_model_batch(model_func, C_ij)
                        
                        # Second-order index: Sᵢⱼ = (Var[E[Y|Xᵢ,Xⱼ]] - Sᵢ - Sⱼ)/Var[Y]
                        numerator_ij = np.mean(Y_A * Y_C_ij) - f0_squared
                        S_ij_total = numerator_ij / total_variance
                        S_ij = S_ij_total - first_order[param_i] - first_order[param_j]
                        
                        second_order[f'{param_i}_{param_j}'] = max(0.0, S_ij)
        
        # Total-order indices
        total_order = {}
        
        for i, param_name in enumerate(param_names):
            # Create matrix C_~i where column i is from A, others from B
            C_not_i = B.copy()
            C_not_i[:, i] = A[:, i]
            
            Y_C_not_i = self._evaluate_model_batch(model_func, C_not_i)
            
            # Total-order index: STᵢ = 1 - Var[E[Y|X~ᵢ]]/Var[Y]
            numerator_total = np.mean(Y_B * Y_C_not_i) - f0_squared
            ST_i = 1.0 - (numerator_total / total_variance)
            
            total_order[param_name] = max(0.0, min(1.0, ST_i))  # Bound between 0 and 1
        
        return {
            'first_order': first_order,
            'second_order': second_order,
            'total_order': total_order,
            'total_variance': total_variance,
            'parameter_names': param_names,
            'num_samples': num_samples * 2 * (num_params + 1)  # Total evaluations
        }
    
    def _evaluate_model_batch(self, model_func: Callable, samples: np.ndarray) -> np.ndarray:
        """Evaluate model function on batch of samples"""
        outputs = []
        
        for sample in samples:
            try:
                output = model_func(sample)
                if np.isscalar(output):
                    outputs.append(output)
                else:
                    outputs.append(output[0])  # Take first output if vector
            except Exception as e:
                warnings.warn(f"Model evaluation failed: {e}")
                outputs.append(0.0)  # Fallback
        
        return np.array(outputs)

class PolynomialChaosExpansion:
    """
    Polynomial Chaos Expansion Implementation
    
    Y = Σ_{|α|≤p} c_α Ψ_α(ξ)
    Ψ_α(ξ) = Π_{i=1}^d H_{α_i}(ξ_i)
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize PCE"""
        self.config = config
        self.coefficients = None
        self.multi_indices = None
        
    def generate_multi_indices(self, num_vars: int, max_order: int) -> List[Tuple]:
        """Generate multi-indices for PCE basis"""
        indices = []
        
        for total_order in range(max_order + 1):
            for alpha in itertools.combinations_with_replacement(range(num_vars), total_order):
                # Convert to multi-index format
                multi_index = [0] * num_vars
                for var_idx in alpha:
                    multi_index[var_idx] += 1
                indices.append(tuple(multi_index))
        
        return indices
    
    def hermite_polynomial(self, x: np.ndarray, order: int) -> np.ndarray:
        """Evaluate Hermite polynomial of given order"""
        if order == 0:
            return np.ones_like(x)
        elif order == 1:
            return x
        else:
            return eval_hermite(order, x)
    
    def evaluate_basis(self, xi: np.ndarray, multi_index: Tuple) -> np.ndarray:
        """
        Evaluate multivariate polynomial basis
        Ψ_α(ξ) = Π_{i=1}^d H_{α_i}(ξ_i)
        """
        result = np.ones(xi.shape[0])
        
        for i, order in enumerate(multi_index):
            if order > 0:
                result *= self.hermite_polynomial(xi[:, i], order)
        
        return result
    
    def fit(self,
            samples: np.ndarray,
            outputs: np.ndarray) -> Dict[str, Any]:
        """
        Fit PCE using least squares
        
        Args:
            samples: Input samples (transformed to standard normal)
            outputs: Model outputs
            
        Returns:
            PCE fitting results
        """
        num_vars = samples.shape[1]
        
        # Generate multi-indices
        self.multi_indices = self.generate_multi_indices(num_vars, self.config.pce_order)
        num_terms = len(self.multi_indices)
        
        # Build basis matrix
        basis_matrix = np.zeros((len(samples), num_terms))
        
        for j, multi_index in enumerate(self.multi_indices):
            basis_matrix[:, j] = self.evaluate_basis(samples, multi_index)
        
        # Solve least squares problem
        try:
            self.coefficients, residuals, rank, s = np.linalg.lstsq(basis_matrix, outputs, rcond=None)
        except np.linalg.LinAlgError:
            warnings.warn("PCE fitting failed - using fallback")
            self.coefficients = np.zeros(num_terms)
            self.coefficients[0] = np.mean(outputs)  # Mean term only
        
        # Compute fit quality metrics
        predictions = basis_matrix @ self.coefficients
        r_squared = 1.0 - np.var(outputs - predictions) / np.var(outputs)
        
        return {
            'coefficients': self.coefficients,
            'multi_indices': self.multi_indices,
            'r_squared': r_squared,
            'num_terms': num_terms,
            'condition_number': np.linalg.cond(basis_matrix)
        }
    
    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Predict using fitted PCE"""
        if self.coefficients is None:
            raise ValueError("PCE not fitted yet")
        
        predictions = np.zeros(len(samples))
        
        for j, multi_index in enumerate(self.multi_indices):
            basis_values = self.evaluate_basis(samples, multi_index)
            predictions += self.coefficients[j] * basis_values
        
        return predictions
    
    def sensitivity_indices(self) -> Dict[str, float]:
        """Compute Sobol indices from PCE coefficients"""
        if self.coefficients is None:
            raise ValueError("PCE not fitted yet")
        
        # Total variance (excluding mean term)
        total_var = np.sum(self.coefficients[1:] ** 2)
        
        if total_var == 0:
            return {}
        
        # First-order indices
        num_vars = len(self.multi_indices[0])
        first_order = {}
        
        for i in range(num_vars):
            var_i = 0.0
            for j, multi_index in enumerate(self.multi_indices[1:], 1):
                if sum(multi_index) == multi_index[i] and multi_index[i] > 0:
                    var_i += self.coefficients[j] ** 2
            
            first_order[f'param_{i}'] = var_i / total_var
        
        return first_order

class NumericalStabilityFramework:
    """
    Comprehensive Numerical Stability Framework
    
    condition = max(|outputs|)/min(|outputs|) < 10¹²
    valid_fraction = |{finite outputs}|/|{all outputs}| > 0.9
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize stability framework"""
        self.config = config
        
    def check_stability(self, outputs: np.ndarray) -> Dict[str, Any]:
        """Comprehensive stability check"""
        results = {}
        
        # Basic checks
        finite_mask = np.isfinite(outputs)
        valid_fraction = np.mean(finite_mask)
        
        results['valid_fraction'] = valid_fraction
        results['meets_valid_threshold'] = valid_fraction >= self.config.valid_fraction_threshold
        
        if np.sum(finite_mask) == 0:
            results['condition_number'] = float('inf')
            results['meets_condition_threshold'] = False
            results['overflow_detected'] = True
            results['recommendations'] = ['All outputs are invalid', 'Check model implementation']
            return results
        
        valid_outputs = outputs[finite_mask]
        
        # Condition number check
        if len(valid_outputs) > 0:
            abs_outputs = np.abs(valid_outputs)
            abs_outputs = abs_outputs[abs_outputs > 0]  # Remove zeros
            
            if len(abs_outputs) > 0:
                condition_number = np.max(abs_outputs) / np.min(abs_outputs)
                results['condition_number'] = condition_number
                results['meets_condition_threshold'] = condition_number < self.config.condition_threshold
            else:
                results['condition_number'] = 1.0
                results['meets_condition_threshold'] = True
        
        # Overflow detection
        overflow_mask = np.abs(outputs) >= self.config.overflow_threshold
        results['overflow_detected'] = np.any(overflow_mask)
        results['overflow_fraction'] = np.mean(overflow_mask)
        
        # Generate recommendations
        recommendations = []
        
        if not results['meets_valid_threshold']:
            recommendations.append(f"Low valid fraction ({valid_fraction:.1%}). Check input bounds.")
        
        if not results.get('meets_condition_threshold', True):
            recommendations.append(f"High condition number ({results.get('condition_number', 0):.1e}). Consider scaling.")
        
        if results['overflow_detected']:
            recommendations.append(f"Overflow detected ({results['overflow_fraction']:.1%} of samples).")
        
        if not recommendations:
            recommendations.append("Numerical stability looks good.")
        
        results['recommendations'] = recommendations
        results['overall_stable'] = (results['meets_valid_threshold'] and 
                                   results.get('meets_condition_threshold', True) and 
                                   not results['overflow_detected'])
        
        return results

# Example usage and testing
if __name__ == "__main__":
    print("🧮 Testing Advanced Uncertainty Propagation")
    print("=" * 50)
    
    # Configuration
    config = UncertaintyConfig(
        num_samples=5000,
        max_samples=20000,
        convergence_threshold=1.1
    )
    
    # Test model function (Casimir-like)
    def test_model(params):
        """Test model: F = A * params[0]^(-4) * params[1] * exp(-params[2])"""
        d, A, T = params[:3]
        if d <= 0:
            return np.array([float('inf')])
        return np.array([A * d**(-4) * np.exp(-T/300)])
    
    # Parameter distributions
    parameter_dists = {
        'separation': stats.uniform(loc=50e-9, scale=950e-9),  # 50-1000 nm
        'area': stats.uniform(loc=50e-12, scale=150e-12),      # 50-200 μm²
        'temperature': stats.norm(loc=300, scale=10)            # 300±10 K
    }
    
    # 1. Test Monte Carlo with convergence
    print("1️⃣ Testing Monte Carlo with Auto-Convergence...")
    mc = MonteCarloWithConvergence(config)
    mc_results = mc.adaptive_sampling(test_model, parameter_dists)
    
    print(f"  Converged: {mc_results['convergence']['converged']}")
    print(f"  Total samples: {mc_results['convergence']['total_samples']}")
    print(f"  Final R̂: {mc_results['convergence']['R_hat_final']:.4f}")
    print(f"  Valid fraction: {mc_results['convergence']['valid_fraction']:.2%}")
    
    # 2. Test Sobol sensitivity analysis
    print("\n2️⃣ Testing Sobol Sensitivity Analysis...")
    sobol = SobolSensitivityAnalysis(config)
    
    parameter_bounds = {
        'separation': (50e-9, 1000e-9),
        'area': (50e-12, 200e-12),
        'temperature': (280, 320)
    }
    
    sobol_results = sobol.compute_sobol_indices(test_model, parameter_bounds)
    
    print("  First-order indices:")
    for param, index in sobol_results['first_order'].items():
        print(f"    {param}: {index:.4f}")
    
    print("  Total-order indices:")
    for param, index in sobol_results['total_order'].items():
        print(f"    {param}: {index:.4f}")
    
    # 3. Test Polynomial Chaos Expansion
    print("\n3️⃣ Testing Polynomial Chaos Expansion...")
    pce = PolynomialChaosExpansion(config)
    
    # Use samples from Monte Carlo
    samples_norm = stats.norm.ppf(stats.uniform.cdf(mc_results['samples']))  # Transform to standard normal
    pce_fit = pce.fit(samples_norm, mc_results['outputs'].flatten())
    
    print(f"  R²: {pce_fit['r_squared']:.4f}")
    print(f"  Number of terms: {pce_fit['num_terms']}")
    print(f"  Condition number: {pce_fit['condition_number']:.2e}")
    
    # 4. Test numerical stability
    print("\n4️⃣ Testing Numerical Stability Framework...")
    stability = NumericalStabilityFramework(config)
    stability_results = stability.check_stability(mc_results['outputs'].flatten())
    
    print(f"  Overall stable: {stability_results['overall_stable']}")
    print(f"  Valid fraction: {stability_results['valid_fraction']:.2%}")
    print(f"  Condition number: {stability_results['condition_number']:.2e}")
    print("  Recommendations:")
    for rec in stability_results['recommendations']:
        print(f"    - {rec}")
    
    print("\n✅ Advanced Uncertainty Propagation implementation complete!")
