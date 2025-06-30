"""
Numerical Stability & Validation Framework
=========================================

Advanced numerical methods and validation for digital twin:
- Adaptive Numerical Solvers with Error Control
- Numerical Stability Analysis and Conditioning
- Cross-Validation and Model Validation Framework
- Convergence Analysis and Performance Metrics

Mathematical formulations:
Adaptive step size: h_{n+1} = h_n * (tol/err)^{1/(p+1)}
Condition number: κ(A) = ||A|| ||A^{-1}||
Validation metric: CV(k) = (1/k) Σᵢ₌₁ᵏ L(y_i, ŷ_i)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
from scipy import linalg, integrate, optimize
from scipy.sparse import csc_matrix, issparse
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
from concurrent.futures import ThreadPoolExecutor

@dataclass
class NumericalConfig:
    """Configuration for numerical stability and validation"""
    
    # Adaptive solver parameters
    initial_step_size: float = 1e-6
    min_step_size: float = 1e-12
    max_step_size: float = 1e-3
    error_tolerance: float = 1e-8
    relative_tolerance: float = 1e-10
    max_iterations: int = 10000
    
    # Stability analysis
    condition_number_threshold: float = 1e12
    eigenvalue_threshold: float = 1e-15
    singular_value_threshold: float = 1e-12
    
    # Validation parameters
    cross_validation_folds: int = 5
    validation_split: float = 0.2
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Performance monitoring
    max_execution_time: float = 300.0  # seconds
    memory_limit: int = 1000  # MB
    convergence_window: int = 50
    
    # Parallel processing
    max_workers: int = 4
    enable_parallel: bool = True

class AdaptiveNumericalSolver:
    """
    Adaptive numerical solver with error control
    
    Implements Runge-Kutta-Fehlberg method with adaptive step size
    """
    
    def __init__(self, config: NumericalConfig):
        self.config = config
        
        # RKF45 coefficients
        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
        ])
        
        self.c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
        
        # 4th order solution coefficients
        self.b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        
        # 5th order solution coefficients
        self.b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        
        # Statistics
        self.step_history = []
        self.error_history = []
        self.rejected_steps = 0
        
    def solve_ivp_adaptive(self,
                          fun: Callable[[float, np.ndarray], np.ndarray],
                          t_span: Tuple[float, float],
                          y0: np.ndarray,
                          args: Tuple = ()) -> Dict[str, Any]:
        """
        Solve initial value problem with adaptive step size
        
        Args:
            fun: Right-hand side function dy/dt = f(t, y)
            t_span: Integration interval (t0, tf)
            y0: Initial conditions
            args: Additional arguments to fun
            
        Returns:
            Solution dictionary with time points, solution, and diagnostics
        """
        t0, tf = t_span
        t = t0
        y = y0.copy()
        h = self.config.initial_step_size
        
        # Solution storage
        t_values = [t0]
        y_values = [y0.copy()]
        
        # Reset statistics
        self.step_history = []
        self.error_history = []
        self.rejected_steps = 0
        
        iteration = 0
        
        while t < tf and iteration < self.config.max_iterations:
            
            if t + h > tf:
                h = tf - t
            
            # Compute RKF45 step
            try:
                y_new, y_error, h_new, step_accepted = self._rkf45_step(fun, t, y, h, args)
                
                if step_accepted:
                    t += h
                    y = y_new
                    
                    t_values.append(t)
                    y_values.append(y.copy())
                    
                    self.step_history.append(h)
                    self.error_history.append(np.linalg.norm(y_error))
                else:
                    self.rejected_steps += 1
                
                h = h_new
                
            except Exception as e:
                warnings.warn(f"Integration step failed: {e}")
                h *= 0.5
                if h < self.config.min_step_size:
                    break
            
            iteration += 1
        
        # Check convergence
        converged = t >= tf - 1e-10
        
        return {
            't': np.array(t_values),
            'y': np.array(y_values),
            'converged': converged,
            'n_steps': len(t_values) - 1,
            'rejected_steps': self.rejected_steps,
            'final_step_size': h,
            'step_history': self.step_history,
            'error_history': self.error_history
        }
    
    def _rkf45_step(self,
                   fun: Callable[[float, np.ndarray], np.ndarray],
                   t: float,
                   y: np.ndarray,
                   h: float,
                   args: Tuple) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Single RKF45 step with error estimation"""
        
        # Compute k values
        k = np.zeros((6, len(y)))
        
        k[0] = h * fun(t, y, *args)
        
        for i in range(1, 6):
            y_temp = y + np.sum(self.a[i, :i] * k[:i].T, axis=1)
            k[i] = h * fun(t + self.c[i] * h, y_temp, *args)
        
        # 4th order solution
        y4 = y + np.sum(self.b4 * k.T, axis=1)
        
        # 5th order solution
        y5 = y + np.sum(self.b5 * k.T, axis=1)
        
        # Error estimate
        error = y5 - y4
        error_norm = np.linalg.norm(error)
        
        # Compute tolerances
        scale = np.maximum(np.abs(y), np.abs(y5))
        tolerance = self.config.error_tolerance + self.config.relative_tolerance * scale
        local_error = error / tolerance
        local_error_norm = np.linalg.norm(local_error)
        
        # Step size adaptation
        if local_error_norm <= 1.0:
            # Accept step
            step_accepted = True
            
            # Compute new step size
            if local_error_norm > 0:
                h_new = h * min(2.0, 0.9 * (1.0 / local_error_norm) ** 0.2)
            else:
                h_new = h * 2.0
            
            h_new = min(h_new, self.config.max_step_size)
        else:
            # Reject step
            step_accepted = False
            y5 = y  # Don't update solution
            
            # Reduce step size
            h_new = h * max(0.1, 0.9 * (1.0 / local_error_norm) ** 0.25)
            h_new = max(h_new, self.config.min_step_size)
        
        return y5, error, h_new, step_accepted

class NumericalStabilityAnalyzer:
    """Analyze numerical stability and conditioning of problems"""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
        self.stability_metrics = {}
        
    def analyze_matrix_conditioning(self, 
                                  matrix: np.ndarray,
                                  matrix_name: str = "Matrix") -> Dict[str, Any]:
        """
        Analyze conditioning of a matrix
        
        Args:
            matrix: Matrix to analyze
            matrix_name: Name for reporting
            
        Returns:
            Dictionary of conditioning metrics
        """
        print(f"🔍 Analyzing conditioning of {matrix_name}")
        
        try:
            # Condition number
            if issparse(matrix):
                # Convert to dense for condition number calculation
                matrix_dense = matrix.toarray()
            else:
                matrix_dense = matrix
            
            # Various condition number calculations
            cond_2 = linalg.norm(matrix_dense, 2) * linalg.norm(linalg.pinv(matrix_dense), 2)
            cond_fro = linalg.norm(matrix_dense, 'fro') * linalg.norm(linalg.pinv(matrix_dense), 'fro')
            
            # Singular value decomposition
            try:
                U, s, Vt = linalg.svd(matrix_dense, full_matrices=False)
                
                singular_values = s
                min_sv = np.min(s)
                max_sv = np.max(s)
                cond_svd = max_sv / min_sv if min_sv > 0 else np.inf
                
                # Rank estimation
                rank_numerical = np.sum(s > self.config.singular_value_threshold)
                rank_theoretical = min(matrix_dense.shape)
                
            except Exception as e:
                warnings.warn(f"SVD analysis failed: {e}")
                singular_values = np.array([])
                cond_svd = np.inf
                rank_numerical = 0
                rank_theoretical = min(matrix_dense.shape)
            
            # Eigenvalue analysis (for square matrices)
            eigenvalues = np.array([])
            spectral_radius = 0.0
            
            if matrix_dense.shape[0] == matrix_dense.shape[1]:
                try:
                    eigenvalues = linalg.eigvals(matrix_dense)
                    spectral_radius = np.max(np.abs(eigenvalues))
                except Exception as e:
                    warnings.warn(f"Eigenvalue analysis failed: {e}")
            
            # Stability assessment
            well_conditioned = cond_2 < self.config.condition_number_threshold
            full_rank = rank_numerical == rank_theoretical
            stable = well_conditioned and full_rank
            
            results = {
                'matrix_name': matrix_name,
                'shape': matrix_dense.shape,
                'condition_number_2': cond_2,
                'condition_number_frobenius': cond_fro,
                'condition_number_svd': cond_svd,
                'singular_values': singular_values,
                'eigenvalues': eigenvalues,
                'spectral_radius': spectral_radius,
                'rank_numerical': rank_numerical,
                'rank_theoretical': rank_theoretical,
                'well_conditioned': well_conditioned,
                'full_rank': full_rank,
                'numerically_stable': stable
            }
            
            # Store results
            self.stability_metrics[matrix_name] = results
            
            # Report
            print(f"  Condition number (2-norm): {cond_2:.2e}")
            print(f"  Condition number (SVD): {cond_svd:.2e}")
            print(f"  Numerical rank: {rank_numerical}/{rank_theoretical}")
            print(f"  Spectral radius: {spectral_radius:.2e}")
            print(f"  Status: {'✅ Stable' if stable else '⚠️ Potentially unstable'}")
            
            return results
            
        except Exception as e:
            error_msg = f"Matrix conditioning analysis failed: {e}"
            warnings.warn(error_msg)
            return {'error': error_msg}
    
    def analyze_ode_stability(self,
                            jacobian_func: Callable[[float, np.ndarray], np.ndarray],
                            t_points: np.ndarray,
                            y_points: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability of ODE system
        
        Args:
            jacobian_func: Function computing Jacobian matrix
            t_points: Time points
            y_points: Solution points
            
        Returns:
            Stability analysis results
        """
        print("🔍 Analyzing ODE stability")
        
        stability_results = {
            't_points': [],
            'max_eigenvalue_real': [],
            'max_eigenvalue_magnitude': [],
            'stiff_ratio': [],
            'stable_points': []
        }
        
        try:
            for i, (t, y) in enumerate(zip(t_points, y_points)):
                if i % max(1, len(t_points) // 20) == 0:  # Sample points
                    
                    try:
                        # Compute Jacobian at this point
                        J = jacobian_func(t, y)
                        
                        # Eigenvalue analysis
                        eigenvals = linalg.eigvals(J)
                        
                        max_real = np.max(np.real(eigenvals))
                        max_magnitude = np.max(np.abs(eigenvals))
                        min_magnitude = np.min(np.abs(eigenvals[np.abs(eigenvals) > 1e-15]))
                        
                        # Stiffness ratio
                        stiff_ratio = max_magnitude / min_magnitude if min_magnitude > 0 else np.inf
                        
                        # Stability check (all eigenvalues have negative real parts)
                        stable = np.all(np.real(eigenvals) < 0)
                        
                        stability_results['t_points'].append(t)
                        stability_results['max_eigenvalue_real'].append(max_real)
                        stability_results['max_eigenvalue_magnitude'].append(max_magnitude)
                        stability_results['stiff_ratio'].append(stiff_ratio)
                        stability_results['stable_points'].append(stable)
                        
                    except Exception as e:
                        warnings.warn(f"Stability analysis failed at t={t}: {e}")
            
            # Overall stability assessment
            if stability_results['stable_points']:
                fraction_stable = np.mean(stability_results['stable_points'])
                max_stiff_ratio = np.max(stability_results['stiff_ratio'])
                avg_stiff_ratio = np.mean(np.array(stability_results['stiff_ratio'])[np.isfinite(stability_results['stiff_ratio'])])
                
                stability_results['fraction_stable'] = fraction_stable
                stability_results['max_stiffness_ratio'] = max_stiff_ratio
                stability_results['average_stiffness_ratio'] = avg_stiff_ratio
                stability_results['overall_stable'] = fraction_stable > 0.9
                stability_results['stiff_system'] = max_stiff_ratio > 1000
                
                print(f"  Fraction of stable points: {fraction_stable:.1%}")
                print(f"  Maximum stiffness ratio: {max_stiff_ratio:.2e}")
                print(f"  Average stiffness ratio: {avg_stiff_ratio:.2e}")
                print(f"  Overall stability: {'✅ Stable' if stability_results['overall_stable'] else '⚠️ Unstable'}")
                print(f"  Stiffness: {'⚠️ Stiff' if stability_results['stiff_system'] else '✅ Non-stiff'}")
            
        except Exception as e:
            error_msg = f"ODE stability analysis failed: {e}"
            warnings.warn(error_msg)
            stability_results['error'] = error_msg
        
        return stability_results
    
    def recommend_numerical_method(self, 
                                 problem_characteristics: Dict[str, Any]) -> Dict[str, str]:
        """Recommend numerical methods based on problem characteristics"""
        
        recommendations = {
            'ode_solver': 'RK45',
            'linear_solver': 'LU',
            'eigenvalue_solver': 'eigs',
            'optimization_method': 'BFGS'
        }
        
        # Analyze characteristics
        is_stiff = problem_characteristics.get('stiff_system', False)
        is_large = problem_characteristics.get('large_system', False)
        is_sparse = problem_characteristics.get('sparse_matrices', False)
        is_ill_conditioned = problem_characteristics.get('ill_conditioned', False)
        
        # ODE solver recommendations
        if is_stiff:
            recommendations['ode_solver'] = 'BDF'  # Backward differentiation formula
        elif is_large:
            recommendations['ode_solver'] = 'CVODE'  # Variable-coefficient ODE solver
        
        # Linear solver recommendations
        if is_sparse and is_large:
            recommendations['linear_solver'] = 'SPARSE_LU'
        elif is_ill_conditioned:
            recommendations['linear_solver'] = 'SVD'
        elif is_large:
            recommendations['linear_solver'] = 'GMRES'
        
        # Eigenvalue solver recommendations
        if is_sparse and is_large:
            recommendations['eigenvalue_solver'] = 'ARPACK'
        elif is_ill_conditioned:
            recommendations['eigenvalue_solver'] = 'SVD'
        
        # Optimization recommendations
        if is_ill_conditioned:
            recommendations['optimization_method'] = 'Nelder-Mead'
        elif is_large:
            recommendations['optimization_method'] = 'L-BFGS-B'
        
        return recommendations

class ModelValidationFramework:
    """Comprehensive model validation and cross-validation"""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
        self.validation_results = {}
        
    def cross_validate_model(self,
                           model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_name: str = "Model") -> Dict[str, Any]:
        """
        Perform k-fold cross-validation
        
        Args:
            model: Model with fit/predict interface
            X: Input features
            y: Target values
            model_name: Name for reporting
            
        Returns:
            Cross-validation results
        """
        print(f"🔄 Cross-validating {model_name}")
        
        try:
            # K-fold cross-validation
            kfold = KFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
            
            scores = {
                'mse': [],
                'rmse': [],
                'mae': [],
                'r2': []
            }
            
            fold_predictions = []
            fold_actuals = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model_copy = self._copy_model(model)
                model_copy.fit(X_train, y_train)
                
                # Predict
                y_pred = model_copy.predict(X_val)
                
                # Compute metrics
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                scores['mse'].append(mse)
                scores['rmse'].append(rmse)
                scores['mae'].append(mae)
                scores['r2'].append(r2)
                
                fold_predictions.extend(y_pred)
                fold_actuals.extend(y_val)
                
                print(f"  Fold {fold + 1}: RMSE = {rmse:.6f}, R² = {r2:.4f}")
            
            # Compute statistics
            cv_results = {}
            for metric, values in scores.items():
                cv_results[f'{metric}_mean'] = np.mean(values)
                cv_results[f'{metric}_std'] = np.std(values)
                cv_results[f'{metric}_min'] = np.min(values)
                cv_results[f'{metric}_max'] = np.max(values)
            
            # Overall metrics
            overall_rmse = np.sqrt(mean_squared_error(fold_actuals, fold_predictions))
            overall_r2 = r2_score(fold_actuals, fold_predictions)
            
            cv_results['overall_rmse'] = overall_rmse
            cv_results['overall_r2'] = overall_r2
            cv_results['fold_predictions'] = np.array(fold_predictions)
            cv_results['fold_actuals'] = np.array(fold_actuals)
            
            print(f"  Overall: RMSE = {overall_rmse:.6f} ± {cv_results['rmse_std']:.6f}")
            print(f"           R² = {overall_r2:.4f} ± {cv_results['r2_std']:.4f}")
            
            self.validation_results[model_name] = cv_results
            
            return cv_results
            
        except Exception as e:
            error_msg = f"Cross-validation failed: {e}"
            warnings.warn(error_msg)
            return {'error': error_msg}
    
    def bootstrap_validation(self,
                           model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_name: str = "Model",
                           n_bootstrap: Optional[int] = None) -> Dict[str, Any]:
        """
        Bootstrap validation for confidence intervals
        
        Args:
            model: Model with fit/predict interface
            X: Input features
            y: Target values
            model_name: Name for reporting
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap validation results
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.bootstrap_samples
        
        print(f"🎲 Bootstrap validation for {model_name} ({n_bootstrap} samples)")
        
        try:
            bootstrap_scores = []
            n_samples = len(X)
            
            for i in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                # Out-of-bag samples
                oob_indices = np.setdiff1d(np.arange(n_samples), indices)
                if len(oob_indices) > 0:
                    X_oob = X[oob_indices]
                    y_oob = y[oob_indices]
                    
                    # Fit and predict
                    model_copy = self._copy_model(model)
                    model_copy.fit(X_boot, y_boot)
                    y_pred_oob = model_copy.predict(X_oob)
                    
                    # Compute score
                    score = r2_score(y_oob, y_pred_oob)
                    bootstrap_scores.append(score)
                
                if (i + 1) % (n_bootstrap // 10) == 0:
                    print(f"  Bootstrap progress: {i + 1}/{n_bootstrap}")
            
            # Compute confidence intervals
            alpha = 1 - self.config.confidence_level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            mean_score = np.mean(bootstrap_scores)
            std_score = np.std(bootstrap_scores)
            ci_lower = np.percentile(bootstrap_scores, lower_percentile)
            ci_upper = np.percentile(bootstrap_scores, upper_percentile)
            
            bootstrap_results = {
                'bootstrap_scores': bootstrap_scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'confidence_interval': (ci_lower, ci_upper),
                'confidence_level': self.config.confidence_level
            }
            
            print(f"  Bootstrap R²: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  {self.config.confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            return bootstrap_results
            
        except Exception as e:
            error_msg = f"Bootstrap validation failed: {e}"
            warnings.warn(error_msg)
            return {'error': error_msg}
    
    def _copy_model(self, model):
        """Create a copy of the model"""
        # This is a simplified copy - in practice, you'd use proper model cloning
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback for non-sklearn models
            import copy
            return copy.deepcopy(model)
    
    def residual_analysis(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze model residuals for validation"""
        
        residuals = y_true - y_pred
        
        # Basic statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75)
        }
        
        # Normality test (Shapiro-Wilk for small samples)
        if len(residuals) <= 5000:
            from scipy.stats import shapiro
            shapiro_stat, shapiro_p = shapiro(residuals)
            residual_stats['normality_test'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Homoscedasticity check
        # Simple check: correlation between |residuals| and predictions
        homoscedasticity_corr = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
        residual_stats['homoscedasticity'] = {
            'correlation': homoscedasticity_corr,
            'is_homoscedastic': abs(homoscedasticity_corr) < 0.3
        }
        
        print(f"📊 Residual Analysis:")
        print(f"  Mean: {residual_stats['mean']:.6f}")
        print(f"  Std: {residual_stats['std']:.6f}")
        print(f"  Normality: {'✅ Normal' if residual_stats.get('normality_test', {}).get('is_normal', False) else '⚠️ Non-normal'}")
        print(f"  Homoscedasticity: {'✅ Homoscedastic' if residual_stats['homoscedasticity']['is_homoscedastic'] else '⚠️ Heteroscedastic'}")
        
        return residual_stats

# Example usage and testing
if __name__ == "__main__":
    print("🔧 Testing Numerical Stability & Validation Framework")
    print("=" * 60)
    
    # Configuration
    config = NumericalConfig(
        error_tolerance=1e-8,
        max_iterations=1000,
        cross_validation_folds=3,
        bootstrap_samples=100
    )
    
    # Test 1: Adaptive ODE Solver
    print("\n🧮 Testing Adaptive ODE Solver...")
    
    def harmonic_oscillator(t, y):
        """Simple harmonic oscillator: d²x/dt² + ω²x = 0"""
        x, v = y
        omega = 2.0  # Angular frequency
        return np.array([v, -omega**2 * x])
    
    solver = AdaptiveNumericalSolver(config)
    
    # Initial conditions: x(0) = 1, v(0) = 0
    y0 = np.array([1.0, 0.0])
    t_span = (0.0, 2*np.pi)  # One period
    
    solution = solver.solve_ivp_adaptive(harmonic_oscillator, t_span, y0)
    
    print(f"  Integration steps: {solution['n_steps']}")
    print(f"  Rejected steps: {solution['rejected_steps']}")
    print(f"  Final step size: {solution['final_step_size']:.2e}")
    print(f"  Converged: {'✅ Yes' if solution['converged'] else '❌ No'}")
    
    # Check conservation of energy
    if solution['converged']:
        x_values = solution['y'][:, 0]
        v_values = solution['y'][:, 1]
        energy = 0.5 * v_values**2 + 0.5 * (2.0**2) * x_values**2
        energy_conservation = np.std(energy) / np.mean(energy)
        print(f"  Energy conservation error: {energy_conservation:.2e}")
    
    # Test 2: Matrix Conditioning Analysis
    print("\n🔍 Testing Matrix Conditioning Analysis...")
    
    stability_analyzer = NumericalStabilityAnalyzer(config)
    
    # Well-conditioned matrix
    A_good = np.array([[4, 1], [1, 3]])
    results_good = stability_analyzer.analyze_matrix_conditioning(A_good, "Well-conditioned matrix")
    
    # Ill-conditioned matrix
    epsilon = 1e-10
    A_bad = np.array([[1, 1], [1, 1+epsilon]])
    results_bad = stability_analyzer.analyze_matrix_conditioning(A_bad, "Ill-conditioned matrix")
    
    # Test 3: Model Validation
    print("\n🔄 Testing Model Validation Framework...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    
    X = np.random.randn(n_samples, n_features)
    true_coeffs = np.array([2.0, -1.5, 0.8])
    y = X @ true_coeffs + 0.1 * np.random.randn(n_samples)
    
    # Simple linear model for testing
    class SimpleLinearModel:
        def __init__(self):
            self.coefficients = None
        
        def fit(self, X, y):
            self.coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        
        def predict(self, X):
            return X @ self.coefficients
    
    model = SimpleLinearModel()
    validator = ModelValidationFramework(config)
    
    # Cross-validation
    cv_results = validator.cross_validate_model(model, X, y, "Linear Model")
    
    # Bootstrap validation
    bootstrap_results = validator.bootstrap_validation(model, X, y, "Linear Model", n_bootstrap=50)
    
    # Fit full model for residual analysis
    model.fit(X, y)
    y_pred = model.predict(X)
    
    residual_results = validator.residual_analysis(y, y_pred)
    
    # Test 4: ODE Stability Analysis
    print("\n📈 Testing ODE Stability Analysis...")
    
    def jacobian_harmonic_oscillator(t, y):
        """Jacobian of harmonic oscillator"""
        omega = 2.0
        return np.array([[0, 1], [-omega**2, 0]])
    
    if solution['converged']:
        ode_stability = stability_analyzer.analyze_ode_stability(
            jacobian_harmonic_oscillator,
            solution['t'][::10],  # Sample points
            solution['y'][::10]   # Corresponding solutions
        )
    
    # Test 5: Method Recommendations
    print("\n💡 Testing Method Recommendations...")
    
    problem_chars = {
        'stiff_system': results_bad['condition_number_2'] > 1e6,
        'large_system': False,
        'sparse_matrices': False,
        'ill_conditioned': results_bad['condition_number_2'] > 1e10
    }
    
    recommendations = stability_analyzer.recommend_numerical_method(problem_chars)
    
    print("  Recommended methods:")
    for method_type, method in recommendations.items():
        print(f"    {method_type}: {method}")
    
    print("\n✅ Numerical Stability & Validation Framework implementation complete!")
    
    # Summary of key results
    print(f"\n📊 Summary:")
    print(f"  ODE solver convergence: {'✅' if solution['converged'] else '❌'}")
    print(f"  Well-conditioned matrix: {'✅' if results_good['numerically_stable'] else '❌'}")
    print(f"  Ill-conditioned matrix: {'⚠️' if not results_bad['numerically_stable'] else '❌'}")
    print(f"  Model validation R²: {cv_results.get('overall_r2', 0):.4f}")
    print(f"  Bootstrap CI width: {bootstrap_results.get('confidence_interval', (0,0))[1] - bootstrap_results.get('confidence_interval', (0,0))[0]:.4f}")
