"""
Bayesian State Estimation for Digital Twin
==========================================

Advanced state estimation algorithms including:
- Enhanced Unscented Kalman Filter with Sigma Points
- Multi-Algorithm Adaptive State Estimation
- Extended Kalman Filter
- Particle Filter

Mathematical formulations based on:
χ(k|k) = [x̂(k|k), x̂(k|k) ± √((n+λ)P(k|k))]
x̂(k+1|k) = Σᵢ Wᵢᵐ f(χᵢ(k|k), u(k))
P(k+1|k) = Σᵢ Wᵢᶜ [χᵢ(k+1|k) - x̂(k+1|k)][χᵢ(k+1|k) - x̂(k+1|k)]ᵀ + Q
"""

import numpy as np
import scipy.linalg
from typing import Tuple, Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import warnings

class EstimationAlgorithm(Enum):
    """Available estimation algorithms"""
    EKF = "extended_kalman_filter"
    UKF = "unscented_kalman_filter"
    PF = "particle_filter"
    ADAPTIVE = "adaptive_selection"

@dataclass
class StateEstimationConfig:
    """Configuration for state estimation"""
    algorithm: EstimationAlgorithm = EstimationAlgorithm.ADAPTIVE
    state_dim: int = 6
    measurement_dim: int = 4
    process_noise_cov: Optional[np.ndarray] = None
    measurement_noise_cov: Optional[np.ndarray] = None
    initial_state_cov: Optional[np.ndarray] = None
    
    # UKF parameters
    alpha: float = 1e-3  # Spread parameter
    beta: float = 2.0    # Distribution parameter (Gaussian = 2)
    kappa: float = 0.0   # Secondary scaling parameter
    
    # Particle filter parameters
    num_particles: int = 1000
    resample_threshold: float = 0.5
    
    # Adaptive selection thresholds
    nonlinearity_threshold_1: float = 1e-2
    nonlinearity_threshold_2: float = 1e-1

class UnscentedKalmanFilter:
    """
    Enhanced Unscented Kalman Filter with Sigma Points
    
    Implements the mathematical formulation:
    χ(k|k) = [x̂(k|k), x̂(k|k) ± √((n+λ)P(k|k))]
    x̂(k+1|k) = Σᵢ Wᵢᵐ f(χᵢ(k|k), u(k))
    P(k+1|k) = Σᵢ Wᵢᶜ [χᵢ(k+1|k) - x̂(k+1|k)][χᵢ(k+1|k) - x̂(k+1|k)]ᵀ + Q
    """
    
    def __init__(self, config: StateEstimationConfig):
        """Initialize UKF with configuration"""
        self.config = config
        self.n = config.state_dim
        
        # Calculate UKF parameters
        self.lambda_ = config.alpha**2 * (self.n + config.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        
        # Initialize weights
        self._compute_weights()
        
        # State and covariance
        self.x_hat = np.zeros(self.n)
        self.P = config.initial_state_cov if config.initial_state_cov is not None else np.eye(self.n)
        
        # Noise covariances
        self.Q = config.process_noise_cov if config.process_noise_cov is not None else np.eye(self.n) * 1e-6
        self.R = config.measurement_noise_cov if config.measurement_noise_cov is not None else np.eye(config.measurement_dim) * 1e-4
    
    def _compute_weights(self):
        """Compute sigma point weights"""
        # Mean weights
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.config.alpha**2 + self.config.beta)
        
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.n + self.lambda_))
    
    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points according to:
        χ(k|k) = [x̂(k|k), x̂(k|k) ± √((n+λ)P(k|k))]
        """
        try:
            # Compute matrix square root
            sqrt_P = scipy.linalg.cholesky((self.n + self.lambda_) * P, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh((self.n + self.lambda_) * P)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive definiteness
            sqrt_P = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Generate sigma points
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x
        
        for i in range(self.n):
            sigma_points[i + 1] = x + sqrt_P[:, i]
            sigma_points[i + 1 + self.n] = x - sqrt_P[:, i]
        
        return sigma_points
    
    def predict(self, 
                dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                control_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF prediction step
        
        Args:
            dynamics_func: Nonlinear dynamics function f(x, u)
            control_input: Control input u(k)
            
        Returns:
            Predicted state and covariance
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x_hat, self.P)
        
        # Propagate sigma points through dynamics
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            try:
                sigma_points_pred[i] = dynamics_func(sigma_points[i], control_input)
            except Exception as e:
                warnings.warn(f"Dynamics function failed for sigma point {i}: {e}")
                sigma_points_pred[i] = sigma_points[i]  # Fallback
        
        # Compute predicted mean: x̂(k+1|k) = Σᵢ Wᵢᵐ f(χᵢ(k|k), u(k))
        x_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance: P(k+1|k) = Σᵢ Wᵢᶜ [χᵢ(k+1|k) - x̂(k+1|k)][χᵢ(k+1|k) - x̂(k+1|k)]ᵀ + Q
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        
        return x_pred, P_pred
    
    def update(self,
               measurement: np.ndarray,
               measurement_func: Callable[[np.ndarray], np.ndarray],
               x_pred: np.ndarray,
               P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF update step
        
        Args:
            measurement: Measurement z(k)
            measurement_func: Measurement function h(x)
            x_pred: Predicted state
            P_pred: Predicted covariance
            
        Returns:
            Updated state and covariance
        """
        # Generate sigma points for prediction
        sigma_points_pred = self.generate_sigma_points(x_pred, P_pred)
        
        # Propagate sigma points through measurement function
        m_dim = len(measurement)
        sigma_measurements = np.zeros((2 * self.n + 1, m_dim))
        
        for i in range(2 * self.n + 1):
            try:
                sigma_measurements[i] = measurement_func(sigma_points_pred[i])
            except Exception as e:
                warnings.warn(f"Measurement function failed for sigma point {i}: {e}")
                sigma_measurements[i] = measurement_func(x_pred)  # Fallback
        
        # Compute predicted measurement
        z_pred = np.sum(self.Wm[:, np.newaxis] * sigma_measurements, axis=0)
        
        # Compute innovation covariance
        S = self.R.copy()
        for i in range(2 * self.n + 1):
            diff = sigma_measurements[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
        
        # Compute cross-covariance
        T = np.zeros((self.n, m_dim))
        for i in range(2 * self.n + 1):
            x_diff = sigma_points_pred[i] - x_pred
            z_diff = sigma_measurements[i] - z_pred
            T += self.Wc[i] * np.outer(x_diff, z_diff)
        
        # Compute Kalman gain
        try:
            K = T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = T @ np.linalg.pinv(S)
        
        # Update state and covariance
        innovation = measurement - z_pred
        x_updated = x_pred + K @ innovation
        P_updated = P_pred - K @ S @ K.T
        
        return x_updated, P_updated
    
    def step(self,
             measurement: np.ndarray,
             control_input: np.ndarray,
             dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
             measurement_func: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """
        Complete UKF estimation step
        
        Returns:
            Dictionary with estimation results
        """
        # Prediction step
        x_pred, P_pred = self.predict(dynamics_func, control_input)
        
        # Update step
        x_updated, P_updated = self.update(measurement, measurement_func, x_pred, P_pred)
        
        # Store results
        self.x_hat = x_updated
        self.P = P_updated
        
        return {
            'state_estimate': x_updated,
            'covariance': P_updated,
            'state_prediction': x_pred,
            'covariance_prediction': P_pred,
            'algorithm': 'UKF'
        }

class ExtendedKalmanFilter:
    """Extended Kalman Filter for comparison"""
    
    def __init__(self, config: StateEstimationConfig):
        """Initialize EKF"""
        self.config = config
        self.n = config.state_dim
        
        # State and covariance
        self.x_hat = np.zeros(self.n)
        self.P = config.initial_state_cov if config.initial_state_cov is not None else np.eye(self.n)
        
        # Noise covariances
        self.Q = config.process_noise_cov if config.process_noise_cov is not None else np.eye(self.n) * 1e-6
        self.R = config.measurement_noise_cov if config.measurement_noise_cov is not None else np.eye(config.measurement_dim) * 1e-4
    
    def compute_jacobian(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix numerically"""
        eps = 1e-8
        n = len(x)
        f_x = func(x)
        m = len(f_x)
        
        jacobian = np.zeros((m, n))
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            jacobian[:, i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        
        return jacobian
    
    def step(self,
             measurement: np.ndarray,
             control_input: np.ndarray,
             dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
             measurement_func: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """Complete EKF estimation step"""
        
        # Prediction step
        x_pred = dynamics_func(self.x_hat, control_input)
        
        # Compute Jacobian of dynamics
        F = self.compute_jacobian(lambda x: dynamics_func(x, control_input), self.x_hat)
        P_pred = F @ self.P @ F.T + self.Q
        
        # Update step
        z_pred = measurement_func(x_pred)
        H = self.compute_jacobian(measurement_func, x_pred)
        
        # Innovation
        innovation = measurement - z_pred
        S = H @ P_pred @ H.T + self.R
        
        # Kalman gain
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P_pred @ H.T @ np.linalg.pinv(S)
        
        # Update
        x_updated = x_pred + K @ innovation
        P_updated = (np.eye(self.n) - K @ H) @ P_pred
        
        # Store results
        self.x_hat = x_updated
        self.P = P_updated
        
        return {
            'state_estimate': x_updated,
            'covariance': P_updated,
            'state_prediction': x_pred,
            'covariance_prediction': P_pred,
            'algorithm': 'EKF'
        }

class ParticleFilter:
    """Particle Filter for highly nonlinear systems"""
    
    def __init__(self, config: StateEstimationConfig):
        """Initialize particle filter"""
        self.config = config
        self.n = config.state_dim
        self.num_particles = config.num_particles
        
        # Initialize particles
        self.particles = np.random.multivariate_normal(
            np.zeros(self.n), 
            config.initial_state_cov if config.initial_state_cov is not None else np.eye(self.n),
            self.num_particles
        )
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Noise covariances
        self.Q = config.process_noise_cov if config.process_noise_cov is not None else np.eye(self.n) * 1e-6
        self.R = config.measurement_noise_cov if config.measurement_noise_cov is not None else np.eye(config.measurement_dim) * 1e-4
    
    def resample(self):
        """Systematic resampling"""
        N = self.num_particles
        positions = (np.arange(N) + np.random.random()) / N
        
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        
        self.particles = self.particles[indexes]
        self.weights = np.ones(N) / N
    
    def step(self,
             measurement: np.ndarray,
             control_input: np.ndarray,
             dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
             measurement_func: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """Complete particle filter step"""
        
        # Prediction step
        for i in range(self.num_particles):
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            self.particles[i] = dynamics_func(self.particles[i], control_input) + noise
        
        # Update step
        for i in range(self.num_particles):
            z_pred = measurement_func(self.particles[i])
            innovation = measurement - z_pred
            
            # Compute likelihood
            try:
                likelihood = np.exp(-0.5 * innovation.T @ np.linalg.inv(self.R) @ innovation)
                likelihood /= np.sqrt((2 * np.pi) ** len(measurement) * np.linalg.det(self.R))
            except np.linalg.LinAlgError:
                likelihood = 1e-10  # Fallback
            
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Resample if needed
        N_eff = 1.0 / np.sum(self.weights**2)
        if N_eff < self.config.resample_threshold * self.num_particles:
            self.resample()
        
        # Estimate state
        x_estimate = np.average(self.particles, weights=self.weights, axis=0)
        
        # Estimate covariance
        P_estimate = np.zeros((self.n, self.n))
        for i in range(self.num_particles):
            diff = self.particles[i] - x_estimate
            P_estimate += self.weights[i] * np.outer(diff, diff)
        
        return {
            'state_estimate': x_estimate,
            'covariance': P_estimate,
            'particles': self.particles.copy(),
            'weights': self.weights.copy(),
            'algorithm': 'PF'
        }

class AdaptiveStateEstimator:
    """
    Multi-Algorithm Adaptive State Estimation
    
    Implements algorithm selection based on:
    Algorithm = {
        EKF    if ||∇²f|| < ε₁
        UKF    if ε₁ ≤ ||∇²f|| < ε₂
        PF     if ||∇²f|| ≥ ε₂
    }
    """
    
    def __init__(self, config: StateEstimationConfig):
        """Initialize adaptive estimator"""
        self.config = config
        
        # Initialize all estimators
        self.ekf = ExtendedKalmanFilter(config)
        self.ukf = UnscentedKalmanFilter(config)
        self.pf = ParticleFilter(config)
        
        # Algorithm selection parameters
        self.eps1 = config.nonlinearity_threshold_1
        self.eps2 = config.nonlinearity_threshold_2
        
        # Current algorithm
        self.current_algorithm = EstimationAlgorithm.UKF
        self.estimator_map = {
            EstimationAlgorithm.EKF: self.ekf,
            EstimationAlgorithm.UKF: self.ukf,
            EstimationAlgorithm.PF: self.pf
        }
    
    def compute_nonlinearity_measure(self, 
                                   dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                   x: np.ndarray,
                                   u: np.ndarray) -> float:
        """
        Compute nonlinearity measure ||∇²f||
        """
        eps = 1e-6
        n = len(x)
        
        # Compute Hessian numerically
        hessian_norm = 0.0
        
        for i in range(n):
            for j in range(n):
                # Second derivative ∂²f/∂xᵢ∂xⱼ
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                
                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                
                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                
                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                try:
                    f_pp = dynamics_func(x_pp, u)
                    f_pm = dynamics_func(x_pm, u)
                    f_mp = dynamics_func(x_mp, u)
                    f_mm = dynamics_func(x_mm, u)
                    
                    hessian_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
                    hessian_norm += np.linalg.norm(hessian_ij)**2
                    
                except Exception:
                    hessian_norm += 1e-6  # Fallback
        
        return np.sqrt(hessian_norm)
    
    def select_algorithm(self,
                        dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                        x: np.ndarray,
                        u: np.ndarray) -> EstimationAlgorithm:
        """
        Select optimal algorithm based on nonlinearity measure
        """
        nonlinearity = self.compute_nonlinearity_measure(dynamics_func, x, u)
        
        if nonlinearity < self.eps1:
            return EstimationAlgorithm.EKF
        elif nonlinearity < self.eps2:
            return EstimationAlgorithm.UKF
        else:
            return EstimationAlgorithm.PF
    
    def step(self,
             measurement: np.ndarray,
             control_input: np.ndarray,
             dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
             measurement_func: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """Complete adaptive estimation step"""
        
        # Get current state estimate
        current_estimator = self.estimator_map[self.current_algorithm]
        current_state = current_estimator.x_hat if hasattr(current_estimator, 'x_hat') else np.zeros(self.config.state_dim)
        
        # Select algorithm
        selected_algorithm = self.select_algorithm(dynamics_func, current_state, control_input)
        
        # Switch algorithm if needed
        if selected_algorithm != self.current_algorithm:
            self.current_algorithm = selected_algorithm
            print(f"Switching to {selected_algorithm.value}")
        
        # Run selected estimator
        estimator = self.estimator_map[self.current_algorithm]
        result = estimator.step(measurement, control_input, dynamics_func, measurement_func)
        
        # Add algorithm information
        result['selected_algorithm'] = selected_algorithm.value
        result['nonlinearity_measure'] = self.compute_nonlinearity_measure(dynamics_func, current_state, control_input)
        
        return result

# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = StateEstimationConfig(
        algorithm=EstimationAlgorithm.ADAPTIVE,
        state_dim=6,
        measurement_dim=4,
        alpha=1e-3,
        beta=2.0,
        num_particles=500
    )
    
    # Initialize adaptive estimator
    estimator = AdaptiveStateEstimator(config)
    
    # Test dynamics (simple linear system for testing)
    def test_dynamics(x, u):
        A = np.eye(6) + 0.1 * np.random.randn(6, 6) * 0.01
        B = np.eye(6) * 0.1
        return A @ x + B @ u
    
    def test_measurement(x):
        H = np.eye(4, 6)
        return H @ x
    
    # Simulate estimation
    x_true = np.array([1.0, 0.1, 0.01, 300.0, 0.15, 400.0])  # [d, F, T, σ, ρ, η]
    
    for k in range(10):
        # Simulate measurement
        measurement = test_measurement(x_true) + np.random.randn(4) * 0.01
        control = np.random.randn(6) * 0.1
        
        # Run estimation
        result = estimator.step(measurement, control, test_dynamics, test_measurement)
        
        print(f"Step {k}: Algorithm = {result['selected_algorithm']}")
        print(f"  State estimate: {result['state_estimate'][:3]}")
        print(f"  Nonlinearity: {result['nonlinearity_measure']:.2e}")
        print()
        
        # Update true state for next iteration
        x_true = test_dynamics(x_true, control)
    
    print("✅ Bayesian State Estimation implementation complete!")
