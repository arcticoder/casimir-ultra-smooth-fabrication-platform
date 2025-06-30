"""
Predictive Control for Digital Twin
==================================

Advanced control algorithms including:
- Stochastic Model Predictive Control (SMPC)
- Failure Prediction with Probability Assessment
- Robust Control under Uncertainty

Mathematical formulations:
min_u (1/N_s) Σᵢ Σₖ ||xₖⁱ - rₖ||²_Q + ||uₖ||²_R
s.t. Pr[constraint violation] ≤ ε

P_fail(k) = 1 - (1 - P_pos(k))(1 - P_vel(k))
τ_fail = min{k : P_fail(k) > 0.1}
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import scipy.optimize
import warnings
from scipy import stats

@dataclass
class ControlConfig:
    """Configuration for predictive control"""
    prediction_horizon: int = 10
    control_horizon: int = 5
    num_scenarios: int = 100
    
    # SMPC parameters
    chance_constraint_probability: float = 0.9
    risk_level: float = 0.1
    
    # Failure prediction
    failure_threshold: float = 0.1
    position_tolerance: float = 1e-8  # m
    velocity_tolerance: float = 1e-6  # m/s
    
    # Optimization
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Weights
    state_weight_matrix: Optional[np.ndarray] = None
    control_weight_matrix: Optional[np.ndarray] = None

class StochasticModelPredictiveControl:
    """
    Stochastic Model Predictive Control with Uncertainty
    
    min_u (1/N_s) Σᵢ₌₁^N_s Σₖ₌₀^(N-1) ||xₖⁱ - rₖ||²_Q + ||uₖ||²_R
    s.t. Pr[constraint violation] ≤ ε
    """
    
    def __init__(self, config: ControlConfig, state_dim: int, control_dim: int):
        """Initialize SMPC controller"""
        self.config = config
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        # Weight matrices
        if config.state_weight_matrix is not None:
            self.Q = config.state_weight_matrix
        else:
            self.Q = np.eye(state_dim)
        
        if config.control_weight_matrix is not None:
            self.R = config.control_weight_matrix
        else:
            self.R = np.eye(control_dim)
        
        # Constraint handling
        self.state_constraints = []
        self.control_constraints = []
        
    def add_state_constraint(self, 
                           constraint_func: Callable[[np.ndarray], float],
                           constraint_type: str = "upper",
                           limit: float = 0.0):
        """
        Add state constraint
        
        Args:
            constraint_func: Function g(x) for constraint g(x) ≤ 0 (upper) or g(x) ≥ 0 (lower)
            constraint_type: "upper" or "lower"
            limit: Constraint limit
        """
        self.state_constraints.append({
            'function': constraint_func,
            'type': constraint_type,
            'limit': limit
        })
    
    def add_control_constraint(self, u_min: np.ndarray, u_max: np.ndarray):
        """Add box constraints on control inputs"""
        self.u_min = u_min
        self.u_max = u_max
    
    def generate_scenarios(self,
                          x0: np.ndarray,
                          dynamics_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                          noise_cov: np.ndarray) -> List[np.ndarray]:
        """
        Generate uncertainty scenarios for SMPC
        
        Args:
            x0: Initial state
            dynamics_func: Stochastic dynamics x_{k+1} = f(x_k, u_k, w_k)
            noise_cov: Process noise covariance
            
        Returns:
            List of noise scenarios
        """
        scenarios = []
        
        for _ in range(self.config.num_scenarios):
            # Generate noise sequence for prediction horizon
            noise_sequence = np.random.multivariate_normal(
                np.zeros(self.state_dim),
                noise_cov,
                size=self.config.prediction_horizon
            )
            scenarios.append(noise_sequence)
        
        return scenarios
    
    def predict_scenarios(self,
                         x0: np.ndarray,
                         u_sequence: np.ndarray,
                         scenarios: List[np.ndarray],
                         dynamics_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> List[np.ndarray]:
        """
        Predict state trajectories for all scenarios
        
        Args:
            x0: Initial state
            u_sequence: Control sequence [u_0, u_1, ..., u_{N-1}]
            scenarios: List of noise scenarios
            dynamics_func: Dynamics function
            
        Returns:
            List of state trajectories
        """
        trajectories = []
        
        for scenario in scenarios:
            trajectory = [x0]
            x_current = x0.copy()
            
            for k in range(self.config.prediction_horizon):
                # Get control input (repeat last if sequence is shorter)
                u_k = u_sequence[min(k, len(u_sequence) - 1)]
                w_k = scenario[k]
                
                # Predict next state
                try:
                    x_next = dynamics_func(x_current, u_k, w_k)
                    trajectory.append(x_next)
                    x_current = x_next
                except Exception as e:
                    warnings.warn(f"Dynamics prediction failed: {e}")
                    trajectory.append(x_current)  # Stay at current state
            
            trajectories.append(np.array(trajectory))
        
        return trajectories
    
    def evaluate_chance_constraints(self,
                                  trajectories: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate chance constraint satisfaction
        
        Returns:
            Dictionary with constraint violation probabilities
        """
        constraint_violations = {}
        
        for i, constraint in enumerate(self.state_constraints):
            violations = 0
            total_evaluations = 0
            
            for trajectory in trajectories:
                for x in trajectory[1:]:  # Skip initial state
                    try:
                        constraint_value = constraint['function'](x)
                        
                        if constraint['type'] == 'upper':
                            violated = constraint_value > constraint['limit']
                        else:  # lower
                            violated = constraint_value < constraint['limit']
                        
                        if violated:
                            violations += 1
                        total_evaluations += 1
                        
                    except Exception:
                        violations += 1  # Conservative: assume violation
                        total_evaluations += 1
            
            violation_probability = violations / total_evaluations if total_evaluations > 0 else 1.0
            constraint_violations[f'constraint_{i}'] = violation_probability
        
        return constraint_violations
    
    def objective_function(self,
                          u_flat: np.ndarray,
                          x0: np.ndarray,
                          reference_trajectory: np.ndarray,
                          scenarios: List[np.ndarray],
                          dynamics_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> float:
        """
        SMPC objective function
        
        J = (1/N_s) Σᵢ Σₖ ||xₖⁱ - rₖ||²_Q + ||uₖ||²_R
        """
        # Reshape control sequence
        u_sequence = u_flat.reshape((self.config.control_horizon, self.control_dim))
        
        # Predict trajectories for all scenarios
        trajectories = self.predict_scenarios(x0, u_sequence, scenarios, dynamics_func)
        
        # Compute cost
        total_cost = 0.0
        num_scenarios = len(trajectories)
        
        for trajectory in trajectories:
            scenario_cost = 0.0
            
            # State cost
            for k in range(min(len(trajectory) - 1, self.config.prediction_horizon)):
                x_k = trajectory[k + 1]  # Skip initial state
                r_k = reference_trajectory[min(k, len(reference_trajectory) - 1)]
                
                state_error = x_k - r_k
                scenario_cost += state_error.T @ self.Q @ state_error
            
            # Control cost
            for k in range(min(len(u_sequence), self.config.control_horizon)):
                u_k = u_sequence[k]
                scenario_cost += u_k.T @ self.R @ u_k
            
            total_cost += scenario_cost
        
        # Average over scenarios
        average_cost = total_cost / num_scenarios
        
        # Add chance constraint penalty
        constraint_violations = self.evaluate_chance_constraints(trajectories)
        penalty = 0.0
        
        for violation_prob in constraint_violations.values():
            if violation_prob > (1.0 - self.config.chance_constraint_probability):
                penalty += 1e6 * (violation_prob - (1.0 - self.config.chance_constraint_probability))**2
        
        return average_cost + penalty
    
    def solve(self,
              x0: np.ndarray,
              reference_trajectory: np.ndarray,
              dynamics_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
              noise_cov: np.ndarray) -> Dict[str, Any]:
        """
        Solve SMPC optimization problem
        
        Args:
            x0: Current state
            reference_trajectory: Reference trajectory
            dynamics_func: Stochastic dynamics function
            noise_cov: Process noise covariance
            
        Returns:
            Optimization results
        """
        # Generate scenarios
        scenarios = self.generate_scenarios(x0, dynamics_func, noise_cov)
        
        # Initial guess for control sequence
        u0 = np.zeros((self.config.control_horizon, self.control_dim))
        u0_flat = u0.flatten()
        
        # Bounds for control inputs
        bounds = []
        if hasattr(self, 'u_min') and hasattr(self, 'u_max'):
            for _ in range(self.config.control_horizon):
                for i in range(self.control_dim):
                    bounds.append((self.u_min[i], self.u_max[i]))
        else:
            bounds = None
        
        # Solve optimization
        try:
            result = scipy.optimize.minimize(
                fun=lambda u: self.objective_function(u, x0, reference_trajectory, scenarios, dynamics_func),
                x0=u0_flat,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.tolerance
                }
            )
            
            if result.success:
                optimal_u = result.x.reshape((self.config.control_horizon, self.control_dim))
                
                # Evaluate final constraint satisfaction
                final_trajectories = self.predict_scenarios(x0, optimal_u, scenarios, dynamics_func)
                constraint_violations = self.evaluate_chance_constraints(final_trajectories)
                
                return {
                    'success': True,
                    'optimal_control': optimal_u[0],  # Return first control action
                    'optimal_sequence': optimal_u,
                    'cost': result.fun,
                    'constraint_violations': constraint_violations,
                    'scenarios': scenarios,
                    'trajectories': final_trajectories
                }
            else:
                warnings.warn(f"SMPC optimization failed: {result.message}")
                return {
                    'success': False,
                    'optimal_control': np.zeros(self.control_dim),
                    'error': result.message
                }
                
        except Exception as e:
            warnings.warn(f"SMPC optimization error: {e}")
            return {
                'success': False,
                'optimal_control': np.zeros(self.control_dim),
                'error': str(e)
            }

class FailurePredictionSystem:
    """
    Failure Prediction with Probability Assessment
    
    P_fail(k) = 1 - (1 - P_pos(k))(1 - P_vel(k))
    τ_fail = min{k : P_fail(k) > 0.1}
    """
    
    def __init__(self, config: ControlConfig):
        """Initialize failure prediction system"""
        self.config = config
        self.failure_history = []
        
    def predict_position_failure(self,
                               state_trajectory: np.ndarray,
                               position_limits: Tuple[float, float]) -> np.ndarray:
        """
        Predict position constraint violation probability
        
        Args:
            state_trajectory: Predicted state trajectory
            position_limits: (min_position, max_position)
            
        Returns:
            Array of failure probabilities at each time step
        """
        min_pos, max_pos = position_limits
        failure_probs = np.zeros(len(state_trajectory))
        
        for k, state in enumerate(state_trajectory):
            position = state[0]  # Assume first state is position
            
            # Distance to constraints
            dist_to_min = position - min_pos
            dist_to_max = max_pos - position
            
            # Probability of violation (using normal distribution assumption)
            sigma_pos = self.config.position_tolerance
            
            prob_min_violation = stats.norm.cdf(min_pos, loc=position, scale=sigma_pos)
            prob_max_violation = 1.0 - stats.norm.cdf(max_pos, loc=position, scale=sigma_pos)
            
            failure_probs[k] = prob_min_violation + prob_max_violation
        
        return np.clip(failure_probs, 0.0, 1.0)
    
    def predict_velocity_failure(self,
                               state_trajectory: np.ndarray,
                               velocity_limits: Tuple[float, float]) -> np.ndarray:
        """
        Predict velocity constraint violation probability
        
        Args:
            state_trajectory: Predicted state trajectory
            velocity_limits: (min_velocity, max_velocity)
            
        Returns:
            Array of failure probabilities at each time step
        """
        min_vel, max_vel = velocity_limits
        failure_probs = np.zeros(len(state_trajectory))
        
        for k, state in enumerate(state_trajectory):
            if len(state) > 1:
                velocity = state[1]  # Assume second state is velocity
            else:
                velocity = 0.0  # Default if no velocity state
            
            # Probability of violation
            sigma_vel = self.config.velocity_tolerance
            
            prob_min_violation = stats.norm.cdf(min_vel, loc=velocity, scale=sigma_vel)
            prob_max_violation = 1.0 - stats.norm.cdf(max_vel, loc=velocity, scale=sigma_vel)
            
            failure_probs[k] = prob_min_violation + prob_max_violation
        
        return np.clip(failure_probs, 0.0, 1.0)
    
    def compute_total_failure_probability(self,
                                        state_trajectory: np.ndarray,
                                        position_limits: Tuple[float, float],
                                        velocity_limits: Tuple[float, float]) -> Dict[str, Any]:
        """
        Compute total failure probability
        
        P_fail(k) = 1 - (1 - P_pos(k))(1 - P_vel(k))
        τ_fail = min{k : P_fail(k) > 0.1}
        """
        # Individual failure probabilities
        p_pos = self.predict_position_failure(state_trajectory, position_limits)
        p_vel = self.predict_velocity_failure(state_trajectory, velocity_limits)
        
        # Total failure probability (assuming independence)
        p_total = 1.0 - (1.0 - p_pos) * (1.0 - p_vel)
        
        # Time to failure
        failure_times = np.where(p_total > self.config.failure_threshold)[0]
        
        if len(failure_times) > 0:
            time_to_failure = failure_times[0]
            failure_predicted = True
        else:
            time_to_failure = len(state_trajectory)  # No failure predicted
            failure_predicted = False
        
        return {
            'position_failure_probs': p_pos,
            'velocity_failure_probs': p_vel,
            'total_failure_probs': p_total,
            'time_to_failure': time_to_failure,
            'failure_predicted': failure_predicted,
            'max_failure_prob': np.max(p_total)
        }
    
    def update_failure_model(self, 
                           actual_failure: bool,
                           predicted_failure: bool,
                           prediction_horizon: int):
        """Update failure prediction model based on observations"""
        self.failure_history.append({
            'actual': actual_failure,
            'predicted': predicted_failure,
            'horizon': prediction_horizon
        })
        
        # Simple adaptive threshold (could be more sophisticated)
        if len(self.failure_history) > 10:
            recent_history = self.failure_history[-10:]
            false_positives = sum(1 for h in recent_history if h['predicted'] and not h['actual'])
            false_negatives = sum(1 for h in recent_history if not h['predicted'] and h['actual'])
            
            if false_positives > 3:  # Too many false alarms
                self.config.failure_threshold *= 1.1  # Increase threshold
            elif false_negatives > 1:  # Missing failures
                self.config.failure_threshold *= 0.9  # Decrease threshold
            
            # Keep threshold in reasonable range
            self.config.failure_threshold = np.clip(self.config.failure_threshold, 0.01, 0.5)

class RobustMPCController:
    """Combined SMPC with Failure Prediction"""
    
    def __init__(self, config: ControlConfig, state_dim: int, control_dim: int):
        """Initialize robust MPC controller"""
        self.config = config
        self.smpc = StochasticModelPredictiveControl(config, state_dim, control_dim)
        self.failure_predictor = FailurePredictionSystem(config)
        
        # Operating constraints
        self.position_limits = (-1e-6, 1e-6)  # ±1 μm default
        self.velocity_limits = (-1e-3, 1e-3)  # ±1 mm/s default
        
    def set_constraints(self,
                       position_limits: Tuple[float, float],
                       velocity_limits: Tuple[float, float],
                       control_limits: Tuple[np.ndarray, np.ndarray]):
        """Set system constraints"""
        self.position_limits = position_limits
        self.velocity_limits = velocity_limits
        self.smpc.add_control_constraint(*control_limits)
        
        # Add position constraint to SMPC
        def position_constraint(x):
            return max(self.position_limits[0] - x[0], x[0] - self.position_limits[1])
        
        self.smpc.add_state_constraint(position_constraint, "upper", 0.0)
    
    def control_step(self,
                    current_state: np.ndarray,
                    reference_trajectory: np.ndarray,
                    dynamics_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                    noise_cov: np.ndarray) -> Dict[str, Any]:
        """
        Complete control step with failure prediction
        
        Returns:
            Control action and system analysis
        """
        # Solve SMPC
        smpc_result = self.smpc.solve(current_state, reference_trajectory, dynamics_func, noise_cov)
        
        if not smpc_result['success']:
            return {
                'control_action': np.zeros(self.smpc.control_dim),
                'failure_analysis': {'error': 'SMPC optimization failed'},
                'success': False
            }
        
        # Predict failure using best trajectory
        best_trajectory = smpc_result['trajectories'][0]  # Use first scenario as representative
        
        failure_analysis = self.failure_predictor.compute_total_failure_probability(
            best_trajectory,
            self.position_limits,
            self.velocity_limits
        )
        
        # Modify control if failure predicted
        control_action = smpc_result['optimal_control'].copy()
        
        if failure_analysis['failure_predicted']:
            # Conservative control modification
            control_action *= 0.5  # Reduce aggressiveness
            
            print(f"⚠️ Failure predicted at step {failure_analysis['time_to_failure']}")
            print(f"   Max failure probability: {failure_analysis['max_failure_prob']:.2%}")
        
        return {
            'control_action': control_action,
            'smpc_result': smpc_result,
            'failure_analysis': failure_analysis,
            'success': True,
            'constraint_violations': smpc_result['constraint_violations']
        }

# Example usage and testing
if __name__ == "__main__":
    print("🎮 Testing Predictive Control System")
    print("=" * 50)
    
    # Configuration
    config = ControlConfig(
        prediction_horizon=10,
        control_horizon=5,
        num_scenarios=50,
        chance_constraint_probability=0.9
    )
    
    # System dimensions
    state_dim = 6  # [position, velocity, force, temperature, roughness, defects]
    control_dim = 3  # [voltage, pressure, flow_rate]
    
    # Initialize controller
    controller = RobustMPCController(config, state_dim, control_dim)
    
    # Set constraints
    position_limits = (-500e-9, 500e-9)  # ±500 nm
    velocity_limits = (-1e-6, 1e-6)      # ±1 μm/s
    control_limits = (np.array([-10, -5, -2]), np.array([10, 5, 2]))  # Control bounds
    
    controller.set_constraints(position_limits, velocity_limits, control_limits)
    
    # Test dynamics function
    def test_dynamics(x, u, w):
        """Simple test dynamics with noise"""
        A = np.eye(state_dim)
        A[0, 1] = 0.1  # Position-velocity coupling
        
        B = np.eye(state_dim, control_dim) * 0.1
        
        return A @ x + B @ u + w
    
    # Reference trajectory (step response)
    reference = np.array([100e-9, 0, 1e-12, 300, 0.1, 0.005])  # Target state
    reference_trajectory = np.tile(reference, (config.prediction_horizon, 1))
    
    # Process noise
    noise_cov = np.eye(state_dim) * 1e-12
    
    # Test control steps
    current_state = np.array([0, 0, 0, 300, 0.15, 0.008])  # Initial state
    
    print("🎯 Running Control Simulation...")
    
    for step in range(5):
        print(f"\nStep {step + 1}:")
        print(f"  Current state: {current_state[:3]}")  # Show position, velocity, force
        
        # Compute control action
        result = controller.control_step(
            current_state,
            reference_trajectory,
            test_dynamics,
            noise_cov
        )
        
        if result['success']:
            control_action = result['control_action']
            failure_analysis = result['failure_analysis']
            
            print(f"  Control action: {control_action}")
            print(f"  Failure predicted: {failure_analysis['failure_predicted']}")
            
            if failure_analysis['failure_predicted']:
                print(f"  Time to failure: {failure_analysis['time_to_failure']} steps")
                print(f"  Max failure prob: {failure_analysis['max_failure_prob']:.2%}")
            
            # Simulate system response
            noise = np.random.multivariate_normal(np.zeros(state_dim), noise_cov)
            current_state = test_dynamics(current_state, control_action, noise)
            
        else:
            print("  ❌ Control computation failed")
            break
    
    print("\n✅ Predictive Control implementation complete!")
