"""
Multi-Physics Digital Twin
=========================

Advanced multi-physics integration including:
- Enhanced Multi-Rate Time Integration
- Cross-Domain Correlation Uncertainty Propagation  
- Multi-Domain State Synchronization

Mathematical formulations:
Fast dynamics (10 kHz): x_f^(n+1) = f_fast(x_f^n, x_s^n, u^n)
Slow dynamics (100 Hz): x_s^(m+1) = f_slow(x_s^m, ⟨x_f⟩^m, u^m)

Cross-domain covariance:
C_cross(k) = E[(x_i(k) - μ_i)(x_j(k) - μ_j)^T]
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

@dataclass
class MultiPhysicsConfig:
    """Configuration for multi-physics digital twin"""
    
    # Time integration
    fast_dt: float = 1e-4    # 10 kHz for actuator dynamics
    slow_dt: float = 1e-2    # 100 Hz for thermal dynamics
    meta_dt: float = 1.0     # 1 Hz for process optimization
    
    # Integration ratios
    fast_slow_ratio: int = 100    # fast_dt / slow_dt
    slow_meta_ratio: int = 100    # slow_dt / meta_dt
    
    # Buffer sizes for averaging
    fast_buffer_size: int = 100
    slow_buffer_size: int = 100
    
    # Correlation tracking
    correlation_window: int = 1000
    correlation_threshold: float = 0.1
    
    # Synchronization
    sync_tolerance: float = 1e-6
    max_sync_iterations: int = 10
    
    # Numerical stability
    adaptive_timestepping: bool = True
    error_tolerance: float = 1e-8
    max_step_factor: float = 2.0

@dataclass
class DomainState:
    """State container for a physics domain"""
    states: np.ndarray
    time: float
    domain_name: str
    state_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.state_names:
            self.state_names = [f"state_{i}" for i in range(len(self.states))]

class PhysicsDomain(ABC):
    """Abstract base class for physics domains"""
    
    def __init__(self, domain_name: str, state_dim: int):
        self.domain_name = domain_name
        self.state_dim = state_dim
        self.current_state = DomainState(
            states=np.zeros(state_dim),
            time=0.0,
            domain_name=domain_name
        )
        
    @abstractmethod
    def dynamics(self, 
                state: np.ndarray, 
                time: float, 
                inputs: Dict[str, np.ndarray],
                coupling_states: Dict[str, np.ndarray] = None) -> np.ndarray:
        """Compute state derivatives"""
        pass
    
    @abstractmethod
    def get_coupling_outputs(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get outputs for coupling to other domains"""
        pass

class ActuatorDynamicsDomain(PhysicsDomain):
    """Fast actuator dynamics (10 kHz)"""
    
    def __init__(self):
        super().__init__("actuator", 6)  # [position, velocity, force, voltage, current, temperature]
        
        # Actuator parameters
        self.mass = 1e-6        # kg
        self.damping = 1e-3     # N⋅s/m  
        self.stiffness = 1e6    # N/m
        self.force_constant = 1e-3  # N/A
        self.resistance = 10.0      # Ω
        self.inductance = 1e-3      # H
        self.thermal_resistance = 100  # K/W
        self.thermal_capacitance = 1e-6  # J/K
        
        self.current_state.state_names = [
            "position", "velocity", "force", "voltage", "current", "temperature"
        ]
    
    def dynamics(self, 
                state: np.ndarray, 
                time: float, 
                inputs: Dict[str, np.ndarray],
                coupling_states: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Fast actuator dynamics
        
        ẍ = (F_actuator - c⋅ẋ - k⋅x) / m
        F_actuator = K_f ⋅ i
        L⋅di/dt = V - R⋅i - K_f⋅ẋ  (back EMF)
        """
        pos, vel, force, voltage, current, temp = state
        
        # External inputs
        input_voltage = inputs.get('voltage', np.array([0.0]))[0]
        external_force = inputs.get('external_force', np.array([0.0]))[0]
        
        # Thermal coupling from slow domain
        ambient_temp = 300.0  # Default
        if coupling_states and 'thermal' in coupling_states:
            ambient_temp = coupling_states['thermal'][0]  # Substrate temperature
        
        # Mechanical dynamics
        actuator_force = self.force_constant * current
        net_force = actuator_force + external_force - self.damping * vel - self.stiffness * pos
        acceleration = net_force / self.mass
        
        # Electrical dynamics
        back_emf = self.force_constant * vel
        voltage_drop = input_voltage - self.resistance * current - back_emf
        current_rate = voltage_drop / self.inductance
        
        # Thermal dynamics (fast component)
        power_dissipation = self.resistance * current**2
        thermal_coupling = (ambient_temp - temp) / self.thermal_resistance
        temp_rate = (power_dissipation + thermal_coupling) / self.thermal_capacitance
        
        return np.array([
            vel,                # dx/dt = v
            acceleration,       # dv/dt = a
            actuator_force,     # Force output (for monitoring)
            0.0,               # Voltage (controlled input)
            current_rate,      # di/dt
            temp_rate          # dT/dt
        ])
    
    def get_coupling_outputs(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Outputs for coupling to other domains"""
        pos, vel, force, voltage, current, temp = state
        
        return {
            'position': np.array([pos]),
            'velocity': np.array([vel]),
            'force': np.array([force]),
            'power_dissipation': np.array([self.resistance * current**2]),
            'actuator_temperature': np.array([temp])
        }

class ThermalDynamicsDomain(PhysicsDomain):
    """Thermal dynamics (100 Hz)"""
    
    def __init__(self):
        super().__init__("thermal", 4)  # [substrate_temp, coolant_temp, heat_flux, thermal_stress]
        
        # Thermal parameters
        self.substrate_thermal_mass = 1e-3      # J/K
        self.coolant_thermal_mass = 1e-2        # J/K
        self.thermal_conductance = 1e-2         # W/K
        self.convection_coefficient = 1e-3      # W/(K⋅m²)
        self.surface_area = 1e-4                # m²
        self.thermal_expansion_coeff = 1e-5     # 1/K
        self.reference_temperature = 300.0      # K
        
        self.current_state.state_names = [
            "substrate_temperature", "coolant_temperature", "heat_flux", "thermal_stress"
        ]
    
    def dynamics(self, 
                state: np.ndarray, 
                time: float, 
                inputs: Dict[str, np.ndarray],
                coupling_states: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Thermal dynamics
        
        C_s⋅dT_s/dt = Q_in - Q_out - Q_conduction
        C_c⋅dT_c/dt = Q_coolant - Q_convection
        """
        T_substrate, T_coolant, heat_flux, thermal_stress = state
        
        # External inputs
        coolant_flow = inputs.get('coolant_flow', np.array([1e-6]))[0]  # m³/s
        ambient_temp = inputs.get('ambient_temperature', np.array([300.0]))[0]
        
        # Heat sources from fast domain coupling
        actuator_power = 0.0
        if coupling_states and 'actuator' in coupling_states:
            actuator_power = coupling_states['actuator'][0]  # Power dissipation
        
        # Manufacturing process heat (from meta domain)
        process_heat = 0.0
        if coupling_states and 'process' in coupling_states:
            process_heat = coupling_states['process'][0]  # Process heat generation
        
        # Thermal dynamics
        # Substrate energy balance
        Q_in = actuator_power + process_heat
        Q_conduction = self.thermal_conductance * (T_substrate - T_coolant)
        Q_convection = self.convection_coefficient * self.surface_area * (T_substrate - ambient_temp)
        
        dT_substrate_dt = (Q_in - Q_conduction - Q_convection) / self.substrate_thermal_mass
        
        # Coolant energy balance  
        Q_coolant_removal = coolant_flow * 4180 * 1000 * (T_coolant - ambient_temp)  # Water properties
        dT_coolant_dt = (Q_conduction - Q_coolant_removal) / self.coolant_thermal_mass
        
        # Heat flux calculation
        heat_flux_new = Q_conduction / self.surface_area
        
        # Thermal stress from temperature gradient
        thermal_strain = self.thermal_expansion_coeff * (T_substrate - self.reference_temperature)
        # Simplified stress (would need full thermomechanical coupling in reality)
        thermal_stress_new = 2e11 * thermal_strain  # Young's modulus * strain
        
        return np.array([
            dT_substrate_dt,
            dT_coolant_dt,
            heat_flux_new,
            thermal_stress_new
        ])
    
    def get_coupling_outputs(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Outputs for coupling to other domains"""
        T_substrate, T_coolant, heat_flux, thermal_stress = state
        
        return {
            'substrate_temperature': np.array([T_substrate]),
            'coolant_temperature': np.array([T_coolant]),
            'heat_flux': np.array([heat_flux]),
            'thermal_stress': np.array([thermal_stress]),
            'temperature_gradient': np.array([T_substrate - T_coolant])
        }

class ProcessOptimizationDomain(PhysicsDomain):
    """Process optimization (1 Hz)"""
    
    def __init__(self):
        super().__init__("process", 5)  # [roughness, defect_density, removal_rate, quality_metric, optimization_state]
        
        # Process parameters
        self.target_roughness = 0.2e-9     # m RMS
        self.target_defect_density = 0.01  # defects/μm²
        self.removal_rate_nominal = 1e-9   # m/s
        self.quality_weight_roughness = 0.5
        self.quality_weight_defects = 0.3
        self.quality_weight_rate = 0.2
        
        self.current_state.state_names = [
            "surface_roughness", "defect_density", "removal_rate", "quality_metric", "optimization_state"
        ]
    
    def dynamics(self, 
                state: np.ndarray, 
                time: float, 
                inputs: Dict[str, np.ndarray],
                coupling_states: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Process optimization dynamics
        
        Slow evolution of surface quality based on thermal and mechanical conditions
        """
        roughness, defect_density, removal_rate, quality_metric, opt_state = state
        
        # Process control inputs
        pressure = inputs.get('pressure', np.array([1e5]))[0]  # Pa
        flow_rate = inputs.get('flow_rate', np.array([1e-6]))[0]  # m³/s
        
        # Coupling from fast domains
        mechanical_stress = 0.0
        temperature = 300.0
        
        if coupling_states:
            if 'actuator' in coupling_states:
                mechanical_stress = abs(coupling_states['actuator'][2])  # Force magnitude
            if 'thermal' in coupling_states:
                temperature = coupling_states['thermal'][0]  # Substrate temperature
        
        # Surface evolution models
        # Roughness evolution (simplified)
        temp_factor = np.exp(-(temperature - 300) / 50)  # Temperature effect
        stress_factor = 1 + mechanical_stress / 1e-6     # Stress effect
        
        roughness_rate = -1e-12 * pressure * flow_rate * temp_factor / stress_factor
        roughness_rate += 1e-13 * mechanical_stress  # Stress-induced roughening
        
        # Defect evolution
        thermal_stress = abs(temperature - 300) / 100
        defect_generation_rate = 1e-6 * thermal_stress * mechanical_stress
        defect_removal_rate = 1e-7 * pressure * flow_rate
        
        defect_rate = defect_generation_rate - defect_removal_rate
        
        # Removal rate dynamics
        removal_rate_new = self.removal_rate_nominal * (pressure / 1e5) * (flow_rate / 1e-6) * temp_factor
        
        # Quality metric calculation
        roughness_score = max(0, 1 - roughness / self.target_roughness)
        defect_score = max(0, 1 - defect_density / self.target_defect_density)
        rate_score = min(1, removal_rate / self.removal_rate_nominal)
        
        quality_new = (self.quality_weight_roughness * roughness_score + 
                      self.quality_weight_defects * defect_score +
                      self.quality_weight_rate * rate_score)
        
        # Optimization state (simple gradient ascent)
        quality_gradient = quality_new - quality_metric
        opt_state_rate = 0.1 * quality_gradient  # Simple integrator
        
        return np.array([
            roughness_rate,
            defect_rate,
            removal_rate_new,
            quality_new,
            opt_state_rate
        ])
    
    def get_coupling_outputs(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Outputs for coupling to other domains"""
        roughness, defect_density, removal_rate, quality_metric, opt_state = state
        
        return {
            'surface_roughness': np.array([roughness]),
            'defect_density': np.array([defect_density]),
            'removal_rate': np.array([removal_rate]),
            'quality_metric': np.array([quality_metric]),
            'process_heat_generation': np.array([1e-3 * removal_rate / self.removal_rate_nominal])
        }

class MultiRateIntegrator:
    """Enhanced multi-rate time integration with synchronization"""
    
    def __init__(self, config: MultiPhysicsConfig):
        self.config = config
        
        # Time step hierarchy
        self.dt_fast = config.fast_dt
        self.dt_slow = config.slow_dt  
        self.dt_meta = config.meta_dt
        
        # Integration counters
        self.fast_step = 0
        self.slow_step = 0
        self.meta_step = 0
        
        # Buffers for averaging
        self.fast_buffer = []
        self.slow_buffer = []
        
        # Interpolators for coupling
        self.interpolators = {}
        
    def integrate_fast_domain(self,
                            domain: PhysicsDomain,
                            inputs: Dict[str, np.ndarray],
                            coupling_states: Dict[str, np.ndarray] = None,
                            num_steps: int = 1) -> List[DomainState]:
        """Integrate fast domain with fixed time step"""
        
        states_history = []
        current_state = domain.current_state.states.copy()
        current_time = domain.current_state.time
        
        for _ in range(num_steps):
            # Compute derivatives
            try:
                derivatives = domain.dynamics(current_state, current_time, inputs, coupling_states)
                
                # Adaptive time stepping if enabled
                if self.config.adaptive_timestepping:
                    dt_adaptive = self._compute_adaptive_timestep(derivatives, current_state)
                    dt = min(dt_adaptive, self.dt_fast)
                else:
                    dt = self.dt_fast
                
                # RK4 integration
                k1 = dt * derivatives
                k2 = dt * domain.dynamics(current_state + k1/2, current_time + dt/2, inputs, coupling_states)
                k3 = dt * domain.dynamics(current_state + k2/2, current_time + dt/2, inputs, coupling_states)
                k4 = dt * domain.dynamics(current_state + k3, current_time + dt, inputs, coupling_states)
                
                current_state = current_state + (k1 + 2*k2 + 2*k3 + k4) / 6
                current_time += dt
                
                # Store state
                state = DomainState(
                    states=current_state.copy(),
                    time=current_time,
                    domain_name=domain.domain_name,
                    state_names=domain.current_state.state_names
                )
                states_history.append(state)
                
            except Exception as e:
                warnings.warn(f"Fast integration error: {e}")
                # Use previous state
                states_history.append(DomainState(
                    states=current_state.copy(),
                    time=current_time,
                    domain_name=domain.domain_name
                ))
        
        # Update domain state
        domain.current_state = states_history[-1]
        return states_history
    
    def integrate_slow_domain(self,
                            domain: PhysicsDomain,
                            inputs: Dict[str, np.ndarray],
                            fast_averaged_states: Dict[str, np.ndarray] = None) -> DomainState:
        """Integrate slow domain with averaging from fast domain"""
        
        current_state = domain.current_state.states.copy()
        current_time = domain.current_state.time
        
        try:
            # Use averaged fast states for coupling
            coupling_states = fast_averaged_states if fast_averaged_states else {}
            
            # Compute derivatives
            derivatives = domain.dynamics(current_state, current_time, inputs, coupling_states)
            
            # Adaptive time stepping
            if self.config.adaptive_timestepping:
                dt_adaptive = self._compute_adaptive_timestep(derivatives, current_state)
                dt = min(dt_adaptive, self.dt_slow)
            else:
                dt = self.dt_slow
            
            # RK4 integration
            k1 = dt * derivatives
            k2 = dt * domain.dynamics(current_state + k1/2, current_time + dt/2, inputs, coupling_states)
            k3 = dt * domain.dynamics(current_state + k2/2, current_time + dt/2, inputs, coupling_states)
            k4 = dt * domain.dynamics(current_state + k3, current_time + dt, inputs, coupling_states)
            
            new_state = current_state + (k1 + 2*k2 + 2*k3 + k4) / 6
            new_time = current_time + dt
            
            # Update domain state
            domain.current_state = DomainState(
                states=new_state,
                time=new_time,
                domain_name=domain.domain_name,
                state_names=domain.current_state.state_names
            )
            
        except Exception as e:
            warnings.warn(f"Slow integration error: {e}")
            # Keep current state
            domain.current_state.time += self.dt_slow
        
        return domain.current_state
    
    def integrate_meta_domain(self,
                            domain: PhysicsDomain,
                            inputs: Dict[str, np.ndarray],
                            slow_averaged_states: Dict[str, np.ndarray] = None) -> DomainState:
        """Integrate meta domain with slow domain averaging"""
        
        current_state = domain.current_state.states.copy()
        current_time = domain.current_state.time
        
        try:
            # Use averaged slow states for coupling
            coupling_states = slow_averaged_states if slow_averaged_states else {}
            
            # Compute derivatives
            derivatives = domain.dynamics(current_state, current_time, inputs, coupling_states)
            
            # Simple Euler integration for meta domain (large time steps)
            dt = self.dt_meta
            new_state = current_state + dt * derivatives
            new_time = current_time + dt
            
            # Update domain state
            domain.current_state = DomainState(
                states=new_state,
                time=new_time,
                domain_name=domain.domain_name,
                state_names=domain.current_state.state_names
            )
            
        except Exception as e:
            warnings.warn(f"Meta integration error: {e}")
            domain.current_state.time += self.dt_meta
        
        return domain.current_state
    
    def _compute_adaptive_timestep(self, derivatives: np.ndarray, states: np.ndarray) -> float:
        """Compute adaptive time step based on error estimation"""
        
        # Avoid division by zero
        state_scale = np.maximum(np.abs(states), 1e-10)
        derivative_scale = np.maximum(np.abs(derivatives), 1e-10)
        
        # Estimate characteristic time scales
        time_scales = state_scale / derivative_scale
        
        # Use minimum time scale with safety factor
        min_time_scale = np.min(time_scales[np.isfinite(time_scales)])
        adaptive_dt = 0.1 * min_time_scale
        
        # Apply limits
        adaptive_dt = np.clip(adaptive_dt, 
                             self.dt_fast / self.config.max_step_factor,
                             self.dt_fast * self.config.max_step_factor)
        
        return adaptive_dt
    
    def average_states(self, states_history: List[DomainState]) -> np.ndarray:
        """Average states over time window"""
        if not states_history:
            return np.zeros(1)
        
        states_array = np.array([state.states for state in states_history])
        return np.mean(states_array, axis=0)

class CrossDomainCorrelationTracker:
    """Track and propagate correlations between physics domains"""
    
    def __init__(self, config: MultiPhysicsConfig):
        self.config = config
        
        # Correlation matrices
        self.correlation_history = []
        self.cross_covariance_matrices = {}
        
        # Domain tracking
        self.domain_names = []
        self.domain_dimensions = {}
        
    def register_domain(self, domain_name: str, state_dimension: int):
        """Register a physics domain for correlation tracking"""
        if domain_name not in self.domain_names:
            self.domain_names.append(domain_name)
            self.domain_dimensions[domain_name] = state_dimension
    
    def update_correlations(self, domain_states: Dict[str, DomainState]):
        """Update cross-domain correlation matrices"""
        
        # Collect current states
        current_states = {}
        for domain_name, domain_state in domain_states.items():
            current_states[domain_name] = domain_state.states
        
        # Add to history
        self.correlation_history.append(current_states)
        
        # Maintain window size
        if len(self.correlation_history) > self.config.correlation_window:
            self.correlation_history.pop(0)
        
        # Compute cross-correlations if sufficient history
        if len(self.correlation_history) >= 10:
            self._compute_cross_covariance_matrices()
    
    def _compute_cross_covariance_matrices(self):
        """Compute cross-covariance matrices between domains"""
        
        # Organize data by domain
        domain_data = {}
        for domain_name in self.domain_names:
            domain_data[domain_name] = []
        
        for state_snapshot in self.correlation_history:
            for domain_name in self.domain_names:
                if domain_name in state_snapshot:
                    domain_data[domain_name].append(state_snapshot[domain_name])
        
        # Convert to arrays and compute cross-covariances
        for i, domain_i in enumerate(self.domain_names):
            for j, domain_j in enumerate(self.domain_names):
                if i <= j and domain_i in domain_data and domain_j in domain_data:
                    
                    data_i = np.array(domain_data[domain_i])
                    data_j = np.array(domain_data[domain_j])
                    
                    if len(data_i) > 1 and len(data_j) > 1:
                        try:
                            # Center the data
                            data_i_centered = data_i - np.mean(data_i, axis=0)
                            data_j_centered = data_j - np.mean(data_j, axis=0)
                            
                            # Compute cross-covariance
                            cross_cov = np.cov(data_i_centered.T, data_j_centered.T)
                            
                            # Extract relevant block
                            dim_i = self.domain_dimensions[domain_i]
                            dim_j = self.domain_dimensions[domain_j]
                            
                            if i == j:  # Auto-covariance
                                self.cross_covariance_matrices[f"{domain_i}_{domain_j}"] = cross_cov[:dim_i, :dim_i]
                            else:  # Cross-covariance
                                self.cross_covariance_matrices[f"{domain_i}_{domain_j}"] = cross_cov[:dim_i, dim_i:dim_i+dim_j]
                                self.cross_covariance_matrices[f"{domain_j}_{domain_i}"] = cross_cov[dim_i:dim_i+dim_j, :dim_i]
                        
                        except Exception as e:
                            warnings.warn(f"Cross-covariance computation error for {domain_i}-{domain_j}: {e}")
    
    def get_correlation_coefficient(self, domain_i: str, state_i: int, domain_j: str, state_j: int) -> float:
        """Get correlation coefficient between specific states in different domains"""
        
        key = f"{domain_i}_{domain_j}"
        reverse_key = f"{domain_j}_{domain_i}"
        
        if key in self.cross_covariance_matrices:
            cross_cov = self.cross_covariance_matrices[key]
        elif reverse_key in self.cross_covariance_matrices:
            cross_cov = self.cross_covariance_matrices[reverse_key].T
        else:
            return 0.0  # No correlation data
        
        try:
            # Get variances for normalization
            var_i_key = f"{domain_i}_{domain_i}"
            var_j_key = f"{domain_j}_{domain_j}"
            
            if var_i_key in self.cross_covariance_matrices and var_j_key in self.cross_covariance_matrices:
                var_i = self.cross_covariance_matrices[var_i_key][state_i, state_i]
                var_j = self.cross_covariance_matrices[var_j_key][state_j, state_j]
                
                if var_i > 0 and var_j > 0:
                    correlation = cross_cov[state_i, state_j] / np.sqrt(var_i * var_j)
                    return np.clip(correlation, -1.0, 1.0)
            
        except (IndexError, KeyError):
            pass
        
        return 0.0
    
    def propagate_uncertainty_correlations(self, 
                                         domain_states: Dict[str, DomainState],
                                         prediction_horizon: int = 10) -> Dict[str, np.ndarray]:
        """
        Propagate uncertainty with cross-domain correlations
        
        Returns uncertainty bounds considering correlations
        """
        uncertainty_bounds = {}
        
        for domain_name, domain_state in domain_states.items():
            state_dim = len(domain_state.states)
            
            # Initialize uncertainty
            if f"{domain_name}_{domain_name}" in self.cross_covariance_matrices:
                auto_cov = self.cross_covariance_matrices[f"{domain_name}_{domain_name}"]
                base_uncertainty = np.sqrt(np.diag(auto_cov))
            else:
                base_uncertainty = np.ones(state_dim) * 1e-6  # Default small uncertainty
            
            # Account for cross-domain correlations
            correlation_amplification = np.ones(state_dim)
            
            for other_domain in self.domain_names:
                if other_domain != domain_name:
                    for i in range(state_dim):
                        for j in range(self.domain_dimensions.get(other_domain, 1)):
                            corr = abs(self.get_correlation_coefficient(domain_name, i, other_domain, j))
                            if corr > self.config.correlation_threshold:
                                # Amplify uncertainty due to correlation
                                correlation_amplification[i] *= (1 + corr)
            
            # Propagate over prediction horizon (simplified)
            horizon_amplification = np.sqrt(1 + 0.1 * prediction_horizon)  # Simple growth model
            
            final_uncertainty = base_uncertainty * correlation_amplification * horizon_amplification
            uncertainty_bounds[domain_name] = final_uncertainty
        
        return uncertainty_bounds

class MultiPhysicsDigitalTwin:
    """Complete multi-physics digital twin with enhanced integration"""
    
    def __init__(self, config: MultiPhysicsConfig):
        self.config = config
        
        # Initialize domains
        self.actuator_domain = ActuatorDynamicsDomain()
        self.thermal_domain = ThermalDynamicsDomain()
        self.process_domain = ProcessOptimizationDomain()
        
        self.domains = {
            'actuator': self.actuator_domain,
            'thermal': self.thermal_domain,
            'process': self.process_domain
        }
        
        # Initialize integrator and correlation tracker
        self.integrator = MultiRateIntegrator(config)
        self.correlation_tracker = CrossDomainCorrelationTracker(config)
        
        # Register domains for correlation tracking
        for domain_name, domain in self.domains.items():
            self.correlation_tracker.register_domain(domain_name, domain.state_dim)
        
        # Simulation state
        self.simulation_time = 0.0
        self.step_count = 0
        
        # History for analysis
        self.state_history = {domain_name: [] for domain_name in self.domains.keys()}
        
    def initialize_states(self, initial_conditions: Dict[str, np.ndarray]):
        """Initialize domain states"""
        for domain_name, initial_state in initial_conditions.items():
            if domain_name in self.domains:
                self.domains[domain_name].current_state.states = initial_state.copy()
                self.domains[domain_name].current_state.time = 0.0
    
    def simulation_step(self, 
                       inputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, DomainState]:
        """
        Execute one multi-rate simulation step
        
        Args:
            inputs: Nested dict {domain_name: {input_name: value}}
            
        Returns:
            Current domain states
        """
        current_states = {}
        
        # Step 1: Fast domain integration (multiple sub-steps)
        fast_inputs = inputs.get('actuator', {})
        
        # Get coupling from slow domains (interpolated if needed)
        slow_coupling = {}
        if self.thermal_domain.current_state.time > 0:
            slow_coupling['thermal'] = self.thermal_domain.get_coupling_outputs(
                self.thermal_domain.current_state.states
            )['substrate_temperature']
        
        # Integrate fast domain
        fast_states_history = self.integrator.integrate_fast_domain(
            self.actuator_domain,
            fast_inputs,
            slow_coupling,
            num_steps=self.config.fast_slow_ratio
        )
        
        # Average fast states for coupling to slow domain
        fast_averaged = self.integrator.average_states(fast_states_history)
        fast_coupling = {'actuator': fast_averaged[:3]}  # Position, velocity, force
        
        # Step 2: Slow domain integration
        slow_inputs = inputs.get('thermal', {})
        
        # Get coupling from meta domain
        meta_coupling = {}
        if self.process_domain.current_state.time > 0:
            meta_coupling['process'] = self.process_domain.get_coupling_outputs(
                self.process_domain.current_state.states
            )['process_heat_generation']
        
        # Combine couplings
        thermal_coupling = {**fast_coupling, **meta_coupling}
        
        thermal_state = self.integrator.integrate_slow_domain(
            self.thermal_domain,
            slow_inputs,
            thermal_coupling
        )
        
        # Step 3: Meta domain integration (every N slow steps)
        if self.step_count % self.config.slow_meta_ratio == 0:
            meta_inputs = inputs.get('process', {})
            
            # Combine fast and slow coupling for meta domain
            meta_coupling_combined = {
                'actuator': fast_averaged[2:3],  # Force
                'thermal': thermal_state.states[:1]  # Substrate temperature
            }
            
            process_state = self.integrator.integrate_meta_domain(
                self.process_domain,
                meta_inputs,
                meta_coupling_combined
            )
        
        # Collect current states
        for domain_name, domain in self.domains.items():
            current_states[domain_name] = domain.current_state
            self.state_history[domain_name].append(domain.current_state)
        
        # Update correlations
        self.correlation_tracker.update_correlations(current_states)
        
        # Update simulation state
        self.simulation_time = max(domain.current_state.time for domain in self.domains.values())
        self.step_count += 1
        
        return current_states
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state with correlations"""
        
        current_states = {name: domain.current_state for name, domain in self.domains.items()}
        
        # Get uncertainty bounds with correlations
        uncertainty_bounds = self.correlation_tracker.propagate_uncertainty_correlations(current_states)
        
        # Compute system-level metrics
        system_metrics = self._compute_system_metrics(current_states)
        
        return {
            'domain_states': current_states,
            'uncertainty_bounds': uncertainty_bounds,
            'system_metrics': system_metrics,
            'simulation_time': self.simulation_time,
            'step_count': self.step_count
        }
    
    def _compute_system_metrics(self, states: Dict[str, DomainState]) -> Dict[str, float]:
        """Compute system-level performance metrics"""
        
        metrics = {}
        
        # Actuator performance
        if 'actuator' in states:
            actuator_state = states['actuator'].states
            metrics['position_error'] = abs(actuator_state[0])  # Position deviation
            metrics['control_effort'] = abs(actuator_state[4])  # Current magnitude
            
        # Thermal performance
        if 'thermal' in states:
            thermal_state = states['thermal'].states
            metrics['temperature_stability'] = abs(thermal_state[0] - 300.0)  # Deviation from 300K
            metrics['thermal_stress'] = abs(thermal_state[3])
            
        # Process performance
        if 'process' in states:
            process_state = states['process'].states
            metrics['surface_roughness'] = process_state[0]
            metrics['defect_density'] = process_state[1] 
            metrics['quality_metric'] = process_state[3]
        
        # Cross-domain correlations
        actuator_thermal_corr = self.correlation_tracker.get_correlation_coefficient(
            'actuator', 0, 'thermal', 0  # Position vs temperature
        )
        thermal_process_corr = self.correlation_tracker.get_correlation_coefficient(
            'thermal', 0, 'process', 0  # Temperature vs roughness
        )
        
        metrics['actuator_thermal_correlation'] = actuator_thermal_corr
        metrics['thermal_process_correlation'] = thermal_process_corr
        
        return metrics

# Example usage and testing
if __name__ == "__main__":
    print("🌐 Testing Multi-Physics Digital Twin")
    print("=" * 50)
    
    # Configuration
    config = MultiPhysicsConfig(
        fast_dt=1e-4,    # 10 kHz
        slow_dt=1e-2,    # 100 Hz  
        meta_dt=1.0,     # 1 Hz
        correlation_window=100
    )
    
    # Initialize digital twin
    digital_twin = MultiPhysicsDigitalTwin(config)
    
    # Initial conditions
    initial_conditions = {
        'actuator': np.array([0, 0, 0, 0, 0, 300]),  # [pos, vel, force, voltage, current, temp]
        'thermal': np.array([300, 295, 0, 0]),       # [T_substrate, T_coolant, heat_flux, stress]
        'process': np.array([0.15e-9, 0.008, 1e-9, 0.5, 0])  # [roughness, defects, rate, quality, opt]
    }
    
    digital_twin.initialize_states(initial_conditions)
    
    print("🔧 Running Multi-Physics Simulation...")
    
    # Simulation loop
    for step in range(20):
        
        # Define inputs for each domain
        inputs = {
            'actuator': {
                'voltage': np.array([0.1 * np.sin(2 * np.pi * step * 0.1)]),  # Sinusoidal voltage
                'external_force': np.array([1e-9])  # Small external force
            },
            'thermal': {
                'coolant_flow': np.array([2e-6]),  # Coolant flow rate
                'ambient_temperature': np.array([295])  # Ambient temperature
            },
            'process': {
                'pressure': np.array([1.2e5]),  # Process pressure
                'flow_rate': np.array([1.5e-6])  # Process flow rate
            }
        }
        
        # Execute simulation step
        states = digital_twin.simulation_step(inputs)
        
        # Get system state with correlations
        system_state = digital_twin.get_system_state()
        
        if step % 5 == 0:  # Print every 5th step
            print(f"\nStep {step}, Time: {system_state['simulation_time']:.4f}s")
            
            # Actuator state
            actuator = states['actuator'].states
            print(f"  Actuator: pos={actuator[0]*1e9:.1f}nm, temp={actuator[5]:.1f}K")
            
            # Thermal state  
            thermal = states['thermal'].states
            print(f"  Thermal: T_sub={thermal[0]:.1f}K, stress={thermal[3]/1e6:.1f}MPa")
            
            # Process state
            process = states['process'].states
            print(f"  Process: roughness={process[0]*1e9:.2f}nm, defects={process[1]:.4f}/μm²")
            
            # Correlations
            metrics = system_state['system_metrics']
            print(f"  Correlations: A-T={metrics.get('actuator_thermal_correlation', 0):.3f}, T-P={metrics.get('thermal_process_correlation', 0):.3f}")
    
    print("\n✅ Multi-Physics Digital Twin implementation complete!")
    print(f"Final quality metric: {system_state['system_metrics'].get('quality_metric', 0):.3f}")
    print(f"Final surface roughness: {system_state['system_metrics'].get('surface_roughness', 0)*1e9:.2f} nm")
    print(f"Final defect density: {system_state['system_metrics'].get('defect_density', 0):.4f} defects/μm²")
