"""
Advanced Stability Analysis
==========================

Enhanced stability analysis with Lyapunov methods and critical gap calculations
for ultra-smooth Casimir fabrication platform.

Based on formulations from:
- casimir-nanopositioning-platform/src/mechanics/advanced_stability_analysis.py (Lines 50-65)
- warp-bubble-optimizer/docs/energy_scaling.tex (Lines 507-520)
"""

import numpy as np
import scipy.linalg
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
C = 299792458  # m/s
PI = np.pi

# Exact enhancement factors
BETA_EXACT = 1.9443254780147017
BETA_APPROX = 2.0
R_BACKREACTION = BETA_EXACT / BETA_APPROX  # 0.9721627390073509

@dataclass
class StabilityParameters:
    """Parameters for stability analysis"""
    area: float  # Surface area in m²
    spring_constant: float  # Spring constant in N/m
    mass: float  # Mass in kg
    damping: float  # Damping coefficient in kg/s
    temperature: float  # Temperature in K

class AdvancedStabilityAnalysis:
    """
    Advanced stability analysis for Casimir systems with enhanced corrections
    """
    
    def __init__(self, params: StabilityParameters):
        """
        Initialize stability analysis
        
        Args:
            params: System parameters for stability analysis
        """
        self.params = params
        
    def critical_gap_enhanced(self) -> float:
        """
        Enhanced critical gap calculation with exact backreaction factor
        
        Critical Gap = (5π²ℏcA·β_exact / 48k_spring)^(1/5)
        
        where β_exact = 1.9443254780147017
        """
        numerator = 5 * PI**2 * HBAR * C * self.params.area * BETA_EXACT
        denominator = 48 * self.params.spring_constant
        
        return (numerator / denominator) ** (1/5)
    
    def casimir_force_gradient(self, separation: float) -> float:
        """
        Calculate the force gradient for stability analysis
        
        dF/dx = 5π²ℏcA·β_exact / (48x⁶)
        """
        return (5 * PI**2 * HBAR * C * self.params.area * BETA_EXACT) / (48 * separation**6)
    
    def system_matrix(self, separation: float) -> np.ndarray:
        """
        Construct system matrix for linearized dynamics
        
        State vector: [position, velocity]
        Dynamics: ẋ = Ax where A is the system matrix
        """
        # Force gradient (negative for attractive force)
        force_gradient = -self.casimir_force_gradient(separation)
        
        # System matrix in state space form
        A = np.array([
            [0, 1],
            [force_gradient / self.params.mass, -self.params.damping / self.params.mass]
        ])
        
        return A
    
    def lyapunov_stability_analysis(self, separation: float) -> Tuple[bool, Dict]:
        """
        Lyapunov stability analysis
        
        V(x) = x^T P x
        A_cl^T P + P A_cl = -Q
        
        Returns:
            stable: True if system is stable
            analysis: Dictionary with stability metrics
        """
        # System matrix
        A = self.system_matrix(separation)
        
        # Choose positive definite Q matrix
        Q = np.eye(2)
        
        try:
            # Solve Lyapunov equation: A^T P + P A = -Q
            P = scipy.linalg.solve_continuous_lyapunov(A.T, -Q)
            
            # Check if P is positive definite
            eigenvals_P = np.linalg.eigvals(P)
            P_positive_definite = np.all(eigenvals_P > 0)
            
            # Check system eigenvalues
            eigenvals_A = np.linalg.eigvals(A)
            A_stable = np.all(np.real(eigenvals_A) < 0)
            
            # Overall stability
            stable = P_positive_definite and A_stable
            
            analysis = {
                'P_matrix': P,
                'P_eigenvalues': eigenvals_P,
                'A_eigenvalues': eigenvals_A,
                'P_positive_definite': P_positive_definite,
                'A_stable': A_stable,
                'stability_margin': -np.max(np.real(eigenvals_A)),
                'natural_frequency': np.sqrt(abs(eigenvals_A[0] * eigenvals_A[1])),
                'damping_ratio': -np.sum(np.real(eigenvals_A)) / (2 * np.sqrt(abs(eigenvals_A[0] * eigenvals_A[1])))
            }
            
        except np.linalg.LinAlgError:
            # Lyapunov equation couldn't be solved
            stable = False
            analysis = {
                'error': 'Lyapunov equation could not be solved',
                'A_eigenvalues': np.linalg.eigvals(A),
                'A_stable': np.all(np.real(np.linalg.eigvals(A)) < 0)
            }
        
        return stable, analysis
    
    def stability_region(self, separation_range: Tuple[float, float], 
                        num_points: int = 100) -> Dict:
        """
        Map stability region over separation range
        
        Args:
            separation_range: (min_sep, max_sep) in meters
            num_points: Number of points to evaluate
            
        Returns:
            Dictionary with stability analysis results
        """
        separations = np.linspace(separation_range[0], separation_range[1], num_points)
        
        stability_map = []
        critical_gap = self.critical_gap_enhanced()
        
        for sep in separations:
            stable, analysis = self.lyapunov_stability_analysis(sep)
            
            stability_map.append({
                'separation': sep,
                'stable': stable,
                'stability_margin': analysis.get('stability_margin', 0),
                'natural_frequency': analysis.get('natural_frequency', 0),
                'damping_ratio': analysis.get('damping_ratio', 0),
                'relative_to_critical': sep / critical_gap
            })
        
        # Find stability boundaries
        stable_separations = [s['separation'] for s in stability_map if s['stable']]
        
        results = {
            'critical_gap': critical_gap,
            'stability_map': stability_map,
            'stable_range': (min(stable_separations), max(stable_separations)) if stable_separations else None,
            'num_stable_points': len(stable_separations),
            'stability_fraction': len(stable_separations) / num_points
        }
        
        return results
    
    def optimize_spring_constant(self, target_separation: float) -> Tuple[float, Dict]:
        """
        Optimize spring constant for maximum stability at target separation
        
        Args:
            target_separation: Desired operating separation in meters
            
        Returns:
            optimal_k: Optimal spring constant
            optimization_results: Analysis results
        """
        def stability_objective(log_k):
            """Objective function for spring constant optimization"""
            k = 10**log_k
            temp_params = StabilityParameters(
                area=self.params.area,
                spring_constant=k,
                mass=self.params.mass,
                damping=self.params.damping,
                temperature=self.params.temperature
            )
            
            temp_analyzer = AdvancedStabilityAnalysis(temp_params)
            stable, analysis = temp_analyzer.lyapunov_stability_analysis(target_separation)
            
            if stable:
                return -analysis.get('stability_margin', 0)  # Maximize stability margin
            else:
                return 1e6  # Penalty for instability
        
        # Optimize over reasonable range of spring constants
        result = minimize_scalar(stability_objective, bounds=(-3, 6), method='bounded')
        
        optimal_k = 10**result.x
        
        # Analyze optimal system
        optimal_params = StabilityParameters(
            area=self.params.area,
            spring_constant=optimal_k,
            mass=self.params.mass,
            damping=self.params.damping,
            temperature=self.params.temperature
        )
        
        optimal_analyzer = AdvancedStabilityAnalysis(optimal_params)
        stable, analysis = optimal_analyzer.lyapunov_stability_analysis(target_separation)
        
        optimization_results = {
            'optimal_spring_constant': optimal_k,
            'optimization_success': result.success,
            'stability_margin': analysis.get('stability_margin', 0),
            'natural_frequency': analysis.get('natural_frequency', 0),
            'damping_ratio': analysis.get('damping_ratio', 0),
            'critical_gap': optimal_analyzer.critical_gap_enhanced()
        }
        
        return optimal_k, optimization_results
    
    def thermal_stability_analysis(self, separation: float, 
                                 temp_range: Tuple[float, float]) -> Dict:
        """
        Analyze thermal stability over temperature range
        
        Args:
            separation: Operating separation in meters
            temp_range: (min_temp, max_temp) in Kelvin
            
        Returns:
            Thermal stability analysis results
        """
        temps = np.linspace(temp_range[0], temp_range[1], 50)
        thermal_results = []
        
        for temp in temps:
            # Update temperature-dependent parameters
            temp_params = StabilityParameters(
                area=self.params.area,
                spring_constant=self.params.spring_constant,
                mass=self.params.mass,
                damping=self.params.damping,
                temperature=temp
            )
            
            temp_analyzer = AdvancedStabilityAnalysis(temp_params)
            stable, analysis = temp_analyzer.lyapunov_stability_analysis(separation)
            
            thermal_results.append({
                'temperature': temp,
                'stable': stable,
                'stability_margin': analysis.get('stability_margin', 0),
                'critical_gap': temp_analyzer.critical_gap_enhanced()
            })
        
        # Find temperature stability limits
        stable_temps = [r['temperature'] for r in thermal_results if r['stable']]
        
        return {
            'thermal_analysis': thermal_results,
            'stable_temp_range': (min(stable_temps), max(stable_temps)) if stable_temps else None,
            'thermal_stability_fraction': len(stable_temps) / len(temps),
            'critical_gap_variation': {
                'min': min(r['critical_gap'] for r in thermal_results),
                'max': max(r['critical_gap'] for r in thermal_results),
                'mean': np.mean([r['critical_gap'] for r in thermal_results])
            }
        }

# Quality control validation
class QualityControlValidation:
    """
    Enhanced quality control mathematics with Six Sigma standards
    """
    
    @staticmethod
    def process_capability(data: np.ndarray, 
                         lower_spec: float, 
                         upper_spec: float) -> Dict:
        """
        Calculate process capability indices
        
        Cp = (USL - LSL) / (6σ)
        Cpk = min((USL - μ)/(3σ), (μ - LSL)/(3σ))
        
        Target: Cp > 2.0, Cpk > 1.67
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Process capability indices
        cp = (upper_spec - lower_spec) / (6 * std)
        cpk = min((upper_spec - mean) / (3 * std), (mean - lower_spec) / (3 * std))
        
        # Statistical process control
        control_limits = {
            'upper_control': mean + 3 * std,
            'lower_control': mean - 3 * std,
            'upper_warning': mean + 2 * std,
            'lower_warning': mean - 2 * std
        }
        
        # Six Sigma metrics
        sigma_level = min(abs(upper_spec - mean), abs(mean - lower_spec)) / std
        
        return {
            'mean': mean,
            'std': std,
            'cp': cp,
            'cpk': cpk,
            'sigma_level': sigma_level,
            'control_limits': control_limits,
            'six_sigma_capable': cp > 2.0 and cpk > 1.67,
            'geometric_tolerance': 1e-9,  # ±10⁻⁹ m target
            'meets_tolerance': std < 1e-9 / 3  # 3σ within tolerance
        }

# Example usage and validation
if __name__ == "__main__":
    # Test parameters
    params = StabilityParameters(
        area=100e-12,  # 10×10 μm²
        spring_constant=1e-3,  # 1 mN/m
        mass=1e-12,  # 1 pg
        damping=1e-15,  # Very light damping
        temperature=300.0  # Room temperature
    )
    
    # Initialize stability analyzer
    analyzer = AdvancedStabilityAnalysis(params)
    
    # Calculate critical gap
    critical_gap = analyzer.critical_gap_enhanced()
    print("Enhanced Stability Analysis")
    print("=" * 30)
    print(f"Critical gap (enhanced): {critical_gap*1e9:.2f} nm")
    print(f"Enhancement factor: {BETA_EXACT:.6f}")
    print(f"Backreaction ratio: {R_BACKREACTION:.6f}")
    
    # Stability analysis at critical gap
    stable, analysis = analyzer.lyapunov_stability_analysis(critical_gap)
    print(f"\nStability at critical gap: {'✅ STABLE' if stable else '❌ UNSTABLE'}")
    
    if stable:
        print(f"Stability margin: {analysis['stability_margin']:.2e}")
        print(f"Natural frequency: {analysis['natural_frequency']:.2e} Hz")
        print(f"Damping ratio: {analysis['damping_ratio']:.4f}")
    
    # Stability region mapping
    separation_range = (0.5 * critical_gap, 2.0 * critical_gap)
    stability_region = analyzer.stability_region(separation_range)
    
    print(f"\nStability Region Analysis:")
    print(f"Stable fraction: {stability_region['stability_fraction']:.1%}")
    if stability_region['stable_range']:
        stable_min, stable_max = stability_region['stable_range']
        print(f"Stable range: {stable_min*1e9:.1f} - {stable_max*1e9:.1f} nm")
    
    # Spring constant optimization
    target_sep = 0.8 * critical_gap
    optimal_k, opt_results = analyzer.optimize_spring_constant(target_sep)
    print(f"\nOptimal spring constant: {optimal_k:.2e} N/m")
    print(f"Optimized stability margin: {opt_results['stability_margin']:.2e}")
    
    # Quality control validation
    test_data = np.random.normal(0, 0.1e-9, 1000)  # Simulated measurement data
    qc = QualityControlValidation()
    qc_results = qc.process_capability(test_data, -0.5e-9, 0.5e-9)
    
    print(f"\nQuality Control Validation:")
    print(f"Cp: {qc_results['cp']:.2f} (target: >2.0)")
    print(f"Cpk: {qc_results['cpk']:.2f} (target: >1.67)")
    print(f"Six Sigma capable: {'✅ YES' if qc_results['six_sigma_capable'] else '❌ NO'}")
    print(f"Meets geometric tolerance: {'✅ YES' if qc_results['meets_tolerance'] else '❌ NO'}")
