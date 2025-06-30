"""
Additional High-Severity UQ Validation Methods
==============================================

Extension of critical UQ resolution framework with additional
high-severity validation implementations.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from critical_uq_resolution import UQValidationResult, CriticalUQValidator

class ExtendedUQValidator(CriticalUQValidator):
    """Extended UQ validator with additional high-severity validations"""
    
    def _validate_thermal_expansion_correlation(self) -> UQValidationResult:
        """
        HIGH: Validate thermal expansion correlation model accuracy
        Severity: 80
        """
        start_time = time.time()
        
        # Material properties for testing
        materials = {
            'zerodur': {'alpha': 0.02e-6, 'uncertainty': 0.01e-6},  # /K
            'invar': {'alpha': 1.2e-6, 'uncertainty': 0.1e-6},
            'silicon': {'alpha': 2.6e-6, 'uncertainty': 0.2e-6},
            'aluminum': {'alpha': 23.1e-6, 'uncertainty': 0.5e-6}
        }
        
        # Temperature ranges and gradients
        temperatures = np.linspace(250, 350, 100)  # 250K to 350K
        temp_gradients = np.logspace(-2, 2, 50)    # 0.01 to 100 K/m
        
        correlation_accuracies = []
        coupling_validations = []
        
        for material_name, props in materials.items():
            alpha = props['alpha']
            alpha_uncertainty = props['uncertainty']
            
            for temp_gradient in temp_gradients:
                # Calculate thermal expansion coupling
                mechanical_strain = alpha * temp_gradient * 0.1  # 0.1m characteristic length
                mechanical_stress = mechanical_strain * 200e9    # 200 GPa Young's modulus
                
                # Thermal-mechanical coupling correlation
                # Based on coupled physics: σ = E(ε - αΔT)
                coupling_strength = abs(alpha * temp_gradient * 200e9) / (1e6)  # Normalized
                
                # Validate against empirical correlation model r = 0.45 ± 0.1
                expected_correlation = 0.45
                model_correlation = np.tanh(coupling_strength)  # Saturation model
                
                correlation_error = abs(model_correlation - expected_correlation)
                correlation_accuracies.append(correlation_error)
                
                # Validate uncertainty propagation in coupling
                thermal_uncertainty = alpha_uncertainty * temp_gradient * 0.1
                mechanical_uncertainty = thermal_uncertainty * 200e9
                
                # Check if uncertainties are properly correlated
                total_uncertainty = np.sqrt(
                    mechanical_uncertainty**2 + 
                    (0.1 * mechanical_stress)**2  # 10% mechanical model uncertainty
                )
                
                uncertainty_ratio = thermal_uncertainty / total_uncertainty
                coupling_validations.append(uncertainty_ratio > 0.1)  # Thermal should contribute >10%
        
        # Cross-material correlation validation
        material_correlations = []
        material_pairs = [('zerodur', 'invar'), ('silicon', 'aluminum'), ('zerodur', 'silicon')]
        
        for mat1, mat2 in material_pairs:
            alpha1 = materials[mat1]['alpha']
            alpha2 = materials[mat2]['alpha']
            
            # Expected correlation based on thermal expansion similarity
            expansion_ratio = min(alpha1, alpha2) / max(alpha1, alpha2)
            expected_correlation = expansion_ratio * 0.8  # 80% of ratio
            
            # Simulate correlated thermal response
            temp_variations = np.random.normal(0, 5, 1000)  # 5K temperature variations
            
            expansion1 = alpha1 * temp_variations
            expansion2 = alpha2 * temp_variations
            
            observed_correlation = np.corrcoef(expansion1, expansion2)[0, 1]
            correlation_error = abs(observed_correlation - expected_correlation)
            material_correlations.append(correlation_error)
        
        # Validation criteria
        max_correlation_error = 0.15  # 15% maximum error in correlation prediction
        min_coupling_fraction = 0.7   # 70% of cases should show significant thermal coupling
        max_material_correlation_error = 0.2  # 20% error in cross-material correlations
        
        avg_correlation_error = np.mean(correlation_accuracies)
        coupling_fraction = np.mean(coupling_validations)
        avg_material_error = np.mean(material_correlations)
        
        passed = (
            avg_correlation_error <= max_correlation_error and
            coupling_fraction >= min_coupling_fraction and
            avg_material_error <= max_material_correlation_error
        )
        
        metrics = {
            'average_correlation_error': avg_correlation_error,
            'thermal_coupling_fraction': coupling_fraction,
            'material_correlation_error': avg_material_error,
            'materials_tested': len(materials),
            'temperature_range_k': 100
        }
        
        recommendations = []
        if not passed:
            if avg_correlation_error > max_correlation_error:
                recommendations.append("Refine thermal-mechanical coupling model")
            if coupling_fraction < min_coupling_fraction:
                recommendations.append("Include additional thermal coupling mechanisms")
            if avg_material_error > max_material_correlation_error:
                recommendations.append("Improve cross-material correlation predictions")
        
        return UQValidationResult(
            concern_title="Thermal Expansion Correlation Model Accuracy",
            severity=80,
            validation_passed=passed,
            metrics=metrics,
            recommendations=recommendations,
            validation_time=time.time() - start_time
        )
    
    def _validate_multi_rate_control_interaction(self) -> UQValidationResult:
        """
        HIGH: Validate multi-rate control loop interaction UQ
        Severity: 80
        """
        start_time = time.time()
        
        # Define control loop parameters
        control_loops = {
            'fast': {'frequency': 10000, 'time_constant': 1e-4},    # 10 kHz positioning
            'medium': {'frequency': 1000, 'time_constant': 1e-3},   # 1 kHz force control
            'slow': {'frequency': 10, 'time_constant': 0.1}         # 10 Hz thermal control
        }
        
        # Simulation parameters
        simulation_time = 1.0  # 1 second
        dt_base = 1e-5  # 10 μs base time step
        n_steps = int(simulation_time / dt_base)
        
        # Generate reference trajectories for each loop
        time_vector = np.linspace(0, simulation_time, n_steps)
        reference_signals = {}
        control_outputs = {}
        
        for loop_name, params in control_loops.items():
            freq = params['frequency']
            # Multi-frequency reference signal
            ref_signal = (np.sin(2 * np.pi * freq * time_vector / 100) +  # Primary frequency
                         0.3 * np.sin(2 * np.pi * freq * time_vector / 50) +  # Harmonic
                         0.1 * np.random.randn(n_steps))  # Noise
            reference_signals[loop_name] = ref_signal
            
            # Simple PI controller response
            kp = 1.0 / params['time_constant']
            ki = kp / (10 * params['time_constant'])
            
            control_output = np.zeros(n_steps)
            integral_error = 0
            
            for i in range(1, n_steps):
                error = ref_signal[i] - control_output[i-1]  # Simple feedback
                integral_error += error * dt_base
                
                control_output[i] = kp * error + ki * integral_error
                
                # Add saturation
                control_output[i] = np.clip(control_output[i], -10, 10)
            
            control_outputs[loop_name] = control_output
        
        # Analyze inter-loop coupling and uncertainty propagation
        coupling_analyses = []
        stability_margins = []
        
        # Fast-Medium loop interaction
        fast_medium_correlation = np.corrcoef(
            control_outputs['fast'][::10],  # Downsample fast loop
            control_outputs['medium']
        )[0, 1]
        
        # Medium-Slow loop interaction
        medium_slow_correlation = np.corrcoef(
            control_outputs['medium'][::100],  # Downsample medium loop
            control_outputs['slow']
        )[0, 1]
        
        coupling_analyses.extend([fast_medium_correlation, medium_slow_correlation])
        
        # Stability analysis for coupled system
        for loop_name, control_signal in control_outputs.items():
            # Calculate stability margin using Nyquist-like criterion
            signal_fft = np.fft.fft(control_signal)
            frequencies = np.fft.fftfreq(len(control_signal), dt_base)
            
            # Find gain and phase margins
            magnitude = np.abs(signal_fft)
            phase = np.angle(signal_fft)
            
            # Stability margin based on peak-to-average ratio
            stability_margin = np.mean(magnitude) / (np.max(magnitude) + 1e-12)
            stability_margins.append(stability_margin)
        
        # Uncertainty propagation between loops
        uncertainty_propagation = []
        
        for i, (loop1, loop2) in enumerate([('fast', 'medium'), ('medium', 'slow')]):
            # Calculate uncertainty transfer
            output1 = control_outputs[loop1]
            output2 = control_outputs[loop2]
            
            # Subsample to match rates
            if loop1 == 'fast' and loop2 == 'medium':
                output1_sampled = output1[::10]
                output2_sampled = output2
            else:  # medium to slow
                output1_sampled = output1[::100]
                output2_sampled = output2
            
            # Calculate cross-correlation for uncertainty transfer
            cross_corr = np.correlate(output1_sampled, output2_sampled, mode='full')
            max_cross_corr = np.max(np.abs(cross_corr)) / len(output1_sampled)
            uncertainty_propagation.append(max_cross_corr)
        
        # Validation criteria
        max_coupling_strength = 0.3    # Maximum 30% coupling between loops
        min_stability_margin = 0.1     # Minimum 10% stability margin
        max_uncertainty_transfer = 0.5  # Maximum 50% uncertainty transfer
        
        max_coupling = np.max(np.abs(coupling_analyses))
        min_stability = np.min(stability_margins)
        max_transfer = np.max(uncertainty_propagation)
        
        passed = (
            max_coupling <= max_coupling_strength and
            min_stability >= min_stability_margin and
            max_transfer <= max_uncertainty_transfer
        )
        
        metrics = {
            'max_inter_loop_coupling': max_coupling,
            'min_stability_margin': min_stability,
            'max_uncertainty_transfer': max_transfer,
            'control_loops_analyzed': len(control_loops),
            'simulation_time_s': simulation_time
        }
        
        recommendations = []
        if not passed:
            if max_coupling > max_coupling_strength:
                recommendations.append("Implement decoupling compensation in control design")
            if min_stability < min_stability_margin:
                recommendations.append("Increase control loop stability margins")
            if max_transfer > max_uncertainty_transfer:
                recommendations.append("Add uncertainty isolation between control loops")
        
        return UQValidationResult(
            concern_title="Multi-Rate Control Loop Interaction UQ",
            severity=80,
            validation_passed=passed,
            metrics=metrics,
            recommendations=recommendations,
            validation_time=time.time() - start_time
        )
    
    def _validate_robustness_parameter_variations(self) -> UQValidationResult:
        """
        HIGH: Validate robustness under parameter variations
        Severity: 80
        """
        start_time = time.time()
        
        # Define critical system parameters and their nominal values
        nominal_parameters = {
            'casimir_constant': 240,           # Casimir force constant
            'spring_constant': 1e6,           # N/m
            'damping_coefficient': 100,       # N⋅s/m
            'thermal_expansion': 2.6e-6,     # /K
            'young_modulus': 200e9,           # Pa
            'material_conductivity': 148,     # W/m⋅K
            'dielectric_constant': 11.7,      # Si dielectric constant
        }
        
        # Define parameter variation ranges (±50% for robustness testing)
        variation_ranges = {
            param: 0.5 for param in nominal_parameters.keys()
        }
        
        # Performance metrics to evaluate
        performance_metrics = []
        stability_metrics = []
        convergence_metrics = []
        
        # Monte Carlo parameter variation study
        n_variations = 1000
        
        for trial in range(n_variations):
            # Generate parameter variations
            varied_parameters = {}
            for param, nominal in nominal_parameters.items():
                variation_factor = np.random.uniform(
                    1 - variation_ranges[param], 
                    1 + variation_ranges[param]
                )
                varied_parameters[param] = nominal * variation_factor
            
            # Simulate system performance with varied parameters
            performance_result = self._simulate_system_performance(varied_parameters)
            
            performance_metrics.append(performance_result['performance_score'])
            stability_metrics.append(performance_result['stability_score'])
            convergence_metrics.append(performance_result['convergence_time'])
        
        # Statistical analysis of robustness
        performance_stats = {
            'mean': np.mean(performance_metrics),
            'std': np.std(performance_metrics),
            'min': np.min(performance_metrics),
            'percentile_5': np.percentile(performance_metrics, 5),
            'percentile_95': np.percentile(performance_metrics, 95)
        }
        
        stability_stats = {
            'mean': np.mean(stability_metrics),
            'std': np.std(stability_metrics),
            'failure_rate': np.mean(np.array(stability_metrics) < 0.5)
        }
        
        convergence_stats = {
            'mean': np.mean(convergence_metrics),
            'std': np.std(convergence_metrics),
            'timeout_rate': np.mean(np.array(convergence_metrics) > 10.0)  # 10s timeout
        }
        
        # Sensitivity analysis - which parameters affect performance most
        parameter_sensitivities = {}
        
        for param in nominal_parameters.keys():
            # Calculate correlation between parameter variation and performance
            param_variations = []
            corresponding_performance = []
            
            for trial in range(min(100, n_variations)):  # Subset for efficiency
                varied_params = {}
                for p, nominal in nominal_parameters.items():
                    if p == param:
                        # Vary only this parameter
                        variation = np.random.uniform(0.5, 1.5)
                        varied_params[p] = nominal * variation
                        param_variations.append(variation)
                    else:
                        varied_params[p] = nominal
                
                result = self._simulate_system_performance(varied_params)
                corresponding_performance.append(result['performance_score'])
            
            # Calculate sensitivity coefficient
            if len(param_variations) > 1:
                sensitivity = abs(np.corrcoef(param_variations, corresponding_performance)[0, 1])
                parameter_sensitivities[param] = sensitivity
        
        # Validation criteria
        min_performance_5th_percentile = 0.7    # 5th percentile performance > 70%
        max_stability_failure_rate = 0.05       # < 5% stability failures
        max_convergence_timeout_rate = 0.02     # < 2% convergence timeouts
        max_performance_sensitivity = 0.8       # No parameter should cause >80% correlation
        
        performance_5th = performance_stats['percentile_5']
        stability_failure_rate = stability_stats['failure_rate']
        convergence_timeout_rate = convergence_stats['timeout_rate']
        max_sensitivity = max(parameter_sensitivities.values()) if parameter_sensitivities else 0
        
        passed = (
            performance_5th >= min_performance_5th_percentile and
            stability_failure_rate <= max_stability_failure_rate and
            convergence_timeout_rate <= max_convergence_timeout_rate and
            max_sensitivity <= max_performance_sensitivity
        )
        
        metrics = {
            'performance_5th_percentile': performance_5th,
            'stability_failure_rate': stability_failure_rate,
            'convergence_timeout_rate': convergence_timeout_rate,
            'max_parameter_sensitivity': max_sensitivity,
            'parameters_tested': len(nominal_parameters),
            'variation_trials': n_variations
        }
        
        recommendations = []
        if not passed:
            if performance_5th < min_performance_5th_percentile:
                recommendations.append("Improve worst-case performance robustness")
            if stability_failure_rate > max_stability_failure_rate:
                recommendations.append("Add stability safeguards for parameter variations")
            if convergence_timeout_rate > max_convergence_timeout_rate:
                recommendations.append("Optimize convergence algorithms for parameter variations")
            if max_sensitivity > max_performance_sensitivity:
                most_sensitive = max(parameter_sensitivities.items(), key=lambda x: x[1])
                recommendations.append(f"Reduce sensitivity to {most_sensitive[0]} parameter")
        
        return UQValidationResult(
            concern_title="Robustness Testing Under Parameter Variations",
            severity=80,
            validation_passed=passed,
            metrics=metrics,
            recommendations=recommendations,
            validation_time=time.time() - start_time
        )
    
    def _simulate_system_performance(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Simulate system performance with given parameter set
        """
        # Simplified system performance simulation
        
        # Calculate Casimir force with parameter variations
        casimir_force = parameters['casimir_constant'] * 1e-12  # Simplified
        
        # Mechanical response
        spring_force = parameters['spring_constant'] * 1e-9  # 1 nm displacement
        damping_force = parameters['damping_coefficient'] * 1e-6  # 1 μm/s velocity
        
        # Thermal effects
        thermal_expansion = parameters['thermal_expansion'] * 10  # 10K temperature change
        thermal_stress = thermal_expansion * parameters['young_modulus']
        
        # Performance score (0-1, higher is better)
        force_balance = abs(casimir_force - spring_force) / max(casimir_force, spring_force)
        performance_score = np.exp(-force_balance)  # Exponential scoring
        
        # Stability score (0-1, higher is better)
        stability_score = 1.0 / (1.0 + abs(thermal_stress) / 1e6)  # Normalized
        
        # Convergence time (seconds)
        time_constant = parameters['damping_coefficient'] / parameters['spring_constant']
        convergence_time = 5 * time_constant  # 5 time constants to settle
        
        return {
            'performance_score': performance_score,
            'stability_score': stability_score,
            'convergence_time': convergence_time
        }

if __name__ == "__main__":
    # Test the extended validator
    validator = ExtendedUQValidator()
    
    # Test individual validations
    thermal_result = validator._validate_thermal_expansion_correlation()
    control_result = validator._validate_multi_rate_control_interaction()
    robustness_result = validator._validate_robustness_parameter_variations()
    
    print("🔍 EXTENDED UQ VALIDATION RESULTS")
    print("="*50)
    
    results = [thermal_result, control_result, robustness_result]
    
    for result in results:
        status = "✅ PASSED" if result.validation_passed else "❌ FAILED"
        print(f"{result.concern_title}: {status}")
        print(f"  Severity: {result.severity}")
        print(f"  Time: {result.validation_time:.2f}s")
        if result.recommendations:
            print("  Recommendations:")
            for rec in result.recommendations:
                print(f"    • {rec}")
        print()
