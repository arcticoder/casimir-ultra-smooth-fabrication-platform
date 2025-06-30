"""
Casimir Ultra-Smooth Fabrication Platform
=========================================

Enhanced mathematical framework for ultra-smooth surface fabrication
with quantum field theory corrections and advanced process control.

Key Components:
- Enhanced Casimir Force Calculations
- Advanced Stability Analysis
- Manufacturing Process Control
- Quality Validation Framework
"""

from .enhanced_casimir_formulations import (
    EnhancedCasimirForce,
    MaterialType,
    MaterialProperties
)

from .advanced_stability_analysis import (
    AdvancedStabilityAnalysis,
    StabilityParameters,
    QualityControlValidation
)

from .manufacturing_process_control import (
    ManufacturingProcessOptimizer,
    CMPParameters,
    IonBeamParameters,
    DefectDensityControl,
    SurfaceRoughnessModel
)

from .ultra_smooth_fabrication_platform import (
    UltraSmoothFabricationPlatform,
    FabricationSpecification,
    ValidationResults
)

__version__ = "1.0.0"
__author__ = "arcticoder"
__description__ = "Ultra-smooth fabrication platform with enhanced mathematical formulations"

# Key constants
HBAR = 1.054571817e-34  # J⋅s
C = 299792458  # m/s
BETA_EXACT = 1.9443254780147017  # Exact enhancement factor

# Performance targets
TARGET_ROUGHNESS = 0.2  # nm RMS
TARGET_DEFECT_DENSITY = 0.01  # μm⁻²
TARGET_ENHANCEMENT = 484  # ×

__all__ = [
    # Main classes
    'EnhancedCasimirForce',
    'AdvancedStabilityAnalysis', 
    'ManufacturingProcessOptimizer',
    'UltraSmoothFabricationPlatform',
    
    # Data classes
    'MaterialType',
    'MaterialProperties',
    'StabilityParameters',
    'CMPParameters',
    'IonBeamParameters',
    'FabricationSpecification',
    'ValidationResults',
    
    # Utility classes
    'QualityControlValidation',
    'DefectDensityControl',
    'SurfaceRoughnessModel',
    
    # Constants
    'HBAR',
    'C',
    'BETA_EXACT',
    'TARGET_ROUGHNESS',
    'TARGET_DEFECT_DENSITY',
    'TARGET_ENHANCEMENT'
]
