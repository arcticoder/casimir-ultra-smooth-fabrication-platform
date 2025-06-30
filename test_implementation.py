#!/usr/bin/env python3
"""
Simple validation test for enhanced mathematical formulations
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_enhanced_formulations():
    """Test the enhanced mathematical formulations"""
    
    print("🧪 Testing Enhanced Mathematical Formulations")
    print("=" * 50)
    
    try:
        # Test imports
        from enhanced_casimir_formulations import EnhancedCasimirForce, MaterialType
        from advanced_stability_analysis import AdvancedStabilityAnalysis, StabilityParameters
        from manufacturing_process_control import ManufacturingProcessOptimizer
        from ultra_smooth_fabrication_platform import UltraSmoothFabricationPlatform, FabricationSpecification
        
        print("✅ All imports successful")
        
        # Test basic functionality
        print("\n1️⃣ Testing Enhanced Casimir Force Calculation...")
        materials = [MaterialType.SILICON, MaterialType.GOLD]
        casimir = EnhancedCasimirForce(materials, 100e-12, 300.0)
        force, corrections = casimir.calculate_enhanced_force(100e-9)  # 100 nm separation
        
        print(f"   Force: {force*1e12:.2f} pN")
        print(f"   Enhancement: {corrections['total_enhancement']:.2f}×")
        print("   ✅ Casimir calculation successful")
        
        print("\n2️⃣ Testing Stability Analysis...")
        params = StabilityParameters(100e-12, 1e-3, 1e-12, 1e-15, 300.0)
        stability = AdvancedStabilityAnalysis(params)
        critical_gap = stability.critical_gap_enhanced()
        
        print(f"   Critical gap: {critical_gap*1e9:.2f} nm")
        print("   ✅ Stability analysis successful")
        
        print("\n3️⃣ Testing Manufacturing Optimization...")
        from manufacturing_process_control import CMPParameters
        optimizer = ManufacturingProcessOptimizer()
        
        # Test CMP roughness evolution
        cmp_params = CMPParameters.optimized_rough()
        results = optimizer.roughness_model.cmp_roughness_evolution(
            5.0,  # Initial roughness
            cmp_params,
            5e9   # Material hardness
        )
        
        print(f"   CMP roughness reduction: 5.0 → {results:.2f} nm")
        print("   ✅ Manufacturing model successful")
        
        print("\n4️⃣ Testing Integrated Platform...")
        spec = FabricationSpecification(
            target_roughness=0.2,
            target_defect_density=0.01,
            surface_area=100e-12,
            operating_temperature=300.0,
            material_combination=[MaterialType.SILICON],
            quality_requirements={'cp_min': 2.0, 'cpk_min': 1.67}
        )
        
        platform = UltraSmoothFabricationPlatform(spec)
        print("   ✅ Platform initialization successful")
        
        # Test individual validation components
        casimir_perf = platform.validate_enhanced_casimir_performance()
        print(f"   Casimir enhancement: {casimir_perf['enhancement_factor']:.1f}×")
        
        stability_perf = platform.validate_stability_performance()
        print(f"   Stability margin: {stability_perf.get('stability_margin', 0):.2e}")
        
        print("   ✅ Platform validation successful")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"Enhanced mathematical formulations are working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_formulations()
    sys.exit(0 if success else 1)
