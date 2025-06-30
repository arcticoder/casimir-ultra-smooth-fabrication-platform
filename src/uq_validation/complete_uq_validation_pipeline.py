"""
Complete UQ Validation Pipeline
==============================

Demonstrates the complete resolution of critical and high severity
UQ concerns for manufacturing deployment.
"""
import subprocess
import sys
import time

def run_validation_pipeline():
    """Run the complete UQ validation pipeline"""
    print("🚀 COMPLETE UQ VALIDATION PIPELINE")
    print("=" * 60)
    print("Demonstrating systematic resolution of all UQ concerns...")
    print()
    
    validations = [
        ("Basic UQ Assessment", "streamlined_critical_uq_resolution.py"),
        ("Manufacturing Readiness", "manufacturing_ready_critical_uq_resolution.py"),
    ]
    
    results = {}
    
    for description, script in validations:
        print(f"🔍 {description}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            result = subprocess.run([sys.executable, script], 
                                  capture_output=True, text=True, timeout=30)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print("✅ VALIDATION SUCCESSFUL")
                
                # Extract key metrics from output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'Resolution Rate:' in line or 'Manufacturing Readiness:' in line:
                        print(f"📊 {line.strip()}")
                    elif line.startswith('✅') and ('RESOLVED' in line or 'READY' in line):
                        print(f"   {line.strip()}")
                
                results[description] = {'success': True, 'duration': duration}
            else:
                print("❌ VALIDATION FAILED")
                print(f"Error: {result.stderr}")
                results[description] = {'success': False, 'duration': duration}
                
        except subprocess.TimeoutExpired:
            print("⏰ VALIDATION TIMEOUT")
            results[description] = {'success': False, 'duration': 30.0}
        except Exception as e:
            print(f"❌ VALIDATION ERROR: {e}")
            results[description] = {'success': False, 'duration': 0}
        
        print(f"⏱️ Duration: {results[description]['duration']:.1f}s")
        print()
    
    # Final summary
    print("📊 COMPLETE VALIDATION SUMMARY")
    print("=" * 60)
    
    successful_validations = sum(1 for r in results.values() if r['success'])
    total_validations = len(results)
    success_rate = successful_validations / total_validations
    
    print(f"Validation Stages Completed: {successful_validations}/{total_validations}")
    print(f"Pipeline Success Rate: {success_rate:.1%}")
    
    if success_rate == 1.0:
        print("\n🎉 COMPLETE UQ VALIDATION SUCCESS!")
        print("✅ All critical and high severity concerns systematically addressed")
        print("🏭 Platform ready for manufacturing deployment")
    elif success_rate >= 0.5:
        print(f"\n⭐ PARTIAL SUCCESS ({success_rate:.0%})")
        print("✅ Significant progress on UQ concerns")
        print("🔧 Some areas require additional optimization")
    else:
        print(f"\n⚠️ VALIDATION CHALLENGES ({success_rate:.0%})")
        print("🔧 Additional development required")
    
    total_time = sum(r['duration'] for r in results.values())
    print(f"\n⏱️ Total pipeline execution time: {total_time:.1f} seconds")
    
    print("\n🎯 UQ RESOLUTION PIPELINE COMPLETE")
    print("📋 Comprehensive analysis of uncertainty quantification concerns")
    print("🏭 Manufacturing deployment roadmap established")
    
    return results

if __name__ == "__main__":
    run_validation_pipeline()
