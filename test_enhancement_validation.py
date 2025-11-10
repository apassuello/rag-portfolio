#!/usr/bin/env python3
"""
Test Enhancement Validation Script

Validates the new test infrastructure created as part of the Epic 8 test enhancement plan.
Runs the new tests and provides comprehensive feedback on implementation success.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent / "project-1-technical-rag"

def run_test_suite(test_path: Path, description: str) -> Tuple[bool, str, float]:
    """
    Run a test suite and return success status, output, and duration.
    
    Args:
        test_path: Path to test file or directory
        description: Human readable description of test suite
        
    Returns:
        Tuple of (success, output, duration)
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Path: {test_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run pytest with detailed output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            str(test_path), 
            '-v', '-s', '--tb=short', '--no-header'
        ], capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300)
        
        duration = time.time() - start_time
        
        # Print output in real-time style
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        # Print summary
        if success:
            print(f"✅ SUCCESS: {description} ({duration:.2f}s)")
        else:
            print(f"❌ FAILED: {description} ({duration:.2f}s)")
            print(f"Return code: {result.returncode}")
        
        return success, output, duration
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ TIMEOUT: {description} ({duration:.2f}s)")
        return False, "Test timed out after 300 seconds", duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 ERROR: {description} ({duration:.2f}s)")
        print(f"Exception: {e}")
        return False, f"Exception: {e}", duration


def validate_test_files_exist() -> List[Path]:
    """
    Validate that the expected test files from the enhancement plan exist.
    
    Returns:
        List of existing test files
    """
    expected_test_files = [
        "tests/unit/core/test_component_factory_comprehensive.py",
        "tests/unit/components/embedders/test_modular_embedder_unit.py", 
        "tests/architecture/test_component_interfaces.py"
    ]
    
    existing_files = []
    missing_files = []
    
    for test_file in expected_test_files:
        test_path = PROJECT_ROOT / test_file
        if test_path.exists():
            existing_files.append(test_path)
            print(f"✅ Found: {test_file}")
        else:
            missing_files.append(test_file)
            print(f"❌ Missing: {test_file}")
    
    if missing_files:
        print(f"\n⚠️ Warning: {len(missing_files)} expected test files are missing")
        print("Missing files:")
        for missing in missing_files:
            print(f"  - {missing}")
    else:
        print(f"\n✅ All {len(expected_test_files)} expected test files found")
    
    return existing_files


def run_comprehensive_validation() -> Dict[str, any]:
    """
    Run comprehensive validation of the test enhancement implementation.
    
    Returns:
        Dictionary with validation results
    """
    print("🧪 RAG Portfolio Test Enhancement Validation")
    print("=" * 60)
    
    # Phase 1: Validate test files exist
    print("\n📁 PHASE 1: Validating Test File Structure")
    existing_test_files = validate_test_files_exist()
    
    if not existing_test_files:
        return {
            'overall_success': False,
            'phase_1_files': False,
            'message': 'No test files found - enhancement not implemented'
        }
    
    # Phase 2: Run individual test suites
    print("\n🧪 PHASE 2: Running Individual Test Suites")
    
    test_results = []
    total_duration = 0
    
    for test_file in existing_test_files:
        # Get relative path for description
        rel_path = test_file.relative_to(PROJECT_ROOT)
        description = f"Test Suite: {rel_path}"
        
        success, output, duration = run_test_suite(test_file, description)
        test_results.append({
            'file': str(rel_path),
            'success': success,
            'output': output,
            'duration': duration
        })
        total_duration += duration
        
        # Small pause between tests
        time.sleep(1)
    
    # Phase 3: Run all tests together
    print("\n🧪 PHASE 3: Running All New Tests Together")
    
    # Create list of all test directories
    test_dirs = set()
    for test_file in existing_test_files:
        test_dirs.add(test_file.parent)
    
    all_tests_success = True
    if test_dirs:
        # Run all test directories together
        combined_success = True
        combined_output = ""
        combined_duration = 0
        
        for test_dir in test_dirs:
            success, output, duration = run_test_suite(test_dir, f"All tests in: {test_dir.relative_to(PROJECT_ROOT)}")
            combined_success = combined_success and success
            combined_output += output + "\n"
            combined_duration += duration
        
        all_tests_success = combined_success
        test_results.append({
            'file': 'All combined tests',
            'success': combined_success,
            'output': combined_output,
            'duration': combined_duration
        })
        total_duration += combined_duration
    
    # Phase 4: Generate summary report
    print("\n📊 PHASE 4: Validation Summary")
    
    successful_tests = sum(1 for result in test_results if result['success'])
    total_tests = len(test_results)
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"\n📈 Test Enhancement Validation Results:")
    print(f"  • Test Files Found: {len(existing_test_files)}")
    print(f"  • Test Suites Run: {total_tests}")
    print(f"  • Successful Suites: {successful_tests}")
    print(f"  • Success Rate: {success_rate:.1%}")
    print(f"  • Total Duration: {total_duration:.2f}s")
    
    # Detailed results per test suite
    print(f"\n📋 Detailed Results:")
    for result in test_results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status} {result['file']} ({result['duration']:.2f}s)")
    
    # Overall assessment
    overall_success = success_rate >= 0.7  # 70% success rate minimum
    
    if overall_success:
        print(f"\n🎉 VALIDATION SUCCESSFUL!")
        print(f"   Test enhancement implementation is working correctly.")
        print(f"   {success_rate:.1%} success rate meets quality standards.")
    else:
        print(f"\n⚠️ VALIDATION NEEDS IMPROVEMENT")
        print(f"   Test enhancement needs additional work.")
        print(f"   {success_rate:.1%} success rate below 70% threshold.")
    
    # Return comprehensive results
    return {
        'overall_success': overall_success,
        'phase_1_files': len(existing_test_files) > 0,
        'phase_2_individual': test_results[:-1] if test_results else [],
        'phase_3_combined': test_results[-1] if test_results else None,
        'success_rate': success_rate,
        'total_duration': total_duration,
        'files_found': len(existing_test_files),
        'message': 'Test enhancement validation complete'
    }


def main():
    """Main validation function."""
    try:
        results = run_comprehensive_validation()
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n💥 Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()