#!/usr/bin/env python3
"""
Quick fix script for critical PlatformOrchestrator test issues.

Addresses:
1. pytest.mock.patch -> unittest.mock.patch import issue
2. Attribute name mismatches in ABTestingServiceImpl tests
"""

import re
from pathlib import Path

def fix_conftest_imports():
    """Fix the pytest.mock.patch import issue in conftest.py."""
    conftest_path = Path("tests/unit/test_platform_orchestrator/conftest.py")
    
    if not conftest_path.exists():
        print(f"Error: {conftest_path} not found")
        return False
    
    content = conftest_path.read_text()
    
    # Fix pytest.mock.patch -> unittest.mock.patch
    fixed_content = content.replace('pytest.mock.patch', 'patch')
    
    if content != fixed_content:
        conftest_path.write_text(fixed_content)
        print(f"✅ Fixed pytest.mock.patch import in {conftest_path}")
        return True
    else:
        print(f"ℹ️ No import fixes needed in {conftest_path}")
        return False

def fix_ab_testing_attributes():
    """Fix attribute name mismatches in ABTestingServiceImpl test."""
    test_path = Path("tests/unit/test_platform_orchestrator/test_ab_testing.py")
    
    if not test_path.exists():
        print(f"Error: {test_path} not found")
        return False
    
    content = test_path.read_text()
    original_content = content
    
    # Fix attribute names to match implementation
    fixes = [
        ('experiment_assignments', 'assignments'),
        ('experiment_results', 'results'), 
        ('experiment_configs', 'experiments')
    ]
    
    for old_attr, new_attr in fixes:
        content = content.replace(f'ab_testing_service.{old_attr}', f'ab_testing_service.{new_attr}')
    
    if content != original_content:
        test_path.write_text(content)
        print(f"✅ Fixed attribute names in {test_path}")
        return True
    else:
        print(f"ℹ️ No attribute fixes needed in {test_path}")
        return False

def main():
    """Apply critical fixes for test infrastructure."""
    print("🔧 Applying critical fixes for PlatformOrchestrator test issues...")
    
    fixes_applied = []
    
    # Fix import issues
    if fix_conftest_imports():
        fixes_applied.append("conftest imports")
    
    # Fix attribute mismatches  
    if fix_ab_testing_attributes():
        fixes_applied.append("AB testing attributes")
    
    if fixes_applied:
        print(f"\n✅ Applied fixes: {', '.join(fixes_applied)}")
        print("🧪 Ready to re-run tests with fixes applied")
    else:
        print("\nℹ️ No fixes needed - investigating other issues...")
    
    return len(fixes_applied) > 0

if __name__ == "__main__":
    main()