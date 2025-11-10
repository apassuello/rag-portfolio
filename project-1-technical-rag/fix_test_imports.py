#!/usr/bin/env python3
"""
Fix import issues in conftest.py for PlatformOrchestrator tests.
"""

from pathlib import Path

def fix_conftest_imports():
    """Fix the patch import in conftest.py."""
    conftest_path = Path("tests/unit/test_platform_orchestrator/conftest.py")
    
    if not conftest_path.exists():
        print(f"Error: {conftest_path} not found")
        return False
    
    content = conftest_path.read_text()
    
    # Add patch import to the existing imports
    if "from unittest.mock import Mock, MagicMock" in content and "patch" not in content:
        content = content.replace(
            "from unittest.mock import Mock, MagicMock",
            "from unittest.mock import Mock, MagicMock, patch"
        )
        
        conftest_path.write_text(content)
        print(f"✅ Fixed patch import in {conftest_path}")
        return True
    else:
        print(f"ℹ️ Import already correct or different pattern in {conftest_path}")
        return False

if __name__ == "__main__":
    fix_conftest_imports()