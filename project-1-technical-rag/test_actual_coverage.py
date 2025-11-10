#!/usr/bin/env python3
"""
Test actual PlatformOrchestrator coverage using existing methods.
"""

import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_actual_coverage():
    """Test actual implementation coverage."""
    print("🧪 Testing actual implementation coverage...")
    
    try:
        from src.core.platform_orchestrator import (
            ComponentHealthServiceImpl,
            SystemAnalyticsServiceImpl,
            ABTestingServiceImpl,
            BackendManagementServiceImpl
        )
        
        # Health Service - test actual methods
        health = ComponentHealthServiceImpl()
        health_result = health.check_component_health(health)
        print(f"✅ Health check: {type(health_result).__name__}")
        
        # Analytics Service - use actual methods
        analytics = SystemAnalyticsServiceImpl()
        analytics.track_component_performance("test_component", {"response_time": 0.1})
        metrics = analytics.aggregate_system_metrics()
        report = analytics.generate_analytics_report()
        print(f"✅ Analytics metrics: {len(metrics)} entries, report: {type(report).__name__}")
        
        # AB Testing Service - test actual structure
        ab_service = ABTestingServiceImpl()
        print(f"✅ AB Testing initialized: experiments={len(ab_service.experiments)}, active={len(ab_service.active_experiments)}")
        
        # Backend Management Service
        backend = BackendManagementServiceImpl()
        backends = backend.list_backends()
        print(f"✅ Backend service: {len(backends)} backends registered")
        
        print("\n📊 Coverage Summary:")
        print("- ComponentHealthServiceImpl: Basic functionality working")
        print("- SystemAnalyticsServiceImpl: Core methods operational") 
        print("- ABTestingServiceImpl: Initialization and basic structure")
        print("- BackendManagementServiceImpl: Basic operations")
        
        return True
        
    except Exception as e:
        print(f"❌ Coverage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Actual PlatformOrchestrator Implementation Coverage")
    print("=" * 60)
    success = test_actual_coverage()
    print("=" * 60)
    print("✅ Implementation validation complete" if success else "❌ Implementation needs fixes")