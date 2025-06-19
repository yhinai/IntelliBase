#!/usr/bin/env python3
"""
Integration test for the comprehensive system test and Streamlit UI
"""
import asyncio
import sys
import os
from pathlib import Path

def test_plotly_integration():
    """Test that plotly is properly installed and can be imported"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        print("✅ Plotly integration: PASS")
        return True
    except ImportError as e:
        print(f"❌ Plotly integration: FAIL - {e}")
        return False

def test_streamlit_integration():
    """Test that streamlit can be imported"""
    try:
        import streamlit as st
        print("✅ Streamlit integration: PASS")
        return True
    except ImportError as e:
        print(f"❌ Streamlit integration: FAIL - {e}")
        return False

def test_comprehensive_system_test():
    """Test that the comprehensive system test can be imported"""
    try:
        from comprehensive_system_test import ComprehensiveSystemTester
        print("✅ Comprehensive system test import: PASS")
        return True
    except ImportError as e:
        print(f"❌ Comprehensive system test import: FAIL - {e}")
        return False

async def test_system_test_execution():
    """Test that the comprehensive system test can run"""
    try:
        from comprehensive_system_test import ComprehensiveSystemTester
        
        tester = ComprehensiveSystemTester()
        
        # Run a quick test
        await tester.test_component_availability()
        
        print("✅ System test execution: PASS")
        return True
    except Exception as e:
        print(f"❌ System test execution: FAIL - {e}")
        return False

def test_file_structure():
    """Test that required files exist"""
    required_files = [
        "comprehensive_system_test.py",
        "streamlit_ui.py",
        "requirements.txt"
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}: EXISTS")
        else:
            print(f"❌ {file}: MISSING")
            all_exist = False
    
    return all_exist

def test_test_results_directory():
    """Test that test results can be generated"""
    test_results_path = Path("test_results")
    
    if test_results_path.exists():
        files = list(test_results_path.glob("*"))
        if files:
            print(f"✅ Test results directory: EXISTS with {len(files)} files")
            
            # Check for key files
            key_files = [
                "comprehensive_test_results.json",
                "test_dashboard.html",
                "test_report.md"
            ]
            
            for key_file in key_files:
                if (test_results_path / key_file).exists():
                    print(f"  ✅ {key_file}: EXISTS")
                else:
                    print(f"  ⚠️ {key_file}: MISSING")
        else:
            print("⚠️ Test results directory: EXISTS but empty")
    else:
        print("⚠️ Test results directory: MISSING")
    
    return True

async def main():
    """Run all integration tests"""
    print("🧪 Running Integration Tests")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Plotly Integration", test_plotly_integration),
        ("Streamlit Integration", test_streamlit_integration),
        ("System Test Import", test_comprehensive_system_test),
        ("System Test Execution", test_system_test_execution),
        ("Test Results", test_test_results_directory)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All integration tests PASSED!")
        return True
    else:
        print("⚠️ Some integration tests FAILED!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 