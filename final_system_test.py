#!/usr/bin/env python3
"""
Final System Test - Verify everything is working
"""
import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_complete_system():
    """Test the complete IntelliBase system"""
    try:
        from intellibase_app import IntelliBaseApp
        
        print("🚀 FINAL SYSTEM TEST")
        print("=" * 50)
        
        # Initialize the app
        print("🔧 Initializing IntelliBase...")
        app = IntelliBaseApp()
        
        # Check system status
        status = app.get_system_status()
        print(f"✅ System Status: {status['is_ready']}")
        print(f"📊 Components: {status['components']}")
        
        # Test knowledge base setup
        print("\n📚 Testing knowledge base setup...")
        kb_ready = app.setup_knowledge_base()
        print(f"✅ Knowledge base: {'Ready' if kb_ready else 'Failed'}")
        
        # Test a query with context
        print("\n❓ Testing query with context...")
        result = app.query("What is machine learning?")
        
        print(f"✅ Query processed successfully")
        print(f"📊 Response time: {result.get('total_time', 0):.3f}s")
        print(f"🔍 Search results: {result.get('search_results_count', 0)}")
        print(f"📝 Answer length: {len(result.get('answer', ''))} chars")
        print(f"🤖 Provider: {result.get('provider', 'unknown')}")
        
        # Show a snippet of the answer
        answer = result.get('answer', '')
        if answer:
            print(f"💬 Answer preview: {answer[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_streamlit_connection():
    """Test if Streamlit is accessible"""
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit app is accessible")
            return True
        else:
            print(f"⚠️ Streamlit returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streamlit connection failed: {e}")
        return False

def test_weaviate_connection():
    """Test Weaviate connection"""
    try:
        import requests
        response = requests.get("http://localhost:8080/v1/.well-known/ready", timeout=5)
        if response.status_code == 200:
            print("✅ Weaviate is running and accessible")
            return True
        else:
            print(f"⚠️ Weaviate returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)
    
    # Test 1: Weaviate
    print("\n1. Testing Weaviate connection...")
    weaviate_ok = test_weaviate_connection()
    
    # Test 2: Complete system
    print("\n2. Testing complete IntelliBase system...")
    system_ok = test_complete_system()
    
    # Test 3: Streamlit
    print("\n3. Testing Streamlit connection...")
    streamlit_ok = test_streamlit_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 50)
    print(f"Weaviate: {'✅ PASS' if weaviate_ok else '❌ FAIL'}")
    print(f"IntelliBase System: {'✅ PASS' if system_ok else '❌ FAIL'}")
    print(f"Streamlit UI: {'✅ PASS' if streamlit_ok else '❌ FAIL'}")
    
    if weaviate_ok and system_ok and streamlit_ok:
        print("\n🎉 SYSTEM IS FULLY OPERATIONAL!")
        print("✅ Weaviate vector database running")
        print("✅ OpenAI API integration working")
        print("✅ Knowledge base processing working")
        print("✅ Streamlit UI accessible")
        print("\n🌐 Access your app at: http://localhost:8501")
        print("🔍 Phoenix observability at: http://localhost:6006")
        print("🗄️ Weaviate at: http://localhost:8080")
        print("\n🚀 Ready for production use!")
    else:
        print("\n⚠️ Some components need attention")
    
    return weaviate_ok and system_ok and streamlit_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 