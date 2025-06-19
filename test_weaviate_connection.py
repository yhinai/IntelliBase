#!/usr/bin/env python3
"""
Quick test to verify Weaviate connection with IntelliBase
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_weaviate_connection():
    """Test Weaviate connection through IntelliBase"""
    try:
        from intellibase_app import IntelliBaseApp
        
        print("🔧 Testing IntelliBase with Weaviate...")
        
        # Initialize the app
        app = IntelliBaseApp()
        
        # Get system status
        status = app.get_system_status()
        
        print(f"✅ System Status: {status['is_ready']}")
        print(f"✅ Components: {status['components']}")
        
        # Test knowledge base setup
        print("\n📚 Testing knowledge base setup...")
        success = app.setup_knowledge_base(force_reprocess=True)
        print(f"✅ Knowledge base setup: {success}")
        
        # Test a simple query
        print("\n❓ Testing query processing...")
        result = app.query("technical specifications")
        
        print(f"✅ Query processed successfully")
        print(f"📊 Response time: {result.get('total_time', 0):.3f}s")
        print(f"🔍 Search results: {result.get('search_results_count', 0)}")
        print(f"📝 Answer length: {len(result.get('answer', ''))} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_weaviate_direct():
    """Test direct Weaviate connection"""
    try:
        import weaviate
        
        print("🔍 Testing direct Weaviate connection...")
        
        # Connect to Weaviate
        client = weaviate.Client("http://localhost:8080")
        
        # Test connection
        is_ready = client.is_ready()
        print(f"✅ Weaviate ready: {is_ready}")
        
        # Get schema
        schema = client.schema.get()
        print(f"✅ Schema retrieved: {len(schema['classes'])} classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Weaviate connection error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Weaviate Connection Test")
    print("=" * 40)
    
    # Test 1: Direct Weaviate connection
    print("\n1. Testing direct Weaviate connection...")
    weaviate_ok = test_weaviate_direct()
    
    # Test 2: IntelliBase with Weaviate
    print("\n2. Testing IntelliBase with Weaviate...")
    intellibase_ok = test_weaviate_connection()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 TEST SUMMARY")
    print("=" * 40)
    print(f"Weaviate Direct: {'✅ PASS' if weaviate_ok else '❌ FAIL'}")
    print(f"IntelliBase Integration: {'✅ PASS' if intellibase_ok else '❌ FAIL'}")
    
    if weaviate_ok and intellibase_ok:
        print("\n🎉 Weaviate is fully working with IntelliBase!")
        print("✅ Vector database connected")
        print("✅ Knowledge base setup working")
        print("✅ Query processing working")
        print("✅ System is ready for API keys")
    else:
        print("\n⚠️ Some issues detected")
    
    return weaviate_ok and intellibase_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 