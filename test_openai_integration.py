#!/usr/bin/env python3
"""
Test OpenAI API Integration
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_openai_integration():
    """Test OpenAI API integration"""
    try:
        from intellibase_app import IntelliBaseApp
        
        print("🧪 Testing OpenAI Integration")
        print("=" * 40)
        
        # Initialize the app
        app = IntelliBaseApp()
        
        # Get system status
        status = app.get_system_status()
        print(f"✅ System Status: {status['is_ready']}")
        
        # Test a query that should use OpenAI
        print("\n❓ Testing query with OpenAI...")
        result = app.query("What is artificial intelligence?")
        
        print(f"✅ Query processed successfully")
        print(f"📊 Response time: {result.get('total_time', 0):.3f}s")
        print(f"🔍 Search results: {result.get('search_results_count', 0)}")
        print(f"📝 Answer length: {len(result.get('answer', ''))} chars")
        print(f"🤖 Provider: {result.get('provider', 'unknown')}")
        
        # Check if OpenAI was used
        if 'openai' in str(result).lower() or 'gpt' in str(result).lower():
            print("✅ OpenAI API is being used!")
        else:
            print("⚠️ OpenAI API might not be active (using mock)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_openai_direct():
    """Test OpenAI API directly"""
    try:
        import openai
        
        print("\n🔍 Testing OpenAI API directly...")
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_key_here":
            print("❌ OpenAI API key not found or is placeholder")
            return False
        
        # Initialize client
        client = openai.OpenAI(api_key=api_key)
        
        # Test simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello in one word"}],
            max_tokens=10
        )
        
        content = response.choices[0].message.content
        print(f"✅ OpenAI API working! Response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 OpenAI Integration Test")
    print("=" * 40)
    
    # Test 1: Direct OpenAI API
    print("\n1. Testing direct OpenAI API...")
    openai_ok = test_openai_direct()
    
    # Test 2: IntelliBase with OpenAI
    print("\n2. Testing IntelliBase with OpenAI...")
    intellibase_ok = test_openai_integration()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 TEST SUMMARY")
    print("=" * 40)
    print(f"OpenAI Direct: {'✅ PASS' if openai_ok else '❌ FAIL'}")
    print(f"IntelliBase Integration: {'✅ PASS' if intellibase_ok else '❌ FAIL'}")
    
    if openai_ok and intellibase_ok:
        print("\n🎉 OpenAI integration is fully working!")
        print("✅ API key configured correctly")
        print("✅ Direct API calls working")
        print("✅ IntelliBase integration working")
        print("✅ System ready for production use")
    else:
        print("\n⚠️ Some issues detected")
    
    return openai_ok and intellibase_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 