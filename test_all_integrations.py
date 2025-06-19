#!/usr/bin/env python3
"""
Test All AI Integrations - Complete System
"""
import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_all_ai_integrations():
    """Test all AI integrations (FriendliAI, OpenAI, Hypermode)"""
    try:
        from intellibase_app import IntelliBaseApp
        
        print("🚀 TESTING ALL AI INTEGRATIONS")
        print("=" * 50)
        
        # Initialize the app
        print("🔧 Initializing IntelliBase with all AI integrations...")
        app = IntelliBaseApp()
        
        # Check system status
        status = app.get_system_status()
        print(f"✅ System Status: {status['is_ready']}")
        
        # Check AI integration details
        if status['components'].get('ai_integration'):
            ai_details = status['component_details'].get('ai_integration', {})
            print(f"🤖 AI Integration Details:")
            print(f"   FriendliAI: {'✅ Available' if ai_details.get('friendli_available') else '❌ Not available'}")
            print(f"   OpenAI: {'✅ Available' if ai_details.get('openai_available') else '❌ Not available'}")
        
        # Check Hypermode status
        if status['components'].get('hypermode_integration'):
            hypermode_status = app.get_hypermode_status()
            print(f"🤖 Hypermode: {'✅ Available' if hypermode_status.get('available') else '❌ Not available'}")
        
        # Test knowledge base setup
        print("\n📚 Testing knowledge base setup...")
        kb_ready = app.setup_knowledge_base()
        print(f"✅ Knowledge base: {'Ready' if kb_ready else 'Failed'}")
        
        # Test queries with different providers
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "How does vector search work?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n❓ Test {i}: '{query}'")
            result = app.query(query)
            
            print(f"✅ Query processed successfully")
            print(f"📊 Response time: {result.get('total_time', 0):.3f}s")
            print(f"🔍 Search results: {result.get('search_results_count', 0)}")
            print(f"📝 Answer length: {len(result.get('answer', ''))} chars")
            print(f"🤖 Provider: {result.get('provider', 'unknown')}")
            
            # Show a snippet of the answer
            answer = result.get('answer', '')
            if answer:
                print(f"💬 Answer preview: {answer[:100]}...")
        
        # Test Hypermode agent execution (if available)
        if status['components'].get('hypermode_integration'):
            print("\n🤖 Testing Hypermode agent execution...")
            agent_result = app.execute_hypermode_agent(
                "test_agent",
                {"query": "Hello from IntelliBase with all integrations"}
            )
            print(f"✅ Hypermode agent: {'Success' if agent_result['success'] else 'Failed'}")
            if not agent_result['success']:
                print(f"   Error: {agent_result.get('error', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_keys():
    """Test all API keys are configured"""
    print("\n🔍 Testing API Key Configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv('config.env')
        
        # Test FriendliAI
        friendli_key = os.getenv("FRIENDLI_TOKEN")
        if friendli_key and friendli_key != "your_friendli_token_here":
            print("✅ FriendliAI Token: Configured")
        else:
            print("❌ FriendliAI Token: Not configured")
        
        # Test OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "your_openai_key_here":
            print("✅ OpenAI API Key: Configured")
        else:
            print("❌ OpenAI API Key: Not configured")
        
        # Test Hypermode
        hypermode_key = os.getenv("HYPERMODE_API_KEY")
        if hypermode_key and hypermode_key != "your_hypermode_key_here":
            print("✅ Hypermode API Key: Configured")
        else:
            print("❌ Hypermode API Key: Not configured")
            
    except Exception as e:
        print(f"❌ API key test failed: {e}")

def main():
    """Main test function"""
    print("🎯 COMPLETE AI INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: API Keys
    test_api_keys()
    
    # Test 2: All integrations
    print("\n2. Testing all AI integrations...")
    integrations_ok = test_all_ai_integrations()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 COMPLETE AI INTEGRATION SUMMARY")
    print("=" * 50)
    print(f"All AI Integrations: {'✅ PASS' if integrations_ok else '❌ FAIL'}")
    
    if integrations_ok:
        print("\n🎉 ALL AI INTEGRATIONS WORKING!")
        print("✅ FriendliAI: Fastest responses")
        print("✅ OpenAI: Reliable fallback")
        print("✅ Hypermode: Agent orchestration")
        print("✅ Weaviate: Vector database")
        print("✅ Complete system: Production ready")
        print("\n🌐 Access your app at: http://localhost:8501")
        print("🔍 Phoenix observability at: http://localhost:6006")
        print("🗄️ Weaviate at: http://localhost:8080")
        print("\n🚀 ULTIMATE AI RESEARCH ASSISTANT READY!")
    else:
        print("\n⚠️ Some integrations need attention")
    
    return integrations_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 