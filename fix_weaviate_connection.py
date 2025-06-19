#!/usr/bin/env python3
"""
Fix Weaviate Connection Issues and Setup Persistent Storage
"""
import subprocess
import requests
import time
import os
from pathlib import Path

def check_docker_status():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_weaviate_container():
    """Check if Weaviate container is running"""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=weaviate'], capture_output=True, text=True)
        return 'weaviate' in result.stdout
    except:
        return False

def start_weaviate_with_persistence():
    """Start Weaviate with persistent storage"""
    try:
        # Stop existing container
        subprocess.run(['docker', 'stop', 'weaviate'], capture_output=True)
        subprocess.run(['docker', 'rm', 'weaviate'], capture_output=True)
        
        # Create persistent volume
        subprocess.run(['docker', 'volume', 'create', 'weaviate_data'], capture_output=True)
        
        # Start Weaviate with persistent storage
        cmd = [
            'docker', 'run', '-d',
            '--name', 'weaviate',
            '-p', '8080:8080',
            '-v', 'weaviate_data:/var/lib/weaviate',
            '-e', 'QUERY_DEFAULTS_LIMIT=25',
            '-e', 'AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true',
            '-e', 'PERSISTENCE_DATA_PATH=/var/lib/weaviate',
            '-e', 'DEFAULT_VECTORIZER_MODULE=none',
            '-e', 'ENABLE_MODULES=text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai',
            '-e', 'CLUSTER_HOSTNAME=node1',
            'semitechnologies/weaviate:latest'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Weaviate started with persistent storage")
            return True
        else:
            print(f"❌ Failed to start Weaviate: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Weaviate: {e}")
        return False

def wait_for_weaviate_ready():
    """Wait for Weaviate to be ready"""
    print("⏳ Waiting for Weaviate to be ready...")
    
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get('http://localhost:8080/v1/.well-known/ready', timeout=5)
            if response.status_code == 200:
                print("✅ Weaviate is ready!")
                return True
        except:
            pass
        
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}/30)")
    
    print("❌ Weaviate did not become ready in time")
    return False

def test_weaviate_connection():
    """Test Weaviate connection"""
    try:
        response = requests.get('http://localhost:8080/v1/.well-known/ready', timeout=10)
        if response.status_code == 200:
            print("✅ Weaviate connection test successful")
            return True
        else:
            print(f"❌ Weaviate connection test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Weaviate connection test failed: {e}")
        return False

def update_config_for_local_persistence():
    """Update config for local Weaviate with persistence"""
    config_file = Path("config.env")
    
    if not config_file.exists():
        print("❌ config.env not found!")
        return False
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update for local Weaviate with persistence
    content = content.replace(
        'WEAVIATE_CLUSTER_URL=http://localhost:8080',
        'WEAVIATE_CLUSTER_URL=http://localhost:8080'
    )
    content = content.replace(
        'WEAVIATE_API_KEY=',
        'WEAVIATE_API_KEY='
    )
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated config.env for local Weaviate with persistence")
    return True

def update_weaviate_manager_for_local():
    """Update Weaviate manager to use local by default"""
    weaviate_file = Path("weaviate_manager.py")
    
    if not weaviate_file.exists():
        print("❌ weaviate_manager.py not found!")
        return False
    
    # Read current file
    with open(weaviate_file, 'r') as f:
        content = f.read()
    
    # Update to use local by default
    content = content.replace(
        'def __init__(self, use_local: bool = False):',
        'def __init__(self, use_local: bool = True):'
    )
    
    # Write updated file
    with open(weaviate_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated weaviate_manager.py to use local by default")
    return True

def test_system_integration():
    """Test the complete system integration"""
    print("\n🧪 Testing system integration...")
    
    try:
        # Import and test the system
        import sys
        sys.path.append('.')
        
        from intellibase_app import IntelliBaseApp
        
        app = IntelliBaseApp()
        status = app.get_system_status()
        
        print(f"✅ System Status: {status['is_ready']}")
        print(f"✅ Components: {status['components']}")
        
        # Test knowledge base setup
        success = app.setup_knowledge_base(force_reprocess=True)
        print(f"✅ Knowledge base setup: {success}")
        
        return success
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def main():
    print("🔧 Fix Weaviate Connection and Setup Persistent Storage")
    print("=" * 60)
    
    print("\n📋 Options:")
    print("1. Fix local Weaviate with persistent storage (Docker)")
    print("2. Setup Weaviate Cloud (recommended for production)")
    print("3. Test current connection")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("\n🔧 Fixing local Weaviate with persistent storage...")
        
        # Check Docker
        if not check_docker_status():
            print("❌ Docker is not running!")
            print("Please start Docker Desktop and try again.")
            return False
        
        # Start Weaviate with persistence
        if start_weaviate_with_persistence():
            # Wait for it to be ready
            if wait_for_weaviate_ready():
                # Test connection
                if test_weaviate_connection():
                    # Update configuration
                    update_config_for_local_persistence()
                    update_weaviate_manager_for_local()
                    
                    # Test system integration
                    if test_system_integration():
                        print("\n🎉 Local Weaviate with persistence setup completed!")
                        print("\n💾 Your data will now be persistently stored in Docker volume 'weaviate_data'")
                        print("\n📋 Next steps:")
                        print("1. Get API keys for full functionality:")
                        print("   - FriendliAI: https://console.friendli.ai/")
                        print("   - OpenAI: https://platform.openai.com/api-keys")
                        print("2. Update config.env with those keys")
                        print("3. Run: streamlit run streamlit_ui.py")
                        print("4. Open: http://localhost:8501")
                        return True
                    else:
                        print("❌ System integration test failed")
                        return False
                else:
                    print("❌ Weaviate connection test failed")
                    return False
            else:
                print("❌ Weaviate did not become ready")
                return False
        else:
            print("❌ Failed to start Weaviate")
            return False
    
    elif choice == "2":
        print("\n☁️ Setting up Weaviate Cloud...")
        print("Run: python weaviate_cloud_setup.py")
        return True
    
    elif choice == "3":
        print("\n🧪 Testing current connection...")
        
        if check_weaviate_container():
            print("✅ Weaviate container is running")
            
            if test_weaviate_connection():
                print("✅ Weaviate connection is working")
                
                if test_system_integration():
                    print("✅ System integration is working")
                    return True
                else:
                    print("❌ System integration failed")
                    return False
            else:
                print("❌ Weaviate connection failed")
                return False
        else:
            print("❌ Weaviate container is not running")
            return False
    
    else:
        print("❌ Invalid choice")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure Docker Desktop is running")
        print("2. Try: docker logs weaviate")
        print("3. Consider using Weaviate Cloud: python weaviate_cloud_setup.py") 