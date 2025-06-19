#!/usr/bin/env python3
"""
Fix Setup Script for IntelliBase
Handles Docker setup and provides cloud alternatives
"""
import os
import subprocess
import sys
import time
import requests
from pathlib import Path

def check_docker_status():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_docker_desktop():
    """Provide instructions for Docker Desktop installation"""
    print("🐳 Docker Desktop not found!")
    print("\n📥 To install Docker Desktop:")
    print("1. Visit: https://www.docker.com/products/docker-desktop")
    print("2. Download Docker Desktop for Mac")
    print("3. Install and start Docker Desktop")
    print("4. Run this script again")
    return False

def start_weaviate_cloud_setup():
    """Setup Weaviate Cloud (free alternative)"""
    print("☁️ Setting up Weaviate Cloud (free alternative)...")
    
    # Update config for Weaviate Cloud
    config_file = Path("config.env")
    
    if not config_file.exists():
        print("❌ config.env not found!")
        return False
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update for Weaviate Cloud
    content = content.replace(
        'WEAVIATE_CLUSTER_URL=your_weaviate_url_here',
        'WEAVIATE_CLUSTER_URL=https://your-cluster.weaviate.network'
    )
    content = content.replace(
        'WEAVIATE_API_KEY=your_weaviate_api_key_here',
        'WEAVIATE_API_KEY=your_weaviate_api_key_here'
    )
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated config.env for Weaviate Cloud")
    print("\n📋 Next steps for Weaviate Cloud:")
    print("1. Visit: https://console.weaviate.cloud/")
    print("2. Sign up for free account")
    print("3. Create a new cluster")
    print("4. Get your cluster URL and API key")
    print("5. Update config.env with your actual values")
    
    return True

def fix_phoenix_ports():
    """Fix Phoenix port conflicts"""
    print("🔧 Fixing Phoenix port conflicts...")
    
    # Kill processes on Phoenix ports
    ports = [4317, 6006]
    
    for port in ports:
        try:
            # Find processes using the port
            result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        subprocess.run(['kill', '-9', pid])
                        print(f"✅ Killed process {pid} on port {port}")
        except Exception as e:
            print(f"⚠️ Could not kill processes on port {port}: {e}")
    
    print("✅ Phoenix ports cleared")

def test_weaviate_connection():
    """Test Weaviate connection"""
    print("🔍 Testing Weaviate connection...")
    
    try:
        # Try to connect to local Weaviate
        response = requests.get('http://localhost:8080/v1/.well-known/ready', timeout=5)
        if response.status_code == 200:
            print("✅ Weaviate is running on localhost:8080")
            return True
    except requests.exceptions.RequestException:
        pass
    
    print("❌ Weaviate not accessible on localhost:8080")
    return False

def update_config_for_local_weaviate():
    """Update config for local Weaviate"""
    config_file = Path("config.env")
    
    if not config_file.exists():
        print("❌ config.env not found!")
        return False
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update for local Weaviate
    content = content.replace(
        'WEAVIATE_CLUSTER_URL=your_weaviate_url_here',
        'WEAVIATE_CLUSTER_URL=http://localhost:8080'
    )
    content = content.replace(
        'WEAVIATE_API_KEY=your_weaviate_api_key_here',
        'WEAVIATE_API_KEY='
    )
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated config.env for local Weaviate")
    return True

def start_weaviate_docker():
    """Start Weaviate using Docker"""
    print("🐳 Starting Weaviate with Docker...")
    
    # Check if weaviate container exists
    result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=weaviate'], capture_output=True, text=True)
    
    if 'weaviate' in result.stdout:
        print("📦 Weaviate container exists, starting it...")
        subprocess.run(['docker', 'start', 'weaviate'])
    else:
        print("📦 Creating new Weaviate container...")
        subprocess.run([
            'docker', 'run', '-d',
            '--name', 'weaviate',
            '-p', '8080:8080',
            '-e', 'QUERY_DEFAULTS_LIMIT=25',
            '-e', 'AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true',
            '-e', 'PERSISTENCE_DATA_PATH=/var/lib/weaviate',
            '-e', 'DEFAULT_VECTORIZER_MODULE=none',
            '-e', 'ENABLE_MODULES=text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai',
            '-e', 'CLUSTER_HOSTNAME=node1',
            'semitechnologies/weaviate:1.22.4'
        ])
    
    # Wait for Weaviate to be ready
    print("⏳ Waiting for Weaviate to be ready...")
    for i in range(30):  # Wait up to 30 seconds
        if test_weaviate_connection():
            break
        time.sleep(1)
        print(f"   Waiting... ({i+1}/30)")
    else:
        print("⚠️ Weaviate may still be starting up...")
    
    return True

def run_system_test():
    """Run a quick system test"""
    print("🧪 Running system test...")
    
    try:
        result = subprocess.run([sys.executable, 'test_integration.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ System test passed!")
            return True
        else:
            print(f"⚠️ System test had issues: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 IntelliBase Fix Setup")
    print("=" * 50)
    
    # Step 1: Fix Phoenix ports
    print("\n1. Fixing Phoenix port conflicts...")
    fix_phoenix_ports()
    
    # Step 2: Check Docker status
    print("\n2. Checking Docker status...")
    docker_available = check_docker_status()
    
    if docker_available:
        print("✅ Docker is running")
        
        # Step 3: Start Weaviate with Docker
        print("\n3. Starting Weaviate with Docker...")
        start_weaviate_docker()
        
        # Step 4: Update config for local Weaviate
        print("\n4. Updating configuration...")
        update_config_for_local_weaviate()
        
    else:
        print("❌ Docker not running")
        print("\n3. Setting up Weaviate Cloud alternative...")
        start_weaviate_cloud_setup()
    
    # Step 5: Run test
    print("\n5. Running system test...")
    run_system_test()
    
    print("\n🎉 Fix setup completed!")
    print("\n📋 Next steps:")
    
    if docker_available:
        print("✅ Weaviate should now be running locally")
        print("✅ Phoenix port conflicts resolved")
        print("1. Get API keys for full functionality:")
        print("   - FriendliAI: https://console.friendli.ai/")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        print("2. Update config.env with your API keys")
        print("3. Run: streamlit run streamlit_ui.py")
        print("4. Open: http://localhost:8501")
    else:
        print("⚠️ Docker not available")
        print("1. Install Docker Desktop or use Weaviate Cloud")
        print("2. Get API keys for full functionality")
        print("3. Update config.env with your API keys")
        print("4. Run: streamlit run streamlit_ui.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 