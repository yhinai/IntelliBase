#!/usr/bin/env python3
"""
Quick Setup Script for IntelliBase
Helps users get the system working with minimal configuration
"""
import os
import subprocess
import sys
from pathlib import Path

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def start_weaviate_local():
    """Start Weaviate locally using Docker"""
    print("🐳 Starting Weaviate locally...")
    
    # Check if weaviate container already exists
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
    
    print("✅ Weaviate started at http://localhost:8080")

def update_config_env():
    """Update config.env with local Weaviate settings"""
    config_file = Path("config.env")
    
    if not config_file.exists():
        print("❌ config.env not found!")
        return False
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update Weaviate settings for local
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

def check_ports():
    """Check if required ports are available"""
    import socket
    
    ports_to_check = [8080, 6006, 8501]
    
    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"⚠️ Port {port} is already in use")
        else:
            print(f"✅ Port {port} is available")

def install_missing_deps():
    """Install missing dependencies"""
    print("📦 Installing missing dependencies...")
    
    deps = [
        'streamlit',
        'plotly', 
        'pandas',
        'weaviate-client',
        'arize-phoenix',
        'openai'
    ]
    
    for dep in deps:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True)
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")

def run_quick_test():
    """Run a quick test to verify setup"""
    print("🧪 Running quick test...")
    
    try:
        result = subprocess.run([sys.executable, 'test_integration.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Quick test passed!")
            return True
        else:
            print(f"❌ Quick test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 IntelliBase Quick Setup")
    print("=" * 50)
    
    # Step 1: Check Docker
    print("\n1. Checking Docker...")
    if check_docker():
        print("✅ Docker is available")
    else:
        print("❌ Docker not found. Please install Docker first.")
        print("   Download from: https://www.docker.com/products/docker-desktop")
        return False
    
    # Step 2: Check ports
    print("\n2. Checking ports...")
    check_ports()
    
    # Step 3: Install dependencies
    print("\n3. Installing dependencies...")
    install_missing_deps()
    
    # Step 4: Start Weaviate
    print("\n4. Starting Weaviate...")
    start_weaviate_local()
    
    # Step 5: Update config
    print("\n5. Updating configuration...")
    update_config_env()
    
    # Step 6: Wait for Weaviate to be ready
    print("\n6. Waiting for Weaviate to be ready...")
    import time
    time.sleep(10)
    
    # Step 7: Run test
    print("\n7. Running quick test...")
    if run_quick_test():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Get API keys for full functionality:")
        print("   - FriendliAI: https://console.friendli.ai/")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        print("2. Update config.env with your API keys")
        print("3. Run: streamlit run streamlit_ui.py")
        print("4. Open: http://localhost:8501")
        
        return True
    else:
        print("\n⚠️ Setup completed with issues. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 