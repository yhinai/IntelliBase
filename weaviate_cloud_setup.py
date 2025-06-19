#!/usr/bin/env python3
"""
Weaviate Cloud Setup with Persistent Storage
"""
import requests
import json
import os
from pathlib import Path
import time

def test_weaviate_cloud_connection(url, api_key):
    """Test connection to Weaviate Cloud"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(f'{url}/v1/.well-known/ready', headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def create_persistent_collection(url, api_key, collection_name="IntellibaseKnowledge"):
    """Create a persistent collection in Weaviate Cloud"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Collection schema with persistence
        collection_schema = {
            "class": collection_name,
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "poolingStrategy": "masked_mean",
                    "vectorizeClassName": False
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The main text content",
                    "indexInverted": True
                },
                {
                    "name": "content_type",
                    "dataType": ["text"],
                    "description": "Type of content (pdf, image, text)",
                    "indexInverted": True
                },
                {
                    "name": "source_file",
                    "dataType": ["text"],
                    "description": "Original file name",
                    "indexInverted": True
                },
                {
                    "name": "source_path",
                    "dataType": ["text"],
                    "description": "Original file path",
                    "indexInverted": True
                },
                {
                    "name": "chunk_id",
                    "dataType": ["text"],
                    "description": "Unique chunk identifier",
                    "indexInverted": True
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Index of chunk within source file"
                },
                {
                    "name": "file_id",
                    "dataType": ["text"],
                    "description": "Source file identifier",
                    "indexInverted": True
                },
                {
                    "name": "processed_at",
                    "dataType": ["date"],
                    "description": "Timestamp when chunk was processed"
                }
            ],
            "replicationConfig": {
                "factor": 1
            },
            "shardingConfig": {
                "virtualPerPhysical": 128,
                "desiredCount": 1,
                "actualCount": 1,
                "desiredVirtualCount": 128,
                "actualVirtualCount": 128,
                "key": "_id",
                "strategy": "hash",
                "function": "murmur3"
            }
        }
        
        # Create collection
        response = requests.post(
            f'{url}/v1/schema',
            headers=headers,
            json=collection_schema,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Collection '{collection_name}' created successfully")
            return True
        elif response.status_code == 422 and "already exists" in response.text:
            print(f"✅ Collection '{collection_name}' already exists")
            return True
        else:
            print(f"❌ Failed to create collection: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

def test_persistent_storage(url, api_key, collection_name="IntellibaseKnowledge"):
    """Test persistent storage by inserting and retrieving data"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Test data
        test_object = {
            "class": collection_name,
            "properties": {
                "content": "This is a test document for persistent storage verification",
                "content_type": "text",
                "source_file": "test_persistence.txt",
                "source_path": "test/test_persistence.txt",
                "chunk_id": "test_persistence_001",
                "chunk_index": 0,
                "file_id": "test_persistence_file",
                "processed_at": "2025-01-01T00:00:00Z"
            }
        }
        
        # Insert test object
        insert_response = requests.post(
            f'{url}/v1/objects',
            headers=headers,
            json=test_object,
            timeout=30
        )
        
        if insert_response.status_code != 200:
            print(f"❌ Failed to insert test object: {insert_response.status_code}")
            return False
        
        object_id = insert_response.json().get("id")
        print(f"✅ Test object inserted with ID: {object_id}")
        
        # Retrieve test object
        retrieve_response = requests.get(
            f'{url}/v1/objects/{object_id}',
            headers=headers,
            timeout=30
        )
        
        if retrieve_response.status_code == 200:
            retrieved_data = retrieve_response.json()
            print(f"✅ Test object retrieved successfully")
            print(f"   Content: {retrieved_data['properties']['content'][:50]}...")
            return True
        else:
            print(f"❌ Failed to retrieve test object: {retrieve_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing persistent storage: {e}")
        return False

def update_config_with_weaviate_cloud(url, api_key):
    """Update config.env with Weaviate Cloud credentials"""
    config_file = Path("config.env")
    
    if not config_file.exists():
        print("❌ config.env not found!")
        return False
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update Weaviate settings
    content = content.replace(
        'WEAVIATE_CLUSTER_URL=http://localhost:8080',
        f'WEAVIATE_CLUSTER_URL={url}'
    )
    content = content.replace(
        'WEAVIATE_API_KEY=',
        f'WEAVIATE_API_KEY={api_key}'
    )
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated config.env with Weaviate Cloud credentials")
    return True

def setup_weaviate_manager_for_cloud():
    """Update Weaviate manager to use cloud instead of local"""
    weaviate_file = Path("weaviate_manager.py")
    
    if not weaviate_file.exists():
        print("❌ weaviate_manager.py not found!")
        return False
    
    # Read current file
    with open(weaviate_file, 'r') as f:
        content = f.read()
    
    # Update to use cloud by default
    content = content.replace(
        'def __init__(self, use_local: bool = True):',
        'def __init__(self, use_local: bool = False):'
    )
    
    # Write updated file
    with open(weaviate_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated weaviate_manager.py to use cloud by default")
    return True

def main():
    print("☁️ Weaviate Cloud Setup with Persistent Storage")
    print("=" * 60)
    
    print("\n📋 To get your Weaviate Cloud credentials:")
    print("1. Visit: https://console.weaviate.cloud/")
    print("2. Sign up for free account")
    print("3. Create a new cluster")
    print("4. Copy your cluster URL and API key")
    print("5. Note: Free tier includes 25GB persistent storage")
    
    print("\n🔑 Enter your Weaviate Cloud credentials:")
    
    url = input("Cluster URL (e.g., https://your-cluster.weaviate.network): ").strip()
    api_key = input("API Key: ").strip()
    
    if not url or not api_key:
        print("❌ URL and API key are required!")
        return False
    
    # Test connection
    print("\n🔍 Testing connection...")
    if test_weaviate_cloud_connection(url, api_key):
        print("✅ Connection successful!")
        
        # Create persistent collection
        print("\n📝 Creating persistent collection...")
        if create_persistent_collection(url, api_key):
            print("✅ Collection created successfully")
            
            # Test persistent storage
            print("\n🧪 Testing persistent storage...")
            if test_persistent_storage(url, api_key):
                print("✅ Persistent storage working correctly")
                
                # Update configuration
                print("\n⚙️ Updating configuration...")
                if update_config_with_weaviate_cloud(url, api_key):
                    if setup_weaviate_manager_for_cloud():
                        print("\n🎉 Weaviate Cloud setup completed successfully!")
                        print("\n📋 Next steps:")
                        print("1. Get other API keys for full functionality:")
                        print("   - FriendliAI: https://console.friendli.ai/")
                        print("   - OpenAI: https://platform.openai.com/api-keys")
                        print("2. Update config.env with those keys")
                        print("3. Run: streamlit run streamlit_ui.py")
                        print("4. Open: http://localhost:8501")
                        print("\n💾 Your data will now be persistently stored in Weaviate Cloud!")
                        return True
            else:
                print("❌ Persistent storage test failed")
                return False
        else:
            print("❌ Collection creation failed")
            return False
    else:
        print("❌ Connection failed! Please check your credentials.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Alternative: Continue using local Weaviate with Docker")
        print("   Run: python fix_setup.py") 