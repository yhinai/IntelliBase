#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - Weaviate Vector Database Manager
Vector database operations and hybrid search capabilities
"""

import os
from typing import List, Dict, Any, Optional
import time
import json
from pathlib import Path

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
    print("✅ Weaviate client available")
except ImportError:
    print("⚠️ Weaviate client not available - using mock implementation")
    WEAVIATE_AVAILABLE = False

try:
    from observability import trace_vector_search, obs_manager
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    print("⚠️ Observability not available - continuing without tracing")
    OBSERVABILITY_AVAILABLE = False
    
    # Create dummy decorators
    def trace_vector_search(**kwargs):
        def decorator(func):
            return func
        return decorator


class WeaviateManager:
    """Manages Weaviate vector database operations for IntelliBase"""
    
    def __init__(self, use_local: bool = True):
        self.client = None
        self.collection_name = "IntellibaseKnowledge"
        self.use_local = use_local
        self.weaviate_available = WEAVIATE_AVAILABLE
        self.data_store = []  # Fallback in-memory store
        
        if self.weaviate_available:
            self.connect()
        else:
            print("🔄 Using in-memory fallback storage")
        
        print(f"🔧 WeaviateManager initialized (Available: {self.weaviate_available})")
    
    def connect(self):
        """Connect to Weaviate instance"""
        if not self.weaviate_available:
            return False
            
        try:
            if self.use_local:
                # Try local Weaviate first
                print("🔌 Attempting to connect to local Weaviate...")
                self.client = weaviate.Client("http://localhost:8080")
            else:
                # Try Weaviate Cloud if configured
                cluster_url = os.getenv("WEAVIATE_CLUSTER_URL")
                api_key = os.getenv("WEAVIATE_API_KEY")
                
                if cluster_url and api_key:
                    print("🌐 Connecting to Weaviate Cloud...")
                    self.client = weaviate.Client(
                        url=cluster_url,
                        auth_client_secret=weaviate.Auth.api_key(api_key)
                    )
                else:
                    print("⚠️ Weaviate Cloud credentials not found in environment")
                    return False
            
            # Test connection
            if self.client and self.client.is_ready():
                print("✅ Connected to Weaviate successfully")
                return True
            else:
                print("❌ Weaviate connection failed")
                self.client = None
                return False
                
        except Exception as e:
            print(f"⚠️ Weaviate connection error: {e}")
            print("🔄 Falling back to in-memory storage")
            self.client = None
            return False
    
    def create_collection(self, reset: bool = True):
        """Create collection with hybrid search capabilities"""
        
        if not self.client:
            print("📦 Using in-memory collection (no Weaviate connection)")
            self.data_store = []
            return True
        
        try:
            # Delete existing collection if it exists and reset is True
            if reset and self.client.schema.exists(self.collection_name):
                print(f"🗑️ Deleting existing collection: {self.collection_name}")
                self.client.schema.delete_class(self.collection_name)
            
            # Create new collection
            print(f"📝 Creating collection: {self.collection_name}")
            
            class_obj = {
                "class": self.collection_name,
                "vectorizer": "none",  # Use no vectorizer for now
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The main text content"
                    },
                    {
                        "name": "content_type",
                        "dataType": ["text"],
                        "description": "Type of content (pdf, image, text)"
                    },
                    {
                        "name": "source_file",
                        "dataType": ["text"],
                        "description": "Original file name"
                    },
                    {
                        "name": "source_path",
                        "dataType": ["text"],
                        "description": "Original file path"
                    },
                    {
                        "name": "chunk_id",
                        "dataType": ["text"],
                        "description": "Unique chunk identifier"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Index of chunk within source file"
                    },
                    {
                        "name": "file_id",
                        "dataType": ["text"],
                        "description": "Source file identifier"
                    }
                ]
            }
            
            self.client.schema.create_class(class_obj)
            print(f"✅ Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            return False
    
    @trace_vector_search(operation="batch_insert")
    def batch_insert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Insert processed chunks into Weaviate"""
        
        if not chunks:
            print("⚠️ No chunks to insert")
            return False
        
        print(f"📥 Inserting {len(chunks)} chunks...")
        
        if not self.client:
            # Fallback storage
            self.data_store.extend(chunks)
            print(f"✅ Stored {len(chunks)} chunks in fallback storage")
            return True
        
        try:
            # Batch insert with error handling
            success_count = 0
            batch_size = 100
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                for chunk in batch:
                    try:
                        self.client.data_object.create(
                            class_name=self.collection_name,
                            data_object={
                                "content": chunk.get("content", ""),
                                "content_type": chunk.get("content_type", ""),
                                "source_file": chunk.get("source_file", ""),
                                "source_path": chunk.get("source_path", ""),
                                "chunk_id": chunk.get("chunk_id", ""),
                                "chunk_index": chunk.get("chunk_index", 0),
                                "file_id": chunk.get("file_id", "")
                            }
                        )
                        success_count += 1
                    except Exception as e:
                        print(f"⚠️ Error inserting chunk {chunk.get('chunk_id', 'unknown')}: {e}")
            
            print(f"✅ Successfully inserted {success_count}/{len(chunks)} chunks")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Batch insert failed: {e}")
            return False
    
    @trace_vector_search(operation="hybrid_search")
    def hybrid_search(self, query: str, limit: int = 5, alpha: float = 0.75) -> List[Dict[str, Any]]:
        """Perform search using BM25 (keyword search) since no vectorizer is available"""
        
        print(f"🔍 Searching for: '{query}' (limit: {limit})")
        
        if not self.client:
            return self._fallback_search(query, limit)
        
        try:
            response = self.client.query.get(
                self.collection_name, 
                ["content", "content_type", "source_file", "source_path", "chunk_id"]
            ).with_bm25(
                query=query
            ).with_limit(limit).do()
            
            results = []
            if "data" in response and "Get" in response["data"]:
                for obj in response["data"]["Get"][self.collection_name]:
                    results.append({
                        "content": obj.get("content", ""),
                        "content_type": obj.get("content_type", ""),
                        "source_file": obj.get("source_file", ""),
                        "source_path": obj.get("source_path", ""),
                        "chunk_id": obj.get("chunk_id", ""),
                        "score": 0.0,  # Score not available in basic search
                        "uuid": obj.get("id", "")
                    })
            
            print(f"🎯 Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return self._fallback_search(query, limit)
    
    def _fallback_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback search using simple text matching"""
        query_lower = query.lower()
        results = []
        
        for chunk in self.data_store:
            content = chunk.get("content", "").lower()
            if query_lower in content:
                # Simple scoring based on query frequency
                score = content.count(query_lower) / len(content.split()) if content else 0
                results.append({
                    **chunk,
                    "score": score,
                    "uuid": chunk.get("chunk_id", "")
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]
        
        print(f"🎯 Found {len(results)} results (fallback search)")
        return results
    
    @trace_vector_search(operation="get_stats")
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        
        if not self.client:
            return {
                "total_objects": len(self.data_store),
                "storage_type": "in_memory_fallback",
                "weaviate_available": False
            }
        
        try:
            # Get collection info
            schema = self.client.schema.get()
            collection_info = None
            
            for class_info in schema["classes"]:
                if class_info["class"] == self.collection_name:
                    collection_info = class_info
                    break
            
            if collection_info:
                # Get object count
                response = self.client.query.aggregate(self.collection_name).with_meta_count().do()
                total_objects = 0
                if "data" in response and "Aggregate" in response["data"]:
                    total_objects = response["data"]["Aggregate"][self.collection_name][0]["meta"]["count"]
                
                return {
                    "total_objects": total_objects,
                    "storage_type": "weaviate",
                    "weaviate_available": True,
                    "collection_name": self.collection_name,
                    "properties": len(collection_info.get("properties", []))
                }
            else:
                return {
                    "total_objects": 0,
                    "storage_type": "weaviate_no_collection",
                    "weaviate_available": True
                }
                
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                "total_objects": 0,
                "storage_type": "error",
                "weaviate_available": True,
                "error": str(e)
            }
    
    def export_data(self, output_file: str = "weaviate_export.json"):
        """Export all data from the collection"""
        
        if not self.client:
            # Export fallback data
            with open(output_file, 'w') as f:
                json.dump(self.data_store, f, indent=2)
            print(f"📤 Exported {len(self.data_store)} objects to {output_file}")
            return True
        
        try:
            response = self.client.query.get(
                self.collection_name,
                ["content", "content_type", "source_file", "source_path", "chunk_id", "chunk_index", "file_id"]
            ).with_limit(10000).do()
            
            if "data" in response and "Get" in response["data"]:
                data = response["data"]["Get"][self.collection_name]
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"📤 Exported {len(data)} objects to {output_file}")
                return True
            else:
                print("⚠️ No data found to export")
                return False
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    def close(self):
        """Close Weaviate connection"""
        if self.client:
            try:
                self.client.close()
                print("🔌 Weaviate connection closed")
            except:
                pass


if __name__ == "__main__":
    # Test the Weaviate manager
    print("🚀 Testing IntelliBase Weaviate Manager")
    print("=" * 50)
    
    # Initialize manager
    manager = WeaviateManager(use_local=True)
    
    # Create collection
    print("\n📝 Creating collection...")
    collection_created = manager.create_collection()
    
    if collection_created:
        # Test with sample data
        print("\n📊 Testing with sample chunks...")
        
        # Load some sample data using our processor
        try:
            from daft_processor_simple import IntelliBaseDataProcessor
            processor = IntelliBaseDataProcessor()
            processed_files = processor.process_directory("./sample_data")
            chunks = processor.create_chunks(processed_files, chunk_size=500)
            
            if chunks:
                # Insert chunks
                print(f"\n📥 Inserting {len(chunks)} chunks...")
                insert_success = manager.batch_insert_chunks(chunks)
                
                if insert_success:
                    # Test search
                    print("\n🔍 Testing search functionality...")
                    
                    test_queries = [
                        "machine learning",
                        "vector database",
                        "AI system",
                        "processing"
                    ]
                    
                    for query in test_queries:
                        results = manager.hybrid_search(query, limit=3)
                        print(f"\nQuery: '{query}' -> {len(results)} results")
                        for i, result in enumerate(results[:2]):
                            print(f"  {i+1}. {result['source_file']} (score: {result['score']:.3f})")
                            print(f"     Content: {result['content'][:80]}...")
                    
                    # Show statistics
                    stats = manager.get_collection_stats()
                    print(f"\n📊 Collection Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    
                    # Export data
                    print(f"\n💾 Exporting data...")
                    manager.export_data("test_export.json")
            
        except ImportError:
            print("⚠️ Data processor not available - using mock data")
            
            # Create mock chunks for testing
            mock_chunks = [
                {
                    "chunk_id": "test_1",
                    "content": "Machine learning is a subset of artificial intelligence",
                    "content_type": "text",
                    "source_file": "test.txt",
                    "source_path": "/test/test.txt",
                    "chunk_index": 0,
                    "file_id": "test_file_1"
                },
                {
                    "chunk_id": "test_2", 
                    "content": "Vector databases enable semantic search capabilities",
                    "content_type": "text",
                    "source_file": "test.txt",
                    "source_path": "/test/test.txt",
                    "chunk_index": 1,
                    "file_id": "test_file_1"
                }
            ]
            
            manager.batch_insert_chunks(mock_chunks)
            results = manager.hybrid_search("machine learning", limit=2)
            print(f"Mock search results: {len(results)}")
    
    # Clean up
    manager.close()
    print("\n✅ Weaviate manager test complete!") 