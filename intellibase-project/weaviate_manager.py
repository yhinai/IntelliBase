#!/usr/bin/env python3
"""
Weaviate manager for IntelliBase vector database operations
"""
import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result from vector database"""
    content: str
    source_file: str
    score: float
    chunk_id: str
    content_type: str
    metadata: Dict[str, Any]

class MockWeaviateManager:
    """Mock Weaviate manager for testing without actual Weaviate instance"""
    
    def __init__(self):
        self.collection_name = "IntellibaseKnowledge"
        self.data_store: List[Dict[str, Any]] = []
        self.connected = False
        self.connect()
    
    def connect(self):
        """Mock connection to Weaviate"""
        logger.info("Connecting to mock Weaviate instance...")
        self.connected = True
        logger.info("✅ Connected to mock Weaviate successfully")
    
    def create_collection(self):
        """Mock collection creation"""
        logger.info(f"Creating collection '{self.collection_name}'...")
        
        # Reset data store
        self.data_store = []
        
        logger.info(f"✅ Collection '{self.collection_name}' created successfully")
    
    def batch_insert_from_dataframe(self, df: pd.DataFrame):
        """Insert data from DataFrame into mock storage"""
        
        logger.info("Starting batch insert...")
        
        for _, row in df.iterrows():
            document = {
                "content": row["content"],
                "content_type": row["content_type"],
                "source_file": row["path"],
                "chunk_id": row["chunk_id"],
                "file_size": row["file_size"],
                "processed_at": row["processed_at"],
                "chunk_index": row.get("chunk_index", 0),
                # Mock embedding - in real implementation, this would be generated
                "embedding": np.random.random(384).tolist()  # Mock 384-dim embedding
            }
            self.data_store.append(document)
        
        logger.info(f"✅ Batch insert completed for {len(df)} items")
    
    def hybrid_search(self, query: str, limit: int = 5, alpha: float = 0.75) -> List[SearchResult]:
        """Perform mock hybrid search (vector + keyword)"""
        
        logger.info(f"Performing hybrid search for: '{query}'")
        
        if not self.data_store:
            logger.warning("No data in store for search")
            return []
        
        # Mock search - in real implementation, this would use actual embeddings
        results = []
        query_lower = query.lower()
        
        for doc in self.data_store:
            # Simple keyword matching for mock
            content_lower = doc["content"].lower()
            score = 0.0
            
            # Calculate mock similarity score
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            
            if query_words & content_words:  # If there's word overlap
                score = len(query_words & content_words) / len(query_words)
                score = min(score, 0.95)  # Cap at 0.95
            
            if score > 0.1:  # Threshold for inclusion
                results.append(SearchResult(
                    content=doc["content"],
                    source_file=doc["source_file"],
                    score=score,
                    chunk_id=doc["chunk_id"],
                    content_type=doc["content_type"],
                    metadata={
                        "file_size": doc["file_size"],
                        "processed_at": doc["processed_at"],
                        "chunk_index": doc["chunk_index"]
                    }
                ))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:limit]
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        return {
            "total_documents": len(self.data_store),
            "content_types": list(set(doc["content_type"] for doc in self.data_store)),
            "total_chunks": len(self.data_store),
            "average_content_length": np.mean([len(doc["content"]) for doc in self.data_store]) if self.data_store else 0
        }

class WeaviateManager:
    """Real Weaviate manager (requires actual Weaviate instance)"""
    
    def __init__(self):
        self.client = None
        self.collection_name = "IntellibaseKnowledge"
        self.connected = False
        
        # Check if we should use real Weaviate
        if os.getenv("WEAVIATE_CLUSTER_URL") and os.getenv("WEAVIATE_API_KEY"):
            self.connect_real()
        else:
            logger.warning("No Weaviate credentials found, falling back to mock")
            self.mock_manager = MockWeaviateManager()
    
    def connect_real(self):
        """Connect to real Weaviate instance"""
        try:
            import weaviate
            from weaviate.classes.init import Auth
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
                auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
            )
            
            self.connected = True
            logger.info("✅ Connected to real Weaviate successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to real Weaviate: {e}")
            logger.info("Falling back to mock implementation")
            self.mock_manager = MockWeaviateManager()
    
    def create_collection(self):
        """Create collection in Weaviate"""
        if hasattr(self, 'mock_manager'):
            return self.mock_manager.create_collection()
        
        # Real Weaviate implementation would go here
        pass
    
    def batch_insert_from_dataframe(self, df: pd.DataFrame):
        """Insert data from DataFrame"""
        if hasattr(self, 'mock_manager'):
            return self.mock_manager.batch_insert_from_dataframe(df)
        
        # Real Weaviate implementation would go here
        pass
    
    def hybrid_search(self, query: str, limit: int = 5, alpha: float = 0.75) -> List[SearchResult]:
        """Perform hybrid search"""
        if hasattr(self, 'mock_manager'):
            return self.mock_manager.hybrid_search(query, limit, alpha)
        
        # Real Weaviate implementation would go here
        return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if hasattr(self, 'mock_manager'):
            return self.mock_manager.get_collection_stats()
        
        # Real Weaviate implementation would go here
        return {}

def test_weaviate_manager():
    """Test the Weaviate manager"""
    
    logger.info("Testing Weaviate manager...")
    
    # Initialize manager
    manager = WeaviateManager()
    
    # Create collection
    manager.create_collection()
    
    # Create test data
    test_data = pd.DataFrame([
        {
            "content": "Machine learning is a subset of artificial intelligence",
            "content_type": "text",
            "path": "test_file_1.txt",
            "chunk_id": "chunk_001",
            "file_size": 100,
            "processed_at": datetime.now().isoformat(),
            "chunk_index": 0
        },
        {
            "content": "Vector databases store embeddings for semantic search",
            "content_type": "text", 
            "path": "test_file_2.txt",
            "chunk_id": "chunk_002",
            "file_size": 120,
            "processed_at": datetime.now().isoformat(),
            "chunk_index": 0
        },
        {
            "content": "RAG combines retrieval with generation for better accuracy",
            "content_type": "text",
            "path": "test_file_3.txt", 
            "chunk_id": "chunk_003",
            "file_size": 80,
            "processed_at": datetime.now().isoformat(),
            "chunk_index": 0
        }
    ])
    
    # Insert test data
    manager.batch_insert_from_dataframe(test_data)
    
    # Test search
    results = manager.hybrid_search("machine learning", limit=3)
    
    logger.info("Search results:")
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. Score: {result.score:.3f} - {result.content[:50]}...")
    
    # Get stats
    stats = manager.get_collection_stats()
    logger.info(f"Collection stats: {stats}")
    
    return len(results) > 0

def main():
    """Test the Weaviate manager"""
    success = test_weaviate_manager()
    
    if success:
        print("✅ Weaviate manager test successful!")
        return 0
    else:
        print("❌ Weaviate manager test failed!")
        return 1

if __name__ == "__main__":
    exit(main())