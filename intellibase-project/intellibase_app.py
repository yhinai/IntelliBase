#!/usr/bin/env python3
"""
Main IntelliBase application orchestrating all components
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our components
from daft_processor import DaftProcessor
from weaviate_manager import WeaviateManager, SearchResult
from friendliai_integration import FriendliAIIntegration, GenerationResponse
from phoenix_observability import ObservabilityManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelliBaseApp:
    """Main IntelliBase application orchestrating all sponsor technologies"""
    
    def __init__(self):
        logger.info("🚀 Initializing IntelliBase...")
        
        # Initialize all components
        self.daft_processor = DaftProcessor()
        self.weaviate_manager = WeaviateManager()
        self.friendli_integration = FriendliAIIntegration()
        self.obs_manager = ObservabilityManager()
        
        # Application state
        self.is_initialized = False
        self.document_count = 0
        
        # Setup system
        self.setup_system()
    
    def setup_system(self):
        """Initialize all system components"""
        
        try:
            # Setup vector database
            self.weaviate_manager.create_collection()
            
            # Mark as initialized
            self.is_initialized = True
            
            logger.info("✅ IntelliBase initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize IntelliBase: {e}")
            raise
    
    @property
    def system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        stats = self.weaviate_manager.get_collection_stats()
        
        return {
            "initialized": self.is_initialized,
            "documents_indexed": stats.get("total_documents", 0),
            "weaviate_status": "connected",
            "friendli_status": "connected",
            "phoenix_url": self.obs_manager.get_phoenix_url(),
            "using_mock_friendli": self.friendli_integration.is_using_mock(),
            "metrics": self.obs_manager.get_metrics()
        }
    
    def ingest_documents(self, data_path: str = "./sample_data/*") -> int:
        """Complete document ingestion pipeline"""
        
        logger.info(f"📊 Starting document ingestion from: {data_path}")
        
        try:
            with self.obs_manager.trace_data_processing() as span:
                # Process with Daft
                df = self.daft_processor.process_multimodal_data(data_path)
                processed_df = self.daft_processor.process_for_vector_db(df)
                
                span.set_attribute("documents_processed", len(processed_df))
                
                # Index in Weaviate
                self.weaviate_manager.batch_insert_from_dataframe(processed_df)
                
                # Update document count
                self.document_count = len(processed_df)
                
                logger.info(f"✅ Successfully ingested {self.document_count} document chunks")
                return self.document_count
                
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Complete query processing pipeline"""
        
        start_time = time.time()
        logger.info(f"🤔 Processing query: {user_query}")
        
        try:
            # 1. Search knowledge base
            with self.obs_manager.trace_search_operation(user_query) as search_span:
                search_results = self.weaviate_manager.hybrid_search(user_query, limit=5)
                
                search_span.set_attribute("result_count", len(search_results))
                if search_results:
                    avg_score = sum(r.score for r in search_results) / len(search_results)
                    search_span.set_attribute("average_score", avg_score)
            
            # Extract context from search results
            context = [result.content for result in search_results[:3]]  # Top 3 results
            context_text = "\n".join(context)
            
            # 2. Generate response with FriendliAI
            with self.obs_manager.trace_llm_generation(
                user_query, len(context_text), 0, "meta-llama-3.1-8b-instruct"
            ) as llm_span:
                generation_response = self.friendli_integration.generate_with_context(
                    user_query, context
                )
                
                llm_span.set_attribute("response_length", len(generation_response.content))
                llm_span.set_attribute("token_count", generation_response.token_count or 0)
            
            # 3. Evaluate response quality
            evaluation = self.obs_manager.evaluate_response_quality(
                user_query, generation_response.content, context_text
            )
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            self.obs_manager.update_metrics(processing_time, True)
            
            # Prepare response
            result = {
                "query": user_query,
                "answer": generation_response.content,
                "sources": [
                    {
                        "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        "source": result.source_file,
                        "score": result.score,
                        "chunk_id": result.chunk_id,
                        "content_type": result.content_type
                    }
                    for result in search_results
                ],
                "evaluation": evaluation,
                "processing_time": processing_time,
                "generation_time": generation_response.processing_time,
                "phoenix_url": self.obs_manager.get_phoenix_url(),
                "model_used": generation_response.model,
                "using_mock": self.friendli_integration.is_using_mock(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Log the complete query event
            self.obs_manager.log_query_event(
                user_query, 
                generation_response.content, 
                processing_time,
                [r.source_file for r in search_results]
            )
            
            logger.info(f"✅ Query processed successfully in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            # Update metrics for failed query
            processing_time = time.time() - start_time
            self.obs_manager.update_metrics(processing_time, False)
            
            logger.error(f"Query processing failed: {e}")
            raise
    
    def search_documents(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search documents without generation"""
        
        logger.info(f"🔍 Searching for: {query}")
        
        with self.obs_manager.trace_search_operation(query):
            results = self.weaviate_manager.hybrid_search(query, limit=limit)
        
        logger.info(f"Found {len(results)} results")
        return results
    
    async def batch_process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently"""
        
        logger.info(f"📝 Processing {len(queries)} queries in batch")
        
        # Process queries concurrently
        tasks = [self.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i+1} failed: {result}")
            else:
                valid_results.append(result)
        
        logger.info(f"✅ Batch processing complete: {len(valid_results)}/{len(queries)} successful")
        return valid_results
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get system analytics and metrics"""
        
        system_status = self.system_status
        weaviate_stats = self.weaviate_manager.get_collection_stats()
        phoenix_metrics = self.obs_manager.get_metrics()
        
        return {
            "system_health": {
                "status": "healthy" if self.is_initialized else "initializing",
                "uptime": "N/A",  # Would track actual uptime in production
                "components": {
                    "daft_processor": "active",
                    "weaviate": "connected",
                    "friendli": "connected" if not self.friendli_integration.is_using_mock() else "mock",
                    "phoenix": "active"
                }
            },
            "data_statistics": {
                "total_documents": weaviate_stats.get("total_documents", 0),
                "content_types": weaviate_stats.get("content_types", []),
                "average_content_length": weaviate_stats.get("average_content_length", 0)
            },
            "performance_metrics": phoenix_metrics,
            "phoenix_dashboard": self.obs_manager.get_phoenix_url()
        }
    
    def close(self):
        """Cleanup resources"""
        logger.info("🔥 Shutting down IntelliBase...")
        
        try:
            # Close observability
            self.obs_manager.close()
            
            logger.info("✅ IntelliBase shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def test_intellibase_app():
    """Test the complete IntelliBase application"""
    
    logger.info("🧪 Testing IntelliBase application...")
    
    # Initialize app
    app = IntelliBaseApp()
    
    # Test document ingestion
    logger.info("\n1. Testing document ingestion...")
    doc_count = app.ingest_documents("./sample_data/*")
    logger.info(f"   ✅ Ingested {doc_count} documents")
    
    # Test system status
    logger.info("\n2. Testing system status...")
    status = app.system_status
    logger.info(f"   ✅ System status: {status['documents_indexed']} docs indexed")
    
    # Test single query
    logger.info("\n3. Testing single query...")
    result = await app.process_query("What is machine learning?")
    logger.info(f"   ✅ Query response: {result['answer'][:100]}...")
    logger.info(f"   📊 Processing time: {result['processing_time']:.3f}s")
    logger.info(f"   🎯 Quality score: {result['evaluation']['overall_quality']:.2f}")
    
    # Test search without generation
    logger.info("\n4. Testing document search...")
    search_results = app.search_documents("vector database", limit=3)
    logger.info(f"   ✅ Found {len(search_results)} search results")
    
    # Test batch queries
    logger.info("\n5. Testing batch query processing...")
    test_queries = [
        "How do vector databases work?",
        "What is RAG?",
        "What are the benefits of observability?"
    ]
    batch_results = await app.batch_process_queries(test_queries)
    logger.info(f"   ✅ Processed {len(batch_results)} queries in batch")
    
    # Test analytics
    logger.info("\n6. Testing analytics...")
    analytics = app.get_analytics()
    logger.info(f"   ✅ Analytics: {analytics['system_health']['status']}")
    logger.info(f"   📈 Total queries: {analytics['performance_metrics']['total_queries']}")
    
    # Cleanup
    app.close()
    
    logger.info("\n🎉 IntelliBase application test complete!")
    return True

def main():
    """Test the IntelliBase application"""
    
    try:
        success = asyncio.run(test_intellibase_app())
        
        if success:
            print("✅ IntelliBase application test successful!")
            return 0
        else:
            print("❌ IntelliBase application test failed!")
            return 1
            
    except Exception as e:
        print(f"❌ IntelliBase application test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())