#!/usr/bin/env python3
"""
Comprehensive system testing for IntelliBase
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Any
from pathlib import Path

# Import all components for testing
from daft_processor import DaftProcessor
from weaviate_manager import WeaviateManager
from friendliai_integration import FriendliAIIntegration
from phoenix_observability import ObservabilityManager
from intellibase_app import IntelliBaseApp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive system testing suite"""
    
    def __init__(self):
        self.test_results = {
            "components": {},
            "integration": {},
            "performance": {},
            "summary": {}
        }
        self.start_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests"""
        
        logger.info("🧪 Starting comprehensive system tests...")
        self.start_time = time.time()
        
        # Component tests
        self.test_daft_processor()
        self.test_weaviate_manager()
        self.test_friendliai_integration()
        self.test_phoenix_observability()
        
        # Integration tests
        asyncio.run(self.test_integration())
        
        # Performance tests
        asyncio.run(self.test_performance())
        
        # Generate summary
        self.generate_summary()
        
        return self.test_results
    
    def test_daft_processor(self):
        """Test Daft data processor"""
        
        logger.info("Testing Daft processor...")
        
        try:
            processor = DaftProcessor()
            
            # Test data processing
            df = processor.process_multimodal_data("./sample_data/*")
            processed_df = processor.process_for_vector_db(df)
            
            # Validate results
            assert len(processed_df) > 0, "No data processed"
            assert "content" in processed_df.columns, "Missing content column"
            assert "chunk_id" in processed_df.columns, "Missing chunk_id column"
            
            self.test_results["components"]["daft"] = {
                "status": "✅ PASS",
                "documents_processed": len(df),
                "chunks_created": len(processed_df),
                "content_types": df["content_type"].unique().tolist(),
                "avg_chunk_size": processed_df["content"].str.len().mean()
            }
            
            logger.info("✅ Daft processor test passed")
            
        except Exception as e:
            self.test_results["components"]["daft"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
            logger.error(f"❌ Daft processor test failed: {e}")
    
    def test_weaviate_manager(self):
        """Test Weaviate manager"""
        
        logger.info("Testing Weaviate manager...")
        
        try:
            manager = WeaviateManager()
            
            # Test connection and collection creation
            manager.create_collection()
            
            # Test data insertion
            test_data = self._create_test_data()
            manager.batch_insert_from_dataframe(test_data)
            
            # Test search
            results = manager.hybrid_search("machine learning", limit=3)
            
            # Test stats
            stats = manager.get_collection_stats()
            
            # Validate results
            assert len(results) > 0, "No search results"
            assert stats["total_documents"] > 0, "No documents in collection"
            
            self.test_results["components"]["weaviate"] = {
                "status": "✅ PASS",
                "documents_indexed": stats["total_documents"],
                "search_results": len(results),
                "avg_score": sum(r.score for r in results) / len(results) if results else 0,
                "using_mock": hasattr(manager, 'mock_manager')
            }
            
            logger.info("✅ Weaviate manager test passed")
            
        except Exception as e:
            self.test_results["components"]["weaviate"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
            logger.error(f"❌ Weaviate manager test failed: {e}")
    
    def test_friendliai_integration(self):
        """Test FriendliAI integration"""
        
        logger.info("Testing FriendliAI integration...")
        
        try:
            integration = FriendliAIIntegration()
            
            # Test generation
            response = integration.generate_with_context(
                "What is machine learning?",
                ["Machine learning is a subset of AI", "It learns from data"]
            )
            
            # Validate response
            assert response.content, "Empty response content"
            assert response.processing_time >= 0, "Invalid processing time"
            assert response.model, "Missing model information"
            
            self.test_results["components"]["friendliai"] = {
                "status": "✅ PASS",
                "response_length": len(response.content),
                "processing_time": response.processing_time,
                "model": response.model,
                "token_count": response.token_count,
                "using_mock": integration.is_using_mock()
            }
            
            logger.info("✅ FriendliAI integration test passed")
            
        except Exception as e:
            self.test_results["components"]["friendliai"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
            logger.error(f"❌ FriendliAI integration test failed: {e}")
    
    def test_phoenix_observability(self):
        """Test Phoenix observability"""
        
        logger.info("Testing Phoenix observability...")
        
        try:
            obs_manager = ObservabilityManager()
            
            # Test tracing
            with obs_manager.trace_data_processing(100):
                time.sleep(0.01)
            
            with obs_manager.trace_search_operation("test query", 5, 0.85):
                time.sleep(0.01)
            
            with obs_manager.trace_llm_generation("test", 200, 150, "llama"):
                time.sleep(0.01)
            
            # Test metrics
            obs_manager.update_metrics(0.1, True)
            metrics = obs_manager.get_metrics()
            
            # Test evaluation
            evaluation = obs_manager.evaluate_response_quality(
                "test query", "test response", "test context"
            )
            
            # Validate
            assert metrics["total_queries"] > 0, "No metrics recorded"
            assert evaluation["overall_quality"] > 0, "Invalid evaluation"
            
            self.test_results["components"]["phoenix"] = {
                "status": "✅ PASS",
                "metrics_recorded": metrics["total_queries"],
                "avg_response_time": metrics["average_response_time"],
                "phoenix_url": obs_manager.get_phoenix_url(),
                "evaluation_score": evaluation["overall_quality"]
            }
            
            logger.info("✅ Phoenix observability test passed")
            
        except Exception as e:
            self.test_results["components"]["phoenix"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
            logger.error(f"❌ Phoenix observability test failed: {e}")
    
    async def test_integration(self):
        """Test full system integration"""
        
        logger.info("Testing system integration...")
        
        try:
            # Initialize application
            app = IntelliBaseApp()
            
            # Test document ingestion
            doc_count = app.ingest_documents("./sample_data/*")
            assert doc_count > 0, "No documents ingested"
            
            # Test single query
            result = await app.process_query("What is machine learning?")
            assert result["answer"], "Empty answer"
            assert result["sources"], "No sources found"
            
            # Test batch queries
            test_queries = [
                "How do vector databases work?",
                "What is RAG?",
                "What are the benefits of observability?"
            ]
            batch_results = await app.batch_process_queries(test_queries)
            assert len(batch_results) == len(test_queries), "Batch processing failed"
            
            # Test system status
            status = app.system_status
            assert status["initialized"], "System not initialized"
            
            # Test analytics
            analytics = app.get_analytics()
            assert analytics["system_health"]["status"] == "healthy", "System not healthy"
            
            self.test_results["integration"]["full_system"] = {
                "status": "✅ PASS",
                "documents_ingested": doc_count,
                "single_query_time": result["processing_time"],
                "batch_queries_processed": len(batch_results),
                "system_health": analytics["system_health"]["status"],
                "total_system_queries": analytics["performance_metrics"]["total_queries"]
            }
            
            logger.info("✅ System integration test passed")
            
        except Exception as e:
            self.test_results["integration"]["full_system"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
            logger.error(f"❌ System integration test failed: {e}")
    
    async def test_performance(self):
        """Test system performance"""
        
        logger.info("Testing system performance...")
        
        try:
            app = IntelliBaseApp()
            
            # Ensure we have data
            app.ingest_documents("./sample_data/*")
            
            # Performance test queries
            test_queries = [
                "What is machine learning?",
                "How do vector databases work?",
                "What is RAG?",
                "What are transformers?",
                "How does Daft process data?"
            ]
            
            # Time individual queries
            query_times = []
            for query in test_queries:
                start_time = time.time()
                await app.process_query(query)
                query_times.append(time.time() - start_time)
            
            # Batch performance test
            batch_start = time.time()
            batch_results = await app.batch_process_queries(test_queries)
            batch_time = time.time() - batch_start
            
            # Calculate metrics
            avg_query_time = sum(query_times) / len(query_times)
            min_query_time = min(query_times)
            max_query_time = max(query_times)
            
            self.test_results["performance"]["queries"] = {
                "status": "✅ PASS",
                "total_queries_tested": len(test_queries),
                "avg_query_time": avg_query_time,
                "min_query_time": min_query_time,
                "max_query_time": max_query_time,
                "batch_time": batch_time,
                "batch_efficiency": batch_time / (avg_query_time * len(test_queries)),
                "queries_per_second": len(test_queries) / batch_time
            }
            
            logger.info("✅ Performance test passed")
            logger.info(f"   Average query time: {avg_query_time:.3f}s")
            logger.info(f"   Queries per second: {len(test_queries) / batch_time:.1f}")
            
        except Exception as e:
            self.test_results["performance"]["queries"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
            logger.error(f"❌ Performance test failed: {e}")
    
    def generate_summary(self):
        """Generate test summary"""
        
        total_time = time.time() - self.start_time
        
        # Count passed/failed tests
        all_tests = []
        for category in ["components", "integration", "performance"]:
            for test_name, result in self.test_results[category].items():
                all_tests.append(result["status"])
        
        passed = sum(1 for status in all_tests if "✅" in status)
        failed = sum(1 for status in all_tests if "❌" in status)
        total = len(all_tests)
        
        self.test_results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_time": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Total tests: {total}")
        logger.info(f"   Passed: {passed}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Success rate: {passed/total*100:.1f}%")
        logger.info(f"   Total time: {total_time:.2f}s")
    
    def _create_test_data(self):
        """Create test data for components"""
        
        import pandas as pd
        from datetime import datetime
        
        return pd.DataFrame([
            {
                "content": "Machine learning is a subset of artificial intelligence",
                "content_type": "text",
                "path": "test_ml.txt",
                "chunk_id": "test_001",
                "file_size": 100,
                "processed_at": datetime.now().isoformat(),
                "chunk_index": 0
            },
            {
                "content": "Vector databases store embeddings for semantic search",
                "content_type": "text",
                "path": "test_vector.txt",
                "chunk_id": "test_002", 
                "file_size": 120,
                "processed_at": datetime.now().isoformat(),
                "chunk_index": 0
            }
        ])

def run_comprehensive_tests():
    """Run all system tests"""
    
    print("🚀 Starting IntelliBase comprehensive system tests...")
    print("=" * 60)
    
    tester = SystemTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    # Print component test results
    print("\n🔧 COMPONENT TESTS:")
    for component, result in results["components"].items():
        print(f"  {component.upper()}: {result['status']}")
        if "error" in result:
            print(f"    Error: {result['error']}")
    
    # Print integration test results  
    print("\n🔗 INTEGRATION TESTS:")
    for test, result in results["integration"].items():
        print(f"  {test.upper()}: {result['status']}")
        if "error" in result:
            print(f"    Error: {result['error']}")
    
    # Print performance test results
    print("\n⚡ PERFORMANCE TESTS:")
    for test, result in results["performance"].items():
        print(f"  {test.upper()}: {result['status']}")
        if "avg_query_time" in result:
            print(f"    Avg Query Time: {result['avg_query_time']:.3f}s")
            print(f"    Queries/Second: {result['queries_per_second']:.1f}")
    
    # Print summary
    summary = results["summary"]
    print(f"\n📈 OVERALL SUMMARY:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    print(f"  Total Time: {summary['total_time']:.2f}s")
    
    # Save detailed results
    results_file = Path("test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    print("=" * 60)
    
    return summary["failed"] == 0

def main():
    """Main test runner"""
    
    success = run_comprehensive_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! IntelliBase is ready for deployment.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED! Please check the results above.")
        return 1

if __name__ == "__main__":
    exit(main())