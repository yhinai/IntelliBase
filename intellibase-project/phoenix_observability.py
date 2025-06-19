#!/usr/bin/env python3
"""
Arize Phoenix observability setup for IntelliBase
"""
import os
import logging
import time
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockPhoenixSession:
    """Mock Phoenix session for testing without actual Phoenix"""
    
    def __init__(self):
        self.url = "http://localhost:6006"
        self.active = True
        logger.info(f"Mock Phoenix session started at {self.url}")
    
    def close(self):
        self.active = False
        logger.info("Mock Phoenix session closed")

class MockTracer:
    """Mock tracer for testing without actual OpenTelemetry"""
    
    def __init__(self):
        self.spans = []
    
    def start_as_current_span(self, operation_name: str):
        return MockSpan(operation_name, self)

class MockSpan:
    """Mock span for testing"""
    
    def __init__(self, operation_name: str, tracer: MockTracer):
        self.operation_name = operation_name
        self.tracer = tracer
        self.attributes = {}
        self.start_time = time.time()
        self.end_time = None
        
    def __enter__(self):
        logger.debug(f"Started span: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.attributes["duration_ms"] = duration * 1000
        
        logger.debug(f"Ended span: {self.operation_name} (duration: {duration:.3f}s)")
        self.tracer.spans.append(self)
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

class ObservabilityManager:
    """Manages observability for IntelliBase"""
    
    def __init__(self):
        self.session: Optional[MockPhoenixSession] = None
        self.tracer: Optional[MockTracer] = None
        self.metrics = {
            "total_queries": 0,
            "total_processing_time": 0.0,
            "average_response_time": 0.0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        self.setup_observability()
    
    def setup_observability(self):
        """Setup Phoenix and tracing"""
        
        try:
            # Try to setup real Phoenix
            self._setup_real_phoenix()
        except Exception as e:
            logger.warning(f"Could not setup real Phoenix: {e}")
            logger.info("Using mock observability")
            self._setup_mock_observability()
    
    def _setup_real_phoenix(self):
        """Setup real Phoenix (requires phoenix package)"""
        try:
            import phoenix as px
            from opentelemetry import trace
            
            # Start Phoenix session
            self.session = px.launch_app()
            logger.info(f"Phoenix UI available at: {self.session.url}")
            
            # Setup tracing
            self.tracer = trace.get_tracer(__name__)
            
            # Try to setup instrumentation
            try:
                from openinference.instrumentation.openai import OpenAIInstrumentor
                OpenAIInstrumentor().instrument()
                logger.info("OpenAI instrumentation enabled")
            except ImportError:
                logger.warning("OpenAI instrumentation not available")
            
            logger.info("✅ Real Phoenix observability setup complete")
            
        except ImportError:
            raise Exception("Phoenix package not available")
    
    def _setup_mock_observability(self):
        """Setup mock observability for testing"""
        self.session = MockPhoenixSession()
        self.tracer = MockTracer()
        logger.info("✅ Mock observability setup complete")
    
    def trace_operation(self, operation_name: str):
        """Decorator for tracing operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("timestamp", datetime.now().isoformat())
                    
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error_message", str(e))
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("duration_seconds", duration)
            return wrapper
        return decorator
    
    def trace_data_processing(self, record_count: int = 0):
        """Trace data processing operations"""
        with self.tracer.start_as_current_span("data_processing") as span:
            span.set_attribute("record_count", record_count)
            span.set_attribute("operation_type", "multimodal_processing")
            return span
    
    def trace_search_operation(self, query: str, result_count: int = 0, avg_score: float = 0.0):
        """Trace vector search operations"""
        with self.tracer.start_as_current_span("vector_search") as span:
            span.set_attribute("query", query)
            span.set_attribute("query_length", len(query))
            span.set_attribute("result_count", result_count)
            span.set_attribute("average_score", avg_score)
            return span
    
    def trace_llm_generation(self, query: str, context_length: int, response_length: int, model: str):
        """Trace LLM generation operations"""
        with self.tracer.start_as_current_span("llm_generation") as span:
            span.set_attribute("query", query)
            span.set_attribute("context_length", context_length)
            span.set_attribute("response_length", response_length)
            span.set_attribute("model", model)
            return span
    
    def update_metrics(self, processing_time: float, success: bool = True):
        """Update system metrics"""
        self.metrics["total_queries"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
        
        # Calculate average
        if self.metrics["total_queries"] > 0:
            self.metrics["average_response_time"] = (
                self.metrics["total_processing_time"] / self.metrics["total_queries"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def get_phoenix_url(self) -> str:
        """Get Phoenix dashboard URL"""
        return self.session.url if self.session else "http://localhost:6006"
    
    def evaluate_response_quality(self, query: str, response: str, context: str) -> Dict[str, float]:
        """Mock response quality evaluation"""
        
        # Simple heuristic evaluation
        relevance_score = min(5.0, len(set(query.lower().split()) & set(response.lower().split())) / len(query.split()) * 5)
        
        # Check if response uses context
        context_usage = min(5.0, len(set(context.lower().split()) & set(response.lower().split())) / max(len(context.split()), 1) * 5)
        
        # Length appropriateness (not too short, not too long)
        length_score = 5.0 if 50 <= len(response) <= 500 else 3.0
        
        overall_quality = (relevance_score + context_usage + length_score) / 3
        
        return {
            "relevance_score": relevance_score,
            "context_usage_score": context_usage,
            "length_score": length_score,
            "overall_quality": overall_quality
        }
    
    def log_query_event(self, query: str, response: str, processing_time: float, sources: list):
        """Log a complete query event"""
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_length": len(response),
            "processing_time": processing_time,
            "source_count": len(sources),
            "phoenix_url": self.get_phoenix_url()
        }
        
        logger.info(f"Query event logged: {json.dumps(event, indent=2)}")
    
    def close(self):
        """Close observability session"""
        if self.session:
            self.session.close()

def test_observability():
    """Test the observability setup"""
    
    logger.info("Testing observability setup...")
    
    # Initialize observability
    obs_manager = ObservabilityManager()
    
    # Test tracing decorators
    @obs_manager.trace_operation("test_function")
    def test_function(data: str):
        time.sleep(0.1)  # Simulate work
        return f"Processed: {data}"
    
    # Test various traces
    result = test_function("test data")
    logger.info(f"Test function result: {result}")
    
    # Test direct tracing
    with obs_manager.trace_data_processing(100):
        time.sleep(0.05)
    
    with obs_manager.trace_search_operation("test query", 5, 0.85):
        time.sleep(0.03)
    
    with obs_manager.trace_llm_generation("test query", 200, 150, "llama-3.1-8b"):
        time.sleep(0.02)
    
    # Test metrics
    obs_manager.update_metrics(0.15, True)
    obs_manager.update_metrics(0.12, True)
    obs_manager.update_metrics(0.20, False)
    
    metrics = obs_manager.get_metrics()
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    # Test evaluation
    evaluation = obs_manager.evaluate_response_quality(
        "What is machine learning?",
        "Machine learning is a subset of AI that learns from data",
        "Machine learning involves algorithms and data processing"
    )
    logger.info(f"Quality evaluation: {json.dumps(evaluation, indent=2)}")
    
    # Test event logging
    obs_manager.log_query_event(
        "test query",
        "test response",
        0.15,
        ["source1.txt", "source2.txt"]
    )
    
    return True

def main():
    """Test the observability setup"""
    success = test_observability()
    
    if success:
        print("✅ Phoenix observability test successful!")
        return 0
    else:
        print("❌ Phoenix observability test failed!")
        return 1

if __name__ == "__main__":
    exit(main())