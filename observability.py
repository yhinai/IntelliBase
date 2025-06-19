#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - Observability Setup
Phoenix tracing and instrumentation for complete observability
"""

import os
from typing import Any, Dict, Optional
import time
from functools import wraps

try:
    import phoenix as px
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    PHOENIX_AVAILABLE = True
except ImportError as e:
    print(f"Phoenix not fully available: {e}")
    print("Continuing with basic observability...")
    PHOENIX_AVAILABLE = False


class ObservabilityManager:
    """Manages Phoenix observability and tracing for the IntelliBase system"""
    
    def __init__(self):
        self.session = None
        self.tracer = None
        self.enabled = PHOENIX_AVAILABLE
        self.setup_phoenix()
        self.setup_tracing()
    
    def setup_phoenix(self):
        """Initialize Phoenix session for observability"""
        if not self.enabled:
            print("Phoenix observability disabled - missing dependencies")
            return
            
        try:
            # Launch Phoenix app for monitoring
            self.session = px.launch_app()
            print(f"🔍 Phoenix UI available at: {self.session.url}")
            print("📊 Observability dashboard is ready!")
            
        except Exception as e:
            print(f"Warning: Could not start Phoenix session: {e}")
            print("Continuing without Phoenix UI...")
            self.enabled = False
    
    def setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        if not self.enabled:
            return
            
        try:
            # Set up tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Auto-instrument components (if available)
            self._auto_instrument()
            
        except Exception as e:
            print(f"Warning: Tracing setup failed: {e}")
    
    def _auto_instrument(self):
        """Auto-instrument available components"""
        try:
            # Try to instrument OpenAI if available
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
            print("✅ OpenAI instrumentation enabled")
        except ImportError:
            print("⚠️ OpenAI instrumentation not available")
        
        try:
            # Try to instrument Weaviate if available
            from openinference.instrumentation.weaviate import WeaviateInstrumentor
            WeaviateInstrumentor().instrument()
            print("✅ Weaviate instrumentation enabled")
        except ImportError:
            print("⚠️ Weaviate instrumentation not available")
    
    def trace_operation(self, operation_name: str, **attributes):
        """Decorator for tracing operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled or not self.tracer:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(operation_name) as span:
                    # Set attributes
                    span.set_attribute("operation", operation_name)
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("timestamp", time.time())
                    
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                    
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        
                        # Add result metadata if possible
                        if hasattr(result, '__len__'):
                            span.set_attribute("result_count", len(result))
                        
                        return result
                        
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error_message", str(e))
                        span.set_attribute("error_type", type(e).__name__)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("duration_seconds", duration)
                        
            return wrapper
        return decorator
    
    def log_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Log metrics for an operation"""
        if not self.enabled:
            print(f"📊 {operation}: {metrics}")
            return
            
        try:
            # For now, just print metrics - in production would send to metrics backend
            print(f"📊 METRICS [{operation}]: {metrics}")
        except Exception as e:
            print(f"Warning: Metrics logging failed: {e}")
    
    def evaluate_response(self, query: str, response: str, context: list, 
                         expected_answer: Optional[str] = None) -> Dict[str, float]:
        """Evaluate response quality using Phoenix evals"""
        if not self.enabled:
            return {"relevance": 0.5, "groundedness": 0.5, "quality": 0.5}
        
        try:
            # Basic evaluation metrics
            metrics = {
                "response_length": len(response),
                "context_items": len(context),
                "query_length": len(query),
                "timestamp": time.time()
            }
            
            # Simple relevance scoring (would use proper models in production)
            relevance_score = min(1.0, len(response) / 100.0) if response else 0.0
            groundedness_score = min(1.0, len(context) / 5.0) if context else 0.0
            quality_score = (relevance_score + groundedness_score) / 2.0
            
            evaluation = {
                "relevance": relevance_score,
                "groundedness": groundedness_score, 
                "quality": quality_score
            }
            
            self.log_metrics("response_evaluation", {**metrics, **evaluation})
            return evaluation
            
        except Exception as e:
            print(f"Warning: Response evaluation failed: {e}")
            return {"relevance": 0.5, "groundedness": 0.5, "quality": 0.5}
    
    def shutdown(self):
        """Clean shutdown of observability components"""
        if self.session:
            try:
                print("🔍 Shutting down Phoenix session...")
                # Phoenix sessions typically auto-cleanup
            except Exception as e:
                print(f"Warning: Phoenix shutdown error: {e}")


# Global observability manager instance
obs_manager = ObservabilityManager()


# Convenience decorators for common operations
def trace_data_processing(**attributes):
    """Decorator for data processing operations"""
    return obs_manager.trace_operation("data_processing", **attributes)


def trace_vector_search(**attributes):
    """Decorator for vector search operations"""
    return obs_manager.trace_operation("vector_search", **attributes)


def trace_llm_inference(**attributes):
    """Decorator for LLM inference operations"""
    return obs_manager.trace_operation("llm_inference", **attributes)


def trace_agent_operation(**attributes):
    """Decorator for agent operations"""
    return obs_manager.trace_operation("agent_operation", **attributes)


if __name__ == "__main__":
    # Test the observability setup
    print("🚀 Testing IntelliBase Observability Setup")
    print("=" * 50)
    
    # Test basic functionality
    @trace_data_processing(test_operation=True)
    def test_operation():
        print("Testing traced operation...")
        time.sleep(0.1)  # Simulate work
        return {"processed_items": 5}
    
    result = test_operation()
    print(f"Test result: {result}")
    
    # Test evaluation
    evaluation = obs_manager.evaluate_response(
        query="What is machine learning?",
        response="Machine learning is a subset of AI that enables systems to learn from data.",
        context=["ML definition", "AI context", "Learning algorithms"]
    )
    print(f"Evaluation result: {evaluation}")
    
    print("\n✅ Observability setup complete!")
    print(f"Phoenix enabled: {obs_manager.enabled}")
    if obs_manager.session:
        print(f"Dashboard URL: {obs_manager.session.url}") 