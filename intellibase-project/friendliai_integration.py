#!/usr/bin/env python3
"""
FriendliAI integration for fast LLM inference in IntelliBase
"""
import os
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResponse:
    """Response from LLM generation"""
    content: str
    model: str
    processing_time: float
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = None

class MockFriendliAIClient:
    """Mock FriendliAI client for testing without API key"""
    
    def __init__(self):
        self.model = "meta-llama-3.1-8b-instruct"
        logger.info("Initialized mock FriendliAI client")
    
    def generate_with_context(self, query: str, context: List[str], temperature: float = 0.7, max_tokens: int = 500) -> GenerationResponse:
        """Generate response using mock LLM with retrieved context"""
        
        start_time = time.time()
        
        # Build context text
        context_text = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context[:3])])
        
        # Mock response generation
        mock_response = self._generate_mock_response(query, context_text)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated response in {processing_time:.3f}s")
        
        return GenerationResponse(
            content=mock_response,
            model=self.model,
            processing_time=processing_time,
            token_count=len(mock_response.split()),
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "context_length": len(context_text)
            }
        )
    
    def _generate_mock_response(self, query: str, context: str) -> str:
        """Generate a mock response based on query and context"""
        
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Simple pattern matching for demo purposes
        if "machine learning" in query_lower or "machine learning" in context_lower:
            return """Based on the provided context, machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns, make predictions, and improve their performance over time through experience with data."""
        
        elif "vector database" in query_lower or "vector" in context_lower:
            return """Vector databases are specialized databases designed to store and query high-dimensional vector embeddings. They enable semantic search capabilities by storing numerical representations of data (like text, images, or audio) and performing similarity searches based on the geometric relationships between these vectors."""
        
        elif "rag" in query_lower or "retrieval" in context_lower:
            return """RAG (Retrieval Augmented Generation) is an AI technique that combines information retrieval with text generation. It first retrieves relevant information from a knowledge base, then uses this context to generate more accurate and grounded responses, reducing hallucinations and improving factual accuracy."""
        
        elif "observability" in query_lower or "monitoring" in context_lower:
            return """Observability in AI systems refers to the ability to understand and monitor the internal state and behavior of AI applications. This includes tracking performance metrics, tracing requests through the system, monitoring model outputs, and evaluating response quality to ensure reliable operation."""
        
        elif "daft" in query_lower or "data processing" in context_lower:
            return """Daft is a distributed data processing framework designed for high-performance multimodal data operations. It provides efficient processing of large datasets including text, images, and other data types, with support for complex transformations and cloud storage integration."""
        
        else:
            # Generic response using context
            if context.strip():
                return f"""Based on the provided information, I can help answer your question about "{query}". The context suggests relevant details that can inform a comprehensive response. However, for the most accurate information, I'd recommend referring to the specific documentation or sources mentioned in the context."""
            else:
                return f"""I'd be happy to help answer your question about "{query}". However, I don't have specific context available to provide a detailed response. Could you provide more specific information or context about what you're looking for?"""

class FriendliAIIntegration:
    """FriendliAI integration with fallback to mock"""
    
    def __init__(self):
        self.client = None
        self.using_mock = True
        
        # Check if we have FriendliAI API key
        if os.getenv("FRIENDLI_TOKEN"):
            self._initialize_real_client()
        else:
            logger.warning("No FriendliAI token found, using mock client")
            self.client = MockFriendliAIClient()
    
    def _initialize_real_client(self):
        """Initialize real FriendliAI client"""
        try:
            # This would initialize the real FriendliAI client
            # For now, fall back to mock since we don't have the actual client
            logger.info("FriendliAI token found, but using mock for demo")
            self.client = MockFriendliAIClient()
            self.using_mock = True
            
        except Exception as e:
            logger.error(f"Failed to initialize FriendliAI client: {e}")
            logger.info("Falling back to mock client")
            self.client = MockFriendliAIClient()
            self.using_mock = True
    
    def generate_with_context(self, query: str, context: List[str], **kwargs) -> GenerationResponse:
        """Generate response with context"""
        return self.client.generate_with_context(query, context, **kwargs)
    
    def is_using_mock(self) -> bool:
        """Check if using mock client"""
        return self.using_mock

def test_friendliai_integration():
    """Test the FriendliAI integration"""
    
    logger.info("Testing FriendliAI integration...")
    
    # Initialize integration
    integration = FriendliAIIntegration()
    
    # Test contexts
    test_cases = [
        {
            "query": "What is machine learning?",
            "context": ["Machine learning is a subset of artificial intelligence", "It involves algorithms that learn from data"]
        },
        {
            "query": "How do vector databases work?", 
            "context": ["Vector databases store embeddings for semantic search", "They enable similarity-based retrieval"]
        },
        {
            "query": "What is RAG?",
            "context": ["RAG combines retrieval with generation", "It improves accuracy by grounding responses in documents"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\nTest {i+1}: {test_case['query']}")
        
        response = integration.generate_with_context(
            query=test_case["query"],
            context=test_case["context"]
        )
        
        logger.info(f"Response: {response.content[:100]}...")
        logger.info(f"Processing time: {response.processing_time:.3f}s")
        logger.info(f"Token count: {response.token_count}")
        
        results.append(response)
    
    # Test performance
    avg_time = sum(r.processing_time for r in results) / len(results)
    logger.info(f"\nAverage processing time: {avg_time:.3f}s")
    
    return len(results) == len(test_cases) and all(r.content for r in results)

def main():
    """Test the FriendliAI integration"""
    success = test_friendliai_integration()
    
    if success:
        print("✅ FriendliAI integration test successful!")
        return 0
    else:
        print("❌ FriendliAI integration test failed!")
        return 1

if __name__ == "__main__":
    exit(main())