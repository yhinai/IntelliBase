#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - FriendliAI Integration
Fast LLM inference using FriendliAI's optimized serving
"""

import os
from typing import List, Dict, Any, Optional
import time
import json

try:
    from friendli import Friendli
    FRIENDLI_AVAILABLE = True
    print("✅ FriendliAI client available")
except ImportError:
    print("⚠️ FriendliAI client not available - using mock implementation")
    FRIENDLI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
    print("✅ OpenAI client available (fallback)")
except ImportError:
    print("⚠️ OpenAI client not available")
    OPENAI_AVAILABLE = False

try:
    from observability import trace_llm_inference, obs_manager
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    print("⚠️ Observability not available - continuing without tracing")
    OBSERVABILITY_AVAILABLE = False
    
    # Create dummy decorators
    def trace_llm_inference(**kwargs):
        def decorator(func):
            return func
        return decorator


class FriendliAIIntegration:
    """Manages FriendliAI accelerated inference for IntelliBase"""
    
    def __init__(self):
        self.friendli_client = None
        self.openai_client = None
        self.friendli_available = FRIENDLI_AVAILABLE
        self.openai_available = OPENAI_AVAILABLE
        
        # Default configuration
        self.default_model = "meta-llama-3.1-8b-instruct"
        self.default_temperature = 0.7
        self.default_max_tokens = 500
        
        # Initialize clients
        self._initialize_clients()
        
        print(f"🔧 FriendliAI Integration initialized")
        print(f"   - FriendliAI: {self.friendli_available and self.friendli_client is not None}")
        print(f"   - OpenAI fallback: {self.openai_available and self.openai_client is not None}")
    
    def _initialize_clients(self):
        """Initialize FriendliAI and fallback clients"""
        
        # Try to initialize FriendliAI
        if self.friendli_available:
            try:
                friendli_token = os.getenv("FRIENDLI_TOKEN")
                if friendli_token and friendli_token != "your_friendli_token_here":
                    self.friendli_client = Friendli(token=friendli_token)
                    print("🚀 FriendliAI client initialized")
                else:
                    print("⚠️ FRIENDLI_TOKEN not found or placeholder - FriendliAI disabled")
            except Exception as e:
                print(f"⚠️ FriendliAI initialization failed: {e}")
        
        # Try to initialize OpenAI as fallback
        if self.openai_available:
            try:
                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key and openai_key != "your_openai_key_here":
                    self.openai_client = openai.OpenAI(api_key=openai_key)
                    print("🔄 OpenAI fallback client initialized")
                else:
                    print("⚠️ OPENAI_API_KEY not found or placeholder - OpenAI fallback disabled")
            except Exception as e:
                print(f"⚠️ OpenAI initialization failed: {e}")
    
    @trace_llm_inference(provider="friendliai")
    def generate_response_friendli(self, prompt: str, context: List[str] = None, 
                                 model: str = None, **kwargs) -> Dict[str, Any]:
        """Generate response using FriendliAI"""
        
        if not self.friendli_client:
            return self._mock_response(prompt, "FriendliAI client not available")
        
        try:
            # Build the conversation
            messages = self._build_messages(prompt, context)
            
            # Use provided model or default
            model = model or self.default_model
            
            # Extract parameters
            temperature = kwargs.get("temperature", self.default_temperature)
            max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
            
            print(f"🤖 Generating response with FriendliAI ({model})")
            start_time = time.time()
            
            response = self.friendli_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            end_time = time.time()
            
            # Extract response content
            content = response.choices[0].message.content if response.choices else "No response generated"
            
            # Log metrics
            if OBSERVABILITY_AVAILABLE:
                obs_manager.log_metrics("friendli_inference", {
                    "model": model,
                    "prompt_length": len(prompt),
                    "context_items": len(context) if context else 0,
                    "response_length": len(content),
                    "generation_time": end_time - start_time,
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
                })
            
            return {
                "content": content,
                "model": model,
                "provider": "friendliai",
                "generation_time": end_time - start_time,
                "usage": response.usage._asdict() if hasattr(response, 'usage') else {},
                "success": True
            }
            
        except Exception as e:
            print(f"❌ FriendliAI generation failed: {e}")
            return {
                "content": "",
                "model": model or self.default_model,
                "provider": "friendliai",
                "generation_time": 0,
                "usage": {},
                "success": False,
                "error": str(e)
            }
    
    @trace_llm_inference(provider="openai")
    def generate_response_openai(self, prompt: str, context: List[str] = None, 
                                model: str = "gpt-3.5-turbo", **kwargs) -> Dict[str, Any]:
        """Generate response using OpenAI as fallback"""
        
        if not self.openai_client:
            return self._mock_response(prompt, "OpenAI client not available")
        
        try:
            # Build the conversation
            messages = self._build_messages(prompt, context)
            
            # Extract parameters
            temperature = kwargs.get("temperature", self.default_temperature)
            max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
            
            print(f"🔄 Generating response with OpenAI ({model})")
            start_time = time.time()
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            end_time = time.time()
            
            # Extract response content
            content = response.choices[0].message.content if response.choices else "No response generated"
            
            # Log metrics
            if OBSERVABILITY_AVAILABLE:
                obs_manager.log_metrics("openai_inference", {
                    "model": model,
                    "prompt_length": len(prompt),
                    "context_items": len(context) if context else 0,
                    "response_length": len(content),
                    "generation_time": end_time - start_time,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                })
            
            return {
                "content": content,
                "model": model,
                "provider": "openai",
                "generation_time": end_time - start_time,
                "usage": {
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0
                },
                "success": True
            }
            
        except Exception as e:
            print(f"❌ OpenAI generation failed: {e}")
            return self._mock_response(prompt, f"OpenAI error: {e}")
    
    def _build_messages(self, prompt: str, context: List[str] = None) -> List[Dict[str, str]]:
        """Build conversation messages with context"""
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI research assistant for the IntelliBase system. 
You provide accurate, informative responses based on the provided context. 
If you don't have enough information in the context, say so clearly."""
            }
        ]
        
        # Add context if provided
        if context:
            context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
            user_message = f"""Based on the following context, please answer the question:

{context_text}

Question: {prompt}

Please provide a clear, comprehensive answer based on the context provided."""
        else:
            user_message = prompt
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _mock_response(self, prompt: str, reason: str) -> Dict[str, Any]:
        """Generate a mock response when real APIs are unavailable"""
        
        # Create a simple rule-based response
        prompt_lower = prompt.lower()
        
        if "machine learning" in prompt_lower or "ml" in prompt_lower:
            content = """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions."""
        elif "vector database" in prompt_lower:
            content = """Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They enable semantic search capabilities by comparing vector embeddings, making them essential for AI applications like RAG systems."""
        elif "ai" in prompt_lower or "artificial intelligence" in prompt_lower:
            content = """Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. It encompasses various techniques including machine learning, deep learning, and natural language processing to solve complex problems."""
        elif "processing" in prompt_lower:
            content = """Data processing involves collecting, transforming, and analyzing data to extract useful information. In AI systems, processing often includes cleaning data, feature extraction, and preparing data for machine learning models."""
        else:
            content = f"I understand you're asking about: {prompt}. However, I'm currently running in mock mode ({reason}). In a full implementation, this would be processed by FriendliAI or OpenAI for a comprehensive response."
        
        return {
            "content": content,
            "model": "mock_model",
            "provider": "mock",
            "generation_time": 0.1,
            "usage": {"total_tokens": len(content.split())},
            "success": True,
            "mock": True,
            "mock_reason": reason
        }
    
    @trace_llm_inference(provider="auto")
    def generate_response(self, prompt: str, context: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate response using best available provider (OpenAI preferred when FriendliAI fails)"""
        
        # Try OpenAI first (more reliable)
        if self.openai_client:
            try:
                result = self.generate_response_openai(prompt, context, **kwargs)
                if result["success"]:
                    return result
            except Exception as e:
                print(f"⚠️ OpenAI fallback failed: {e}")
        
        # Try FriendliAI as backup
        if self.friendli_client:
            try:
                result = self.generate_response_friendli(prompt, context, **kwargs)
                if result["success"]:
                    return result
            except Exception as e:
                print(f"⚠️ FriendliAI failed: {e}")
        
        # Final fallback to mock
        return self._mock_response(prompt, "No LLM providers available")
    
    def generate_with_retrieval(self, query: str, search_results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate response with retrieved context from vector database"""
        
        if not search_results:
            return self.generate_response(
                f"Please answer this question: {query}", 
                context=["No relevant context found in the knowledge base."],
                **kwargs
            )
        
        # Extract content from search results
        context = []
        sources = []
        
        for result in search_results:
            content = result.get("content", "")
            source = result.get("source_file", "unknown")
            
            if content:
                context.append(content)
                sources.append(source)
        
        # Generate response with context
        response = self.generate_response(query, context, **kwargs)
        
        # Add source information
        response["sources"] = sources
        response["context_used"] = len(context)
        response["search_results_count"] = len(search_results)
        
        # Evaluate response quality
        if OBSERVABILITY_AVAILABLE:
            evaluation = obs_manager.evaluate_response(
                query=query,
                response=response["content"],
                context=context
            )
            response["evaluation"] = evaluation
        
        return response
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = []
        
        if self.friendli_client:
            # FriendliAI models (common ones)
            models.extend([
                "meta-llama-3.1-8b-instruct",
                "meta-llama-3.1-70b-instruct", 
                "mixtral-8x7b-instruct-v0.1"
            ])
        
        if self.openai_client:
            # OpenAI models
            models.extend([
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo"
            ])
        
        if not models:
            models = ["mock_model"]
        
        return models
    
    def benchmark_inference(self, test_prompt: str = "What is machine learning?") -> Dict[str, Any]:
        """Benchmark inference speed across providers"""
        
        results = {}
        
        # Test FriendliAI
        if self.friendli_client:
            start_time = time.time()
            friendli_result = self.generate_response_friendli(test_prompt)
            friendli_time = time.time() - start_time
            
            results["friendliai"] = {
                "generation_time": friendli_time,
                "response_length": len(friendli_result["content"]),
                "success": friendli_result["success"]
            }
        
        # Test OpenAI
        if self.openai_client:
            start_time = time.time()
            openai_result = self.generate_response_openai(test_prompt)
            openai_time = time.time() - start_time
            
            results["openai"] = {
                "generation_time": openai_time,
                "response_length": len(openai_result["content"]),
                "success": openai_result["success"]
            }
        
        # Calculate speedup
        if "friendliai" in results and "openai" in results:
            if results["openai"]["generation_time"] > 0:
                speedup = results["openai"]["generation_time"] / results["friendliai"]["generation_time"]
                results["speedup"] = f"{speedup:.2f}x faster with FriendliAI"
        
        return results


if __name__ == "__main__":
    # Test the FriendliAI integration
    print("🚀 Testing IntelliBase FriendliAI Integration")
    print("=" * 50)
    
    # Initialize integration
    ai = FriendliAIIntegration()
    
    # Test basic generation
    print("\n🤖 Testing basic response generation...")
    test_queries = [
        "What is machine learning?",
        "Explain vector databases",
        "How do AI systems work?",
        "What is data processing?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        response = ai.generate_response(query)
        print(f"🤖 Response ({response['provider']}):")
        print(f"   {response['content'][:150]}...")
        print(f"   Time: {response['generation_time']:.3f}s")
        if response.get("mock"):
            print(f"   (Mock mode: {response['mock_reason']})")
    
    # Test with context
    print("\n📚 Testing response with context...")
    sample_context = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "It is a branch of artificial intelligence based on the idea that systems can learn from data."
    ]
    
    response_with_context = ai.generate_response(
        "What is machine learning?", 
        context=sample_context
    )
    print(f"🤖 Response with context ({response_with_context['provider']}):")
    print(f"   {response_with_context['content'][:200]}...")
    
    # Show available models
    models = ai.get_available_models()
    print(f"\n🔧 Available models: {models}")
    
    # Benchmark if multiple providers available
    print(f"\n⚡ Running inference benchmark...")
    benchmark_results = ai.benchmark_inference()
    print(f"📊 Benchmark Results:")
    for provider, stats in benchmark_results.items():
        if provider != "speedup":
            print(f"  {provider}: {stats['generation_time']:.3f}s")
    
    if "speedup" in benchmark_results:
        print(f"  🚀 {benchmark_results['speedup']}")
    
    print("\n✅ FriendliAI integration test complete!") 