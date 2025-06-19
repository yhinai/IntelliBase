#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - Main Application
Complete multi-modal AI research assistant integrating all components
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json

# Load environment variables from config.env
try:
    from dotenv import load_dotenv
    load_dotenv('config.env')
    print("✅ Environment variables loaded from config.env")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️ Could not load config.env: {e}")

# Import our components
try:
    from daft_processor_simple import IntelliBaseDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️ Data processor not available")
    DATA_PROCESSOR_AVAILABLE = False

try:
    from weaviate_manager import WeaviateManager
    WEAVIATE_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ Weaviate manager not available")
    WEAVIATE_MANAGER_AVAILABLE = False

try:
    from friendliai_integration import FriendliAIIntegration
    AI_INTEGRATION_AVAILABLE = True
except ImportError:
    print("⚠️ AI integration not available")
    AI_INTEGRATION_AVAILABLE = False

try:
    from hypermode_integration import hypermode_integration
    HYPERMODE_AVAILABLE = True
except ImportError:
    print("⚠️ Hypermode integration not available")
    HYPERMODE_AVAILABLE = False

try:
    from observability import obs_manager, trace_agent_operation
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    print("⚠️ Observability not available")
    OBSERVABILITY_AVAILABLE = False
    
    # Create dummy decorator
    def trace_agent_operation(**kwargs):
        def decorator(func):
            return func
        return decorator


class IntelliBaseApp:
    """Main IntelliBase application orchestrating all components"""
    
    def __init__(self, data_directory: str = "./sample_data"):
        self.data_directory = data_directory
        self.data_processor = None
        self.vector_manager = None
        self.ai_integration = None
        self.hypermode_integration = None
        
        # Initialize components
        self._initialize_components()
        
        # Application state
        self.is_ready = False
        self.stats = {
            "files_processed": 0,
            "chunks_indexed": 0,
            "queries_processed": 0,
            "total_generation_time": 0.0
        }
        
        print(f"🚀 IntelliBase Application initialized")
        print(f"   Data directory: {data_directory}")
        self._check_readiness()
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        print("🔧 Initializing IntelliBase components...")
        
        # Initialize data processor
        if DATA_PROCESSOR_AVAILABLE:
            self.data_processor = IntelliBaseDataProcessor()
            print("✅ Data processor ready")
        else:
            print("❌ Data processor unavailable")
        
        # Initialize vector manager
        if WEAVIATE_MANAGER_AVAILABLE:
            self.vector_manager = WeaviateManager(use_local=True)
            print("✅ Vector manager ready")
        else:
            print("❌ Vector manager unavailable")
        
        # Initialize AI integration
        if AI_INTEGRATION_AVAILABLE:
            self.ai_integration = FriendliAIIntegration()
            print("✅ AI integration ready")
        else:
            print("❌ AI integration unavailable")
        
        # Initialize Hypermode integration
        if HYPERMODE_AVAILABLE:
            self.hypermode_integration = hypermode_integration
            print("✅ Hypermode integration ready")
        else:
            print("❌ Hypermode integration unavailable")
    
    def _check_readiness(self):
        """Check if the system is ready for operation"""
        
        required_components = [
            (self.data_processor, "Data Processor"),
            (self.vector_manager, "Vector Manager"),
            (self.ai_integration, "AI Integration")
        ]
        
        optional_components = [
            (self.hypermode_integration, "Hypermode Integration")
        ]
        
        missing_components = []
        for component, name in required_components:
            if component is None:
                missing_components.append(name)
        
        if missing_components:
            print(f"⚠️ Missing required components: {', '.join(missing_components)}")
            print("🔄 Running in limited mode...")
            self.is_ready = False
        else:
            print("✅ All required components ready - IntelliBase is operational!")
            
            # Check optional components
            for component, name in optional_components:
                if component is None:
                    print(f"⚠️ Optional component missing: {name}")
                else:
                    print(f"✅ Optional component ready: {name}")
            
            self.is_ready = True
    
    @trace_agent_operation(operation="setup_knowledge_base")
    def setup_knowledge_base(self, force_reprocess: bool = False) -> bool:
        """Set up the knowledge base by processing documents and creating vectors"""
        
        print("📚 Setting up IntelliBase knowledge base...")
        
        if not self.data_processor or not self.vector_manager:
            print("❌ Cannot setup knowledge base - missing required components")
            return False
        
        try:
            # Step 1: Process documents
            print(f"📊 Processing documents from {self.data_directory}...")
            processed_files = self.data_processor.process_directory(self.data_directory)
            
            if not processed_files:
                print("❌ No documents processed successfully")
                return False
            
            self.stats["files_processed"] = len(processed_files)
            print(f"✅ Processed {len(processed_files)} files")
            
            # Step 2: Create chunks for vector database
            print("✂️ Creating chunks for vector database...")
            chunks = self.data_processor.create_chunks(processed_files, chunk_size=800)
            
            if not chunks:
                print("❌ No chunks created")
                return False
            
            # Step 3: Create vector collection
            print("📝 Creating vector database collection...")
            collection_created = self.vector_manager.create_collection(reset=force_reprocess)
            
            if not collection_created:
                print("❌ Failed to create vector collection")
                return False
            
            # Step 4: Index chunks in vector database
            print(f"📥 Indexing {len(chunks)} chunks in vector database...")
            index_success = self.vector_manager.batch_insert_chunks(chunks)
            
            if not index_success:
                print("❌ Failed to index chunks")
                return False
            
            self.stats["chunks_indexed"] = len(chunks)
            
            # Step 5: Verify setup
            stats = self.vector_manager.get_collection_stats()
            print(f"📊 Knowledge base statistics:")
            print(f"   Total objects: {stats.get('total_objects', 0)}")
            print(f"   Storage type: {stats.get('storage_type', 'unknown')}")
            
            print("✅ Knowledge base setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Knowledge base setup failed: {e}")
            return False
    
    @trace_agent_operation(operation="query_knowledge_base")
    def query(self, user_query: str, max_results: int = 5, **kwargs) -> Dict[str, Any]:
        """Query the knowledge base and generate a response"""
        
        print(f"❓ Processing query: '{user_query}'")
        
        if not self.is_ready:
            return {
                "query": user_query,
                "answer": "IntelliBase is not fully operational. Some components are missing.",
                "sources": [],
                "success": False,
                "error": "System not ready"
            }
        
        try:
            start_time = time.time()
            
            # Step 1: Search the vector database
            print("🔍 Searching knowledge base...")
            search_results = self.vector_manager.hybrid_search(
                query=user_query,
                limit=max_results,
                alpha=kwargs.get("search_alpha", 0.75)
            )
            
            if not search_results:
                print("🤷 No relevant results found")
                response = self.ai_integration.generate_response(
                    f"Please answer this question: {user_query}",
                    context=["No relevant information found in the knowledge base."]
                )
            else:
                print(f"📋 Found {len(search_results)} relevant results")
                
                # Step 2: Generate response with context
                print("🤖 Generating AI response...")
                response = self.ai_integration.generate_with_retrieval(
                    query=user_query,
                    search_results=search_results,
                    **kwargs
                )
            
            end_time = time.time()
            total_time = end_time - start_time
            self.stats["queries_processed"] += 1
            self.stats["total_generation_time"] += total_time
            
            # Build comprehensive response
            result = {
                "query": user_query,
                "answer": response.get("content", "No response generated"),
                "sources": response.get("sources", []),
                "provider": response.get("provider", "unknown"),
                "model": response.get("model", "unknown"),
                "generation_time": response.get("generation_time", 0),
                "total_time": total_time,
                "search_results_count": len(search_results),
                "success": True
            }
            
            # Add evaluation if available
            if "evaluation" in response:
                result["evaluation"] = response["evaluation"]
            
            # Add mock flag if applicable
            if response.get("mock"):
                result["mock"] = True
                result["mock_reason"] = response.get("mock_reason", "")
            
            print(f"✅ Query processed in {total_time:.3f}s")
            return result
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            return {
                "query": user_query,
                "answer": f"Error processing query: {e}",
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        components = {
            "data_processor": self.data_processor is not None,
            "vector_manager": self.vector_manager is not None,
            "ai_integration": self.ai_integration is not None,
            "hypermode_integration": self.hypermode_integration is not None,
            "observability": OBSERVABILITY_AVAILABLE
        }
        
        # Get detailed component status
        component_details = {}
        
        if self.vector_manager:
            component_details["vector_manager"] = {
                "available": True,
                "storage_type": self.vector_manager.get_collection_stats().get("storage_type", "unknown"),
                "total_objects": self.vector_manager.get_collection_stats().get("total_objects", 0)
            }
        
        if self.ai_integration:
            component_details["ai_integration"] = {
                "friendli_available": self.ai_integration.friendli_client is not None,
                "openai_available": self.ai_integration.openai_client is not None
            }
        
        if self.hypermode_integration:
            component_details["hypermode_integration"] = self.hypermode_integration.get_status()
        
        return {
            "is_ready": self.is_ready,
            "components": components,
            "component_details": component_details,
            "stats": self.stats,
            "data_directory": self.data_directory
        }
    
    def run_demo(self):
        """Run a comprehensive demo of the system"""
        
        print("🎭 Running IntelliBase Demo")
        print("=" * 60)
        
        # Setup knowledge base
        print("\n🚀 Phase 1: Setting up knowledge base...")
        setup_success = self.setup_knowledge_base()
        
        if not setup_success:
            print("❌ Demo failed - knowledge base setup unsuccessful")
            return
        
        # Test queries
        print("\n🚀 Phase 2: Testing query capabilities...")
        demo_queries = [
            "What is machine learning?",
            "How do vector databases work?",
            "Explain the IntelliBase architecture",
            "What are the benefits of AI observability?",
            "How does multi-modal processing work?"
        ]
        
        results = []
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 Demo Query {i}: {query}")
            result = self.query(query)
            
            print(f"🤖 Answer: {result['answer'][:150]}...")
            print(f"📊 Sources: {len(result['sources'])} documents")
            print(f"⏱️ Time: {result['total_time']:.3f}s")
            
            if result.get("mock"):
                print(f"🎭 (Mock mode: {result['mock_reason']})")
            
            results.append(result)
        
        # Show system statistics
        print(f"\n🚀 Phase 3: System statistics...")
        status = self.get_system_status()
        
        print(f"📊 System Status:")
        print(f"   Ready: {status['is_ready']}")
        print(f"   Files processed: {status['stats']['files_processed']}")
        print(f"   Chunks indexed: {status['stats']['chunks_indexed']}")
        print(f"   Queries processed: {status['stats']['queries_processed']}")
        print(f"   Avg query time: {status['stats']['total_generation_time'] / max(1, status['stats']['queries_processed']):.3f}s")
        
        # Component status
        print(f"🔧 Component Status:")
        for component, available in status['components'].items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {component}")
        
        print("\n🎉 IntelliBase Demo Complete!")
        
        # Export demo results
        demo_report = {
            "demo_timestamp": time.time(),
            "system_status": status,
            "demo_queries": demo_queries,
            "query_results": results
        }
        
        with open("demo_report.json", "w") as f:
            json.dump(demo_report, f, indent=2, default=str)
        
        print("📄 Demo report saved to demo_report.json")
    
    def interactive_mode(self):
        """Run the system in interactive mode"""
        
        print("🎮 IntelliBase Interactive Mode")
        print("=" * 40)
        print("Commands:")
        print("  - Type a question to query the knowledge base")
        print("  - 'status' to see system status")
        print("  - 'setup' to re-setup knowledge base")
        print("  - 'quit' or 'exit' to quit")
        print("=" * 40)
        
        # Ensure knowledge base is set up
        if not self.is_ready:
            print("\n🔧 Setting up knowledge base first...")
            self.setup_knowledge_base()
        
        while True:
            try:
                user_input = input("\n❓ IntelliBase> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\n📊 System Status: {'Ready' if status['is_ready'] else 'Not Ready'}")
                    print(f"   Files: {status['stats']['files_processed']}")
                    print(f"   Chunks: {status['stats']['chunks_indexed']}")
                    print(f"   Queries: {status['stats']['queries_processed']}")
                
                elif user_input.lower() == 'setup':
                    print("\n🔧 Re-setting up knowledge base...")
                    self.setup_knowledge_base(force_reprocess=True)
                
                else:
                    # Process as query
                    result = self.query(user_input)
                    print(f"\n🤖 Answer: {result['answer']}")
                    
                    if result['sources']:
                        print(f"📚 Sources: {', '.join(result['sources'])}")
                    
                    print(f"⏱️ Response time: {result['total_time']:.3f}s")
                    
                    if result.get("mock"):
                        print(f"🎭 (Running in mock mode)")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.vector_manager:
            self.vector_manager.close()
        
        if OBSERVABILITY_AVAILABLE:
            obs_manager.shutdown()

    @trace_agent_operation(operation="execute_hypermode_agent")
    def execute_hypermode_agent(self, agent_name: str, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute a Hypermode agent"""
        
        if not self.hypermode_integration:
            return {
                "success": False,
                "error": "Hypermode integration not available"
            }
        
        try:
            result = self.hypermode_integration.execute_agent(agent_name, inputs, **kwargs)
            
            if result["success"]:
                self.stats["queries_processed"] += 1
                self.stats["total_generation_time"] += result.get("execution_time", 0)
            
            return result
            
        except Exception as e:
            print(f"❌ Hypermode agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_hypermode_status(self) -> Dict[str, Any]:
        """Get Hypermode integration status"""
        
        if not self.hypermode_integration:
            return {"available": False, "error": "Not initialized"}
        
        return self.hypermode_integration.get_status()


if __name__ == "__main__":
    # Main application entry point
    print("🌟 Welcome to IntelliBase!")
    print("Multi-modal AI Research Assistant")
    print("=" * 50)
    
    # Initialize application
    app = IntelliBaseApp()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "demo":
            app.run_demo()
        elif mode == "interactive":
            app.interactive_mode()
        elif mode == "setup":
            app.setup_knowledge_base(force_reprocess=True)
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: demo, interactive, setup")
    else:
        # Default: run demo
        app.run_demo()
    
    # Cleanup
    app.cleanup()
    print("\n🔒 IntelliBase shutdown complete") 