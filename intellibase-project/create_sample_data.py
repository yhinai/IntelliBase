#!/usr/bin/env python3
"""
Create sample multimodal dataset for IntelliBase testing
"""
import os
from pathlib import Path

def create_sample_dataset():
    """Create sample multimodal dataset for testing"""
    
    data_dir = Path("./sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample text file
    with open(data_dir / "research_notes.txt", "w") as f:
        f.write("""AI Research Notes:
- Large Language Models are transformer-based architectures that process text
- Vector databases enable semantic search capabilities by storing embeddings
- RAG (Retrieval Augmented Generation) combines retrieval with generation for better accuracy
- Observability is crucial for production AI systems to monitor performance
- Multi-modal AI systems can process text, images, and other data types
- Daft provides high-performance data processing for large datasets
- Weaviate offers hybrid search combining vector and keyword search
- FriendliAI provides 2-4x faster LLM inference compared to standard endpoints
- Hypermode enables WebAssembly-powered AI agent deployment
- Arize Phoenix provides complete observability for AI applications
""")
    
    # Create sample markdown file
    with open(data_dir / "project_overview.md", "w") as f:
        f.write("""# IntelliBase Project

## Overview
A multi-modal AI research assistant that processes documents and images
to build a searchable knowledge base.

## Features
- Document processing with Daft for multi-modal data handling
- Vector storage with Weaviate for semantic search capabilities
- Fast inference with FriendliAI for optimized LLM responses
- Agent orchestration with Hypermode for workflow coordination
- Complete observability with Arize Phoenix for monitoring

## Architecture
The system follows a pipeline architecture:
1. Data ingestion and processing
2. Vector database indexing
3. Hybrid search retrieval
4. Fast LLM inference
5. Response generation with source attribution

## Use Cases
- Research paper analysis
- Document question answering
- Multi-modal content search
- Knowledge base construction
- Semantic information retrieval
""")
    
    # Create sample technical documentation
    with open(data_dir / "technical_specs.txt", "w") as f:
        f.write("""Technical Specifications:

Vector Database:
- Weaviate with hybrid search (vector + keyword)
- Support for multiple data types (text, images, metadata)
- Real-time indexing and retrieval
- Configurable similarity thresholds

Data Processing:
- Daft for distributed data processing
- Multi-modal support (PDF, images, text)
- Chunking strategies for large documents
- Metadata extraction and preservation

LLM Integration:
- FriendliAI for accelerated inference
- Meta-Llama-3.1-8B-Instruct model
- Configurable temperature and token limits
- Streaming and batch processing support

Observability:
- Arize Phoenix for end-to-end tracing
- Performance metrics and evaluation
- Real-time monitoring dashboard
- Custom span creation for detailed tracking

Agent Platform:
- Hypermode for agent orchestration
- WebAssembly-based execution
- GraphQL API generation
- Multi-language support (Go, AssemblyScript)
""")
    
    # Create sample FAQ
    with open(data_dir / "faq.md", "w") as f:
        f.write("""# Frequently Asked Questions

## What is IntelliBase?
IntelliBase is a multi-modal AI research assistant that processes various types of content (text, images, documents) to create an intelligent, searchable knowledge base.

## How does the system work?
1. **Data Ingestion**: Upload documents, images, or text files
2. **Processing**: Daft processes the content and extracts relevant information
3. **Indexing**: Content is stored in Weaviate vector database with embeddings
4. **Query**: Users ask questions in natural language
5. **Retrieval**: System finds relevant content using hybrid search
6. **Generation**: FriendliAI generates responses based on retrieved context

## What file types are supported?
- Text files (.txt, .md)
- PDF documents
- Images (.jpg, .png, .gif)
- More formats can be added through Daft's extensible processing pipeline

## How accurate are the responses?
The system uses retrieval-augmented generation (RAG) which significantly improves accuracy by grounding responses in your actual documents rather than relying solely on the LLM's training data.

## Can I monitor system performance?
Yes, Arize Phoenix provides comprehensive observability including response times, quality metrics, and detailed tracing of all operations.

## Is the system scalable?
Yes, all components are designed for production use:
- Daft handles large-scale data processing
- Weaviate scales to millions of documents
- FriendliAI provides efficient inference
- Hypermode enables distributed agent deployment
""")
    
    print(f"Sample data created in {data_dir}")
    print("Files created:")
    for file in data_dir.glob("*"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    create_sample_dataset()