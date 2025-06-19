#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - Sample Data Creation
Creates a multi-modal dataset for testing the system
"""

import os
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont
import io


def create_sample_dataset():
    """Create sample multimodal dataset for testing"""
    
    data_dir = Path("./sample_data")
    data_dir.mkdir(exist_ok=True)
    
    print("Creating sample data directory:", data_dir.absolute())
    
    # Create sample text file
    with open(data_dir / "research_notes.txt", "w") as f:
        f.write("""
AI Research Notes:
==================

Large Language Models (LLMs):
- Transformer-based architectures that revolutionized NLP
- Self-attention mechanisms enable parallel processing
- GPT, BERT, and T5 are foundational architectures
- Scaling laws show performance increases with model size and data

Vector Databases:
- Enable semantic search capabilities through embeddings
- Support high-dimensional vector operations
- Essential for Retrieval-Augmented Generation (RAG)
- Popular solutions include Weaviate, Pinecone, and Chroma

RAG Systems:
- Combine retrieval with generation for better accuracy
- Retrieve relevant context from knowledge base
- Generate responses conditioned on retrieved context
- Reduces hallucination and improves factual accuracy

AI Observability:
- Crucial for production AI systems
- Monitor model performance, latency, and costs
- Track data drift and model degradation
- Tools like Arize Phoenix provide comprehensive monitoring

Multi-modal AI:
- Process multiple data types (text, images, audio)
- Vision-language models like CLIP enable cross-modal understanding
- Applications in document processing, image captioning, and visual QA
- Requires specialized data processing pipelines
        """)
    
    # Create sample markdown file
    with open(data_dir / "project_overview.md", "w") as f:
        f.write("""
# IntelliBase Project

## Overview
A multi-modal AI research assistant that processes documents and images
to build a searchable knowledge base with fast inference capabilities.

## Architecture Components

### Data Processing Layer
- **Daft**: High-performance multi-modal data processing
- Handles PDFs, images, and text files uniformly
- Scalable processing with lazy evaluation

### Vector Storage Layer  
- **Weaviate**: Vector database with hybrid search
- Combines semantic similarity with keyword matching
- Production-ready with horizontal scaling

### Inference Layer
- **FriendliAI**: Accelerated LLM inference (2-4x faster)
- Optimized model serving with reduced latency
- Cost-effective inference at scale

### Agent Orchestration
- **Hypermode**: AI agent platform with WebAssembly execution
- Serverless function deployment
- Event-driven architecture for complex workflows

### Observability Layer
- **Arize Phoenix**: Complete AI observability and evaluation
- End-to-end tracing and monitoring
- Model performance analytics

## Key Features

1. **Multi-modal Processing**: Handle diverse file types
2. **Hybrid Search**: Semantic + keyword search capabilities  
3. **Fast Inference**: 2-4x faster responses via FriendliAI
4. **Agent Coordination**: Complex workflow orchestration
5. **Complete Observability**: End-to-end monitoring and evaluation

## Use Cases

- Research paper analysis and summarization
- Technical documentation search
- Image content extraction and analysis
- Cross-modal question answering
- Knowledge base construction from mixed media
        """)
    
    # Create sample technical documentation
    with open(data_dir / "technical_specs.md", "w") as f:
        f.write("""
# Technical Specifications

## System Requirements

### Minimum Hardware
- 8GB RAM for local development
- 4 CPU cores recommended
- 10GB storage for sample datasets

### Production Deployment
- Container orchestration (Docker/Kubernetes)
- Load balancer for high availability
- Persistent storage for vector database

## API Endpoints

### Data Processing
- POST /api/upload - Upload files for processing
- GET /api/status/{job_id} - Check processing status
- GET /api/processed/{job_id} - Retrieve processed data

### Search and Query
- POST /api/search - Perform hybrid search
- POST /api/query - Generate AI responses
- GET /api/health - System health check

### Observability
- GET /api/metrics - System metrics
- GET /api/traces - Distributed traces
- POST /api/evaluate - Model evaluation

## Configuration

### Environment Variables
- FRIENDLI_TOKEN: API token for FriendliAI
- WEAVIATE_CLUSTER_URL: Weaviate instance URL
- WEAVIATE_API_KEY: Weaviate authentication key
- PHOENIX_COLLECTOR_ENDPOINT: Observability endpoint

### Model Configuration
- Default model: meta-llama-3.1-8b-instruct
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Vector dimensions: 384
- Max context length: 4096 tokens
        """)
    
    # Create a simple synthetic image with text
    create_sample_image(data_dir / "diagram.png", "AI System Architecture")
    create_sample_image(data_dir / "flowchart.png", "Data Processing Flow")
    
    print(f"\nSample data created successfully in {data_dir}")
    print("Files created:")
    for file in data_dir.iterdir():
        print(f"  - {file.name} ({file.stat().st_size} bytes)")


def create_sample_image(filepath: Path, title: str):
    """Create a simple image with text for testing"""
    
    # Create a 800x600 image with white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a basic font, fall back to default if not available
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        font_text = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        try:
            font_title = ImageFont.truetype("arial.ttf", 36)
            font_text = ImageFont.truetype("arial.ttf", 20)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
    
    # Draw title
    draw.text((50, 50), title, fill='black', font=font_title)
    
    # Draw some sample content
    content = [
        "• Multi-modal data processing",
        "• Vector-based semantic search", 
        "• Fast LLM inference",
        "• Complete observability",
        "• Agent orchestration"
    ]
    
    y_pos = 150
    for line in content:
        draw.text((100, y_pos), line, fill='black', font=font_text)
        y_pos += 40
    
    # Draw some boxes to simulate a diagram
    draw.rectangle([50, 350, 350, 450], outline='blue', width=2)
    draw.text((60, 370), "Data Processing", fill='blue', font=font_text)
    draw.text((60, 395), "(Daft)", fill='blue', font=font_text)
    
    draw.rectangle([400, 350, 700, 450], outline='green', width=2)
    draw.text((410, 370), "Vector Storage", fill='green', font=font_text)
    draw.text((410, 395), "(Weaviate)", fill='green', font=font_text)
    
    draw.rectangle([225, 480, 525, 580], outline='red', width=2)
    draw.text((235, 500), "LLM Inference", fill='red', font=font_text)
    draw.text((235, 525), "(FriendliAI)", fill='red', font=font_text)
    
    # Save the image
    img.save(filepath)
    print(f"Created sample image: {filepath}")


if __name__ == "__main__":
    create_sample_dataset() 