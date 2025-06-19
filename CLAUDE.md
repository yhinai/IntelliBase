# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **IntelliBase Hackathon Project** - a multi-modal AI research assistant that integrates five sponsor technologies:
- **Daft**: High-performance multi-modal data processing
- **Weaviate**: Vector database with hybrid search capabilities  
- **FriendliAI**: Accelerated LLM inference (2-4x faster)
- **Hypermode**: AI agent platform with WebAssembly execution
- **Arize Phoenix**: Complete AI observability and evaluation

## Architecture

```
User Query → Hypermode Agent → FriendliAI LLM
                ↓                    ↓
Arize Phoenix ← Weaviate Vector ← Daft Processing
(Observability)   Database      (Multi-modal)
```

## Development Commands

### Environment Setup
```bash
# Install core dependencies
pip install daft weaviate-client friendli-client arize-phoenix openai streamlit pandas numpy pillow requests

# Install observability components
pip install openinference-instrumentation-openai openinference-instrumentation-weaviate opentelemetry-api opentelemetry-sdk

# Install additional dependencies
pip install PyMuPDF tenacity python-dotenv
```

### Quick Start
```bash
# Create sample data and start system
python create_sample_data.py
python -c "from observability import obs_manager; print('Phoenix started')"
streamlit run streamlit_ui.py --server.port 8501
```

### Testing
```bash
# Run complete system test
python test_system.py

# Start demo sequence
python demo_script.py
```

### Hypermode Agent (Optional)
```bash
# Initialize Modus project
npm install -g @hypermode/modus-cli
modus new intellibase-agent
cd intellibase-agent
modus dev  # Start agent server
```

## Core Implementation Files

### Data Processing Pipeline
- `daft_processor.py`: Multi-modal data processing (PDFs, images, text)
- `weaviate_manager.py`: Vector database operations and hybrid search
- `weaviate_friendliai_integration.py`: Weaviate + FriendliAI integration

### AI Components  
- `hypermode_bridge.py`: Agent orchestration bridge
- `llm_evaluation.py`: Response quality evaluation
- `observability.py`: Phoenix tracing and instrumentation

### Application Layer
- `intellibase_app.py`: Main application orchestrating all components
- `streamlit_ui.py`: Web interface for document upload and querying
- `test_system.py`: End-to-end system testing

## SPARC Development Workflow

This project follows the SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology:

### Core SPARC Commands
- `npx claude-flow sparc modes`: List available SPARC development modes
- `npx claude-flow sparc run <mode> "<task>"`: Execute specific SPARC mode
- `npx claude-flow sparc tdd "<feature>"`: Run complete TDD workflow
- `npx claude-flow sparc info <mode>`: Get detailed mode information

### Standard Build Commands
- `python test_system.py`: Run the test suite
- `streamlit run streamlit_ui.py`: Start the web interface
- `python demo_script.py`: Run hackathon demo sequence

## Key Integration Patterns

### 1. Data Pipeline (Daft → Weaviate)
```python
# Process files with Daft
df = daft.from_glob_path("./data/*")
df = df.with_column("embeddings", generate_embeddings(df["content"]))

# Index in Weaviate
weaviate_client.collections.get("Documents").data.insert_many(df.to_pylist())
```

### 2. Accelerated Inference (Weaviate → FriendliAI)  
```python
# Retrieve context
results = weaviate_client.collections.get("Documents").query.hybrid(query)

# Fast inference with FriendliAI
response = friendli_client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": f"Context: {results}\nQ: {query}"}]
)
```

### 3. Full Observability (Phoenix)
```python
# Instrument everything
from openinference.instrumentation import using_attributes

with using_attributes(session_id="demo", query=user_query):
    results = weaviate_search(user_query)  
    response = friendli_inference(user_query, results)
```

## Environment Variables Required

Create `.env` file with:
```bash
FRIENDLI_TOKEN=your_friendli_token_here
WEAVIATE_CLUSTER_URL=your_weaviate_url_here  
WEAVIATE_API_KEY=your_weaviate_api_key_here
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
HYPERMODE_API_KEY=your_hypermode_key_here  # Optional
```

## Sample Data Structure

The system processes multiple file types:
```
sample_data/
├── research_notes.txt      # Text content
├── project_overview.md     # Markdown documentation  
├── research_paper.pdf      # PDF documents
├── diagram.jpg            # Image files
└── screenshot.png         # Additional images
```

## Hackathon Demo Flow

1. **Data Processing**: Upload mixed media files, process with Daft
2. **Vector Indexing**: Index processed content in Weaviate with hybrid search
3. **Query Processing**: Ask questions, retrieve relevant context
4. **Fast Generation**: Generate responses using FriendliAI's optimized inference
5. **Agent Orchestration**: Coordinate workflow through Hypermode agent
6. **Observability**: Monitor entire pipeline with Phoenix tracing

## Key Features Demonstrated

- **Multi-modal Processing**: Handle PDFs, images, and text uniformly
- **Hybrid Search**: Combine vector similarity with keyword matching
- **Fast Inference**: 2-4x faster LLM responses via FriendliAI
- **Agent Coordination**: Orchestrate complex workflows with Hypermode
- **Complete Observability**: End-to-end tracing and evaluation with Phoenix

## Important Notes

- All integrations use production-tested patterns from official repositories
- System designed for 2-hour hackathon implementation timeline
- Includes comprehensive error handling and retry logic
- Phoenix provides real-time monitoring at http://localhost:6006
- Streamlit UI available at http://localhost:8501

## SPARC Methodology Reminders

- **Red-Green-Refactor**: Write failing tests first, implement minimal code, then optimize
- Use SPARC memory system to maintain context across sessions
- Document architectural decisions for future reference
- Run tests before any commits
- Regular security reviews for authentication and data handling