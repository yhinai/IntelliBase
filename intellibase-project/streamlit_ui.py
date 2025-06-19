#!/usr/bin/env python3
"""
Streamlit web interface for IntelliBase
"""
import streamlit as st
import asyncio
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime

# Import the main application
from intellibase_app import IntelliBaseApp

# Page configuration
st.set_page_config(
    page_title="IntelliBase - AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .source-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app' not in st.session_state:
    with st.spinner("🚀 Initializing IntelliBase..."):
        st.session_state.app = IntelliBaseApp()
        st.session_state.query_history = []

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🧠 IntelliBase</h1>
        <p style="color: white; margin: 0;">Multi-modal AI Research Assistant</p>
        <small style="color: #e0e0e0;">Powered by Daft • Weaviate • FriendliAI • Hypermode • Arize Phoenix</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Control")
        
        # System Status
        if st.button("🔄 Refresh Status"):
            status = st.session_state.app.system_status
            
            st.subheader("📊 System Status")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", status["documents_indexed"])
                st.metric("Queries", status["metrics"]["total_queries"])
            with col2:
                st.metric("Success Rate", f"{(status['metrics']['successful_queries'] / max(status['metrics']['total_queries'], 1) * 100):.1f}%")
                st.metric("Avg Time", f"{status['metrics']['average_response_time']:.3f}s")
            
            # Component Status
            st.subheader("🔗 Components")
            components = {
                "🔍 Weaviate": "✅ Connected",
                "🔥 FriendliAI": "✅ Mock" if status["using_mock_friendli"] else "✅ Connected",
                "🔬 Phoenix": "✅ Active",
                "📊 Daft": "✅ Active"
            }
            
            for component, status_text in components.items():
                st.write(f"{component}: {status_text}")
            
            # Phoenix Dashboard Link
            st.markdown(f"🔬 [Phoenix Dashboard]({status['phoenix_url']})")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        search_limit = st.slider("Search Results", 1, 10, 5)
        show_sources = st.checkbox("Show Sources", True)
        show_metrics = st.checkbox("Show Metrics", True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📁 Documents", "📊 Analytics", "🧪 Testing"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        document_management()
    
    with tab3:
        analytics_dashboard()
    
    with tab4:
        testing_interface()

def chat_interface():
    """Chat interface for querying the knowledge base"""
    
    st.header("💬 Ask Questions")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "What would you like to know?",
            placeholder="e.g., What is machine learning? How do vector databases work?",
            key="query_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Example queries
    st.write("**Example questions:**")
    example_queries = [
        "What is machine learning?",
        "How do vector databases work?", 
        "What is RAG and how does it work?",
        "What are the benefits of observability?",
        "How does Daft process multimodal data?"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        if cols[i].button(f"💡 {example}", key=f"example_{i}"):
            st.session_state.query_input = example
            st.rerun()
    
    # Process query
    if (search_button or query) and query:
        with st.spinner("🤔 Thinking..."):
            try:
                # Process query
                result = asyncio.run(st.session_state.app.process_query(query))
                
                # Add to history
                st.session_state.query_history.insert(0, {
                    "query": query,
                    "result": result,
                    "timestamp": datetime.now()
                })
                
                # Display result
                display_query_result(result)
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
    
    # Query History
    if st.session_state.query_history:
        st.divider()
        st.subheader("📝 Recent Queries")
        
        for i, entry in enumerate(st.session_state.query_history[:5]):
            with st.expander(f"Q: {entry['query'][:50]}..." if len(entry['query']) > 50 else f"Q: {entry['query']}"):
                display_query_result(entry['result'])

def display_query_result(result):
    """Display query result with answer, sources, and metrics"""
    
    # Answer
    st.subheader("🎯 Answer")
    st.write(result["answer"])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processing Time", f"{result['processing_time']:.3f}s")
    with col2:
        st.metric("Quality Score", f"{result['evaluation']['overall_quality']:.2f}/5")
    with col3:
        st.metric("Sources Found", len(result["sources"]))
    with col4:
        st.metric("Model", result["model_used"])
    
    # Sources
    if result["sources"]:
        st.subheader("📚 Sources")
        
        for i, source in enumerate(result["sources"]):
            with st.expander(f"📄 {Path(source['source']).name} (Score: {source['score']:.3f})"):
                st.write(source["content"])
                st.caption(f"Type: {source['content_type']} | Chunk ID: {source['chunk_id']}")
    
    # Detailed metrics (expandable)
    with st.expander("📊 Detailed Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Evaluation Scores:**")
            for key, value in result["evaluation"].items():
                st.write(f"- {key.replace('_', ' ').title()}: {value:.2f}")
        
        with col2:
            st.write("**Technical Details:**")
            st.write(f"- Generation Time: {result['generation_time']:.3f}s")
            st.write(f"- Using Mock: {result['using_mock']}")
            st.write(f"- Timestamp: {result['timestamp']}")

def document_management():
    """Document upload and management interface"""
    
    st.header("📁 Document Management")
    
    # File upload
    st.subheader("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['txt', 'md', 'pdf', 'jpg', 'png', 'jpeg'],
        help="Supported formats: TXT, MD, PDF, JPG, PNG"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Process Uploaded Files") and uploaded_files:
            process_uploaded_files(uploaded_files)
    
    with col2:
        if st.button("📁 Process Sample Data"):
            process_sample_data()
    
    with col3:
        if st.button("🔍 Show Current Documents"):
            show_document_stats()
    
    st.divider()
    
    # Document statistics
    show_document_stats()

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    
    with st.spinner("📊 Processing uploaded files..."):
        try:
            # Save uploaded files
            data_dir = Path("./uploaded_data")
            data_dir.mkdir(exist_ok=True)
            
            saved_files = []
            for file in uploaded_files:
                file_path = data_dir / file.name
                with open(file_path, "wb") as f:
                    f.write(file.read())
                saved_files.append(str(file_path))
            
            # Process files
            count = st.session_state.app.ingest_documents("./uploaded_data/*")
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} files into {count} chunks!")
            
            # Show processed files
            with st.expander("📄 Processed Files"):
                for file_path in saved_files:
                    st.write(f"- {Path(file_path).name}")
                    
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")

def process_sample_data():
    """Process sample data"""
    
    with st.spinner("📊 Processing sample data..."):
        try:
            count = st.session_state.app.ingest_documents("./sample_data/*")
            st.success(f"✅ Successfully processed sample data into {count} chunks!")
        except Exception as e:
            st.error(f"❌ Error processing sample data: {str(e)}")

def show_document_stats():
    """Show document statistics"""
    
    try:
        stats = st.session_state.app.weaviate_manager.get_collection_stats()
        
        st.subheader("📊 Document Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats.get("total_documents", 0))
        with col2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        with col3:
            st.metric("Avg Content Length", f"{stats.get('average_content_length', 0):.0f} chars")
        
        if stats.get("content_types"):
            st.write("**Content Types:**")
            for content_type in stats["content_types"]:
                st.write(f"- {content_type}")
                
    except Exception as e:
        st.error(f"❌ Error retrieving stats: {str(e)}")

def analytics_dashboard():
    """Analytics and monitoring dashboard"""
    
    st.header("📊 Analytics Dashboard")
    
    try:
        # Get analytics
        analytics = st.session_state.app.get_analytics()
        
        # System Health
        st.subheader("🏥 System Health")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", analytics["system_health"]["status"].title())
        with col2:
            st.metric("Active Components", len(analytics["system_health"]["components"]))
        with col3:
            phoenix_url = analytics["phoenix_dashboard"]
            st.markdown(f"[🔬 Phoenix Dashboard]({phoenix_url})")
        
        # Performance Metrics
        st.subheader("⚡ Performance Metrics")
        
        metrics = analytics["performance_metrics"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", metrics["total_queries"])
        with col2:
            st.metric("Successful", metrics["successful_queries"])
        with col3:
            st.metric("Failed", metrics["failed_queries"])
        with col4:
            success_rate = (metrics["successful_queries"] / max(metrics["total_queries"], 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Data Statistics
        st.subheader("📈 Data Statistics")
        
        data_stats = analytics["data_statistics"]
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Documents Indexed", data_stats["total_documents"])
            st.metric("Avg Content Length", f"{data_stats['average_content_length']:.0f} chars")
        
        with col2:
            if data_stats["content_types"]:
                st.write("**Content Types:**")
                for content_type in data_stats["content_types"]:
                    st.write(f"- {content_type}")
        
        # Component Status
        st.subheader("🔗 Component Status")
        
        components = analytics["system_health"]["components"]
        cols = st.columns(len(components))
        
        for i, (component, status) in enumerate(components.items()):
            with cols[i]:
                color = "🟢" if status == "active" or status == "connected" else "🟡"
                st.write(f"{color} **{component.title()}**")
                st.write(f"Status: {status}")
                
    except Exception as e:
        st.error(f"❌ Error loading analytics: {str(e)}")

def testing_interface():
    """Testing and debugging interface"""
    
    st.header("🧪 Testing Interface")
    
    # Batch query testing
    st.subheader("📝 Batch Query Testing")
    
    test_queries = st.text_area(
        "Enter test queries (one per line):",
        value="What is machine learning?\nHow do vector databases work?\nWhat is RAG?",
        height=100
    )
    
    if st.button("🚀 Run Batch Test"):
        queries = [q.strip() for q in test_queries.split('\n') if q.strip()]
        
        if queries:
            with st.spinner(f"🔄 Processing {len(queries)} queries..."):
                try:
                    results = asyncio.run(st.session_state.app.batch_process_queries(queries))
                    
                    st.success(f"✅ Processed {len(results)}/{len(queries)} queries successfully")
                    
                    # Show results
                    for i, result in enumerate(results):
                        with st.expander(f"Query {i+1}: {result['query'][:50]}..."):
                            st.write(f"**Answer:** {result['answer'][:200]}...")
                            st.write(f"**Processing Time:** {result['processing_time']:.3f}s")
                            st.write(f"**Quality Score:** {result['evaluation']['overall_quality']:.2f}")
                            
                except Exception as e:
                    st.error(f"❌ Batch test failed: {str(e)}")
    
    st.divider()
    
    # Search testing
    st.subheader("🔍 Search Testing")
    
    search_query = st.text_input("Search query:", placeholder="vector database")
    search_limit = st.slider("Number of results:", 1, 20, 5)
    
    if st.button("🔍 Test Search") and search_query:
        try:
            results = st.session_state.app.search_documents(search_query, search_limit)
            
            st.write(f"Found {len(results)} results:")
            
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1}: Score {result.score:.3f}"):
                    st.write(f"**Source:** {result.source_file}")
                    st.write(f"**Content:** {result.content[:300]}...")
                    st.write(f"**Type:** {result.content_type}")
                    
        except Exception as e:
            st.error(f"❌ Search test failed: {str(e)}")
    
    st.divider()
    
    # System diagnostics
    st.subheader("🔧 System Diagnostics")
    
    if st.button("🏥 Run Diagnostics"):
        try:
            status = st.session_state.app.system_status
            
            st.json(status)
            
        except Exception as e:
            st.error(f"❌ Diagnostics failed: {str(e)}")

if __name__ == "__main__":
    main()