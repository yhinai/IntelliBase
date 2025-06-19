#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - Streamlit Web Interface
Beautiful and modern UI for the IntelliBase system
"""

import streamlit as st
import time
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Import our main application
try:
    from intellibase_app import IntelliBaseApp
    APP_AVAILABLE = True
except ImportError:
    st.error("❌ IntelliBase application not available")
    APP_AVAILABLE = False


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
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_intellibase():
    """Initialize the IntelliBase application with caching"""
    if not APP_AVAILABLE:
        return None
    
    try:
        app = IntelliBaseApp(data_directory="./sample_data")
        return app
    except Exception as e:
        st.error(f"Failed to initialize IntelliBase: {e}")
        return None


def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🧠 IntelliBase</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Multi-modal AI Research Assistant</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ---
    **IntelliBase** integrates cutting-edge AI technologies to provide fast, accurate responses to your questions:
    - 🔄 **Daft** for high-performance data processing
    - 🔍 **Weaviate** for vector-based semantic search  
    - ⚡ **FriendliAI** for accelerated LLM inference
    - 🤖 **Hypermode** for agent orchestration
    - 📊 **Arize Phoenix** for complete observability
    """)


def render_sidebar(app):
    """Render the sidebar with system information"""
    with st.sidebar:
        st.header("🔧 System Status")
        
        if app is None:
            st.error("❌ IntelliBase not available")
            return
        
        # Get system status
        try:
            status = app.get_system_status()
            
            # Overall readiness
            if status["is_ready"]:
                st.success("✅ System Ready")
            else:
                st.warning("⚠️ Limited Mode")
            
            # Component status
            st.subheader("Components")
            components = status["components"]
            
            for component, available in components.items():
                icon = "✅" if available else "❌"
                label = component.replace("_", " ").title()
                st.write(f"{icon} {label}")
            
            # Statistics
            st.subheader("📊 Statistics")
            stats = status["stats"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", stats["files_processed"])
                st.metric("Chunks", stats["chunks_indexed"])
            
            with col2:
                st.metric("Queries", stats["queries_processed"])
                avg_time = stats["total_generation_time"] / max(1, stats["queries_processed"])
                st.metric("Avg Time", f"{avg_time:.3f}s")
            
            # Phoenix link
            st.subheader("🔍 Observability")
            st.markdown("[📊 Phoenix Dashboard](http://localhost:6006/)")
            
        except Exception as e:
            st.error(f"Error getting status: {e}")


def render_setup_section(app):
    """Render the knowledge base setup section"""
    st.header("📚 Knowledge Base Setup")
    
    if app is None:
        st.error("❌ Cannot setup knowledge base - IntelliBase not available")
        return
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write("Process documents and build the knowledge base for querying.")
    
    with col2:
        if st.button("🔄 Setup Knowledge Base", type="primary"):
            with st.spinner("Setting up knowledge base..."):
                try:
                    success = app.setup_knowledge_base(force_reprocess=True)
                    if success:
                        st.success("✅ Knowledge base setup complete!")
                        st.rerun()
                    else:
                        st.error("❌ Knowledge base setup failed")
                except Exception as e:
                    st.error(f"❌ Setup error: {e}")
    
    with col3:
        if st.button("📊 System Status"):
            st.rerun()


def render_query_section(app):
    """Render the main query interface"""
    st.header("❓ Ask IntelliBase")
    
    if app is None:
        st.error("❌ Cannot process queries - IntelliBase not available")
        return
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning? How do vector databases work?",
        help="Ask any question about the knowledge base"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_results = st.slider("Max search results", 1, 10, 5)
            search_alpha = st.slider("Search balance (vector vs keyword)", 0.0, 1.0, 0.75, 0.05)
        
        with col2:
            temperature = st.slider("Response creativity", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.slider("Max response length", 100, 1000, 500, 50)
    
    # Process query
    if st.button("🚀 Ask IntelliBase", type="primary", disabled=not query):
        if not query.strip():
            st.warning("Please enter a question")
            return
        
        # Show query processing
        with st.spinner("🔍 Searching knowledge base and generating response..."):
            try:
                result = app.query(
                    query,
                    max_results=max_results,
                    search_alpha=search_alpha,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Display results
                render_query_results(result)
                
            except Exception as e:
                st.error(f"❌ Query processing failed: {e}")


def render_query_results(result):
    """Render query results in a nice format"""
    
    if not result["success"]:
        st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        return
    
    # Main answer
    st.subheader("🤖 Answer")
    st.write(result["answer"])
    
    # Response metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Response Time", f"{result['total_time']:.3f}s")
    
    with col2:
        st.metric("📊 Sources Found", result["search_results_count"])
    
    with col3:
        st.metric("🤖 Provider", result["provider"])
    
    with col4:
        st.metric("🧠 Model", result["model"])
    
    # Sources
    if result["sources"]:
        st.subheader("📚 Sources")
        for i, source in enumerate(result["sources"], 1):
            st.write(f"{i}. {source}")
    
    # Mock mode warning
    if result.get("mock"):
        st.warning(f"🎭 Running in mock mode: {result['mock_reason']}")
    
    # Evaluation metrics
    if "evaluation" in result:
        st.subheader("📊 Response Quality")
        eval_data = result["evaluation"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Relevance", f"{eval_data['relevance']:.2f}")
        
        with col2:
            st.metric("Groundedness", f"{eval_data['groundedness']:.2f}")
        
        with col3:
            st.metric("Quality", f"{eval_data['quality']:.2f}")


def render_analytics_section(app):
    """Render analytics and visualization section"""
    st.header("📊 Analytics & Insights")
    
    if app is None:
        st.write("Analytics not available - IntelliBase not initialized")
        return
    
    try:
        status = app.get_system_status()
        
        # Create sample visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Component status pie chart
            component_data = status["components"]
            available_count = sum(component_data.values())
            total_count = len(component_data)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=["Available", "Unavailable"],
                    values=[available_count, total_count - available_count],
                    hole=0.3
                )
            ])
            fig.update_layout(title="Component Availability")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance metrics
            stats = status["stats"]
            
            metrics_df = pd.DataFrame({
                "Metric": ["Files", "Chunks", "Queries"],
                "Count": [
                    stats["files_processed"],
                    stats["chunks_indexed"],
                    stats["queries_processed"]
                ]
            })
            
            fig = px.bar(metrics_df, x="Metric", y="Count", title="System Metrics")
            st.plotly_chart(fig, use_container_width=True)
        
        # Load demo report if available
        demo_report_path = Path("demo_report.json")
        if demo_report_path.exists():
            st.subheader("📋 Latest Demo Report")
            
            with open(demo_report_path) as f:
                demo_data = json.load(f)
            
            # Show demo timestamp
            demo_time = datetime.fromtimestamp(demo_data["demo_timestamp"])
            st.write(f"**Demo Run:** {demo_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show demo queries and results
            if "query_results" in demo_data:
                results_df = pd.DataFrame([
                    {
                        "Query": r["query"][:50] + "..." if len(r["query"]) > 50 else r["query"],
                        "Response Time (s)": r["total_time"],
                        "Sources": len(r["sources"]),
                        "Success": "✅" if r["success"] else "❌"
                    }
                    for r in demo_data["query_results"]
                ])
                
                st.dataframe(results_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")


def render_demo_section(app):
    """Render demo section"""
    st.header("🎭 Run Demo")
    
    if app is None:
        st.error("❌ Cannot run demo - IntelliBase not available")
        return
    
    st.write("""
    Run a comprehensive demo that:
    1. Sets up the knowledge base
    2. Tests multiple query types
    3. Shows system capabilities
    4. Generates a detailed report
    """)
    
    if st.button("🚀 Run Full Demo", type="primary"):
        with st.spinner("Running IntelliBase demo..."):
            try:
                # Capture demo output
                import io
                import contextlib
                
                output_buffer = io.StringIO()
                
                with contextlib.redirect_stdout(output_buffer):
                    app.run_demo()
                
                demo_output = output_buffer.getvalue()
                
                st.success("✅ Demo completed successfully!")
                
                # Show demo output
                with st.expander("📋 Demo Output"):
                    st.text(demo_output)
                
                # Reload analytics
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Demo failed: {e}")


def render_system_test_section(app):
    """Render comprehensive system testing section with plotly visualizations"""
    st.header("🧪 System Testing & Performance")
    
    st.write("""
    Comprehensive system testing with performance monitoring and visualizations.
    This section runs tests on all components and generates interactive charts.
    """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write("**Test Features:**")
        st.write("- Component availability testing")
        st.write("- Integration testing")
        st.write("- Performance benchmarking")
        st.write("- Load testing")
        st.write("- Interactive Plotly visualizations")
    
    with col2:
        if st.button("🧪 Run System Test", type="primary"):
            with st.spinner("Running comprehensive system tests..."):
                try:
                    # Import and run the comprehensive test
                    import subprocess
                    import sys
                    
                    # Run the comprehensive test
                    result = subprocess.run([
                        sys.executable, "comprehensive_system_test.py"
                    ], capture_output=True, text=True, cwd=".")
                    
                    if result.returncode == 0:
                        st.success("✅ System tests completed successfully!")
                        
                        # Load and display test results
                        display_test_results()
                    else:
                        st.error(f"❌ System tests failed: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"❌ Error running tests: {e}")
    
    with col3:
        if st.button("📊 View Latest Results"):
            display_test_results()
    
    # Check if test results exist and display them
    test_results_path = Path("test_results")
    if test_results_path.exists():
        st.divider()
        display_test_results()


def display_test_results():
    """Display comprehensive test results with plotly visualizations"""
    
    test_results_path = Path("test_results")
    
    if not test_results_path.exists():
        st.info("No test results found. Run a system test first.")
        return
    
    # Load test results
    results_file = test_results_path / "comprehensive_test_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                test_data = json.load(f)
            
            # Display summary
            summary = test_data.get("summary", {})
            metadata = test_data.get("metadata", {})
            
            st.subheader("📋 Test Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = summary.get("overall_status", "Unknown")
                st.metric("Overall Status", status)
            
            with col2:
                success_rate = summary.get("success_rate", 0)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col3:
                duration = summary.get("total_duration", 0)
                st.metric("Duration", f"{duration:.2f}s")
            
            with col4:
                perf_score = summary.get("performance_score", 0)
                st.metric("Performance Score", f"{perf_score:.1f}/100")
            
            # Component status
            components = test_data.get("components", {})
            if components:
                st.subheader("🧩 Component Status")
                
                component_df = pd.DataFrame([
                    {
                        "Component": comp.replace("_", " ").title(),
                        "Status": data.get("status", "Unknown"),
                        "Available": "✅" if data.get("available", False) else "❌"
                    }
                    for comp, data in components.items()
                ])
                
                st.dataframe(component_df, use_container_width=True)
            
            # Load and display visualizations
            dashboard_file = test_results_path / "test_dashboard.html"
            if dashboard_file.exists():
                st.subheader("📊 Interactive Dashboard")
                
                with open(dashboard_file, 'r') as f:
                    dashboard_html = f.read()
                
                # Display the plotly dashboard
                st.components.v1.html(dashboard_html, height=600, scrolling=True)
            
            # Performance metrics
            metrics_file = test_results_path / "performance_metrics.csv"
            if metrics_file.exists():
                st.subheader("⚡ Performance Metrics")
                
                metrics_df = pd.read_csv(metrics_file)
                
                if not metrics_df.empty:
                    # Create performance chart
                    fig = px.line(
                        metrics_df, 
                        x="timestamp", 
                        y="throughput",
                        color="operation",
                        title="Performance Over Time",
                        labels={"throughput": "Throughput", "timestamp": "Time"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show metrics table
                    st.dataframe(metrics_df, use_container_width=True)
            
            # Recommendations
            recommendations = summary.get("recommendations", [])
            if recommendations:
                st.subheader("💡 Recommendations")
                for rec in recommendations:
                    st.write(f"- {rec}")
            
            # Additional visualizations
            timeline_file = test_results_path / "component_timeline.html"
            perf_dist_file = test_results_path / "performance_distribution.html"
            
            if timeline_file.exists() or perf_dist_file.exists():
                st.subheader("📈 Additional Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if timeline_file.exists():
                        if st.button("📊 Component Timeline"):
                            with open(timeline_file, 'r') as f:
                                timeline_html = f.read()
                            st.components.v1.html(timeline_html, height=400)
                
                with col2:
                    if perf_dist_file.exists():
                        if st.button("📈 Performance Distribution"):
                            with open(perf_dist_file, 'r') as f:
                                perf_html = f.read()
                            st.components.v1.html(perf_html, height=400)
        
        except Exception as e:
            st.error(f"Error loading test results: {e}")
    
    else:
        st.info("No test results file found. Run a system test to generate results.")


def main():
    """Main Streamlit application"""
    
    # Initialize IntelliBase
    app = initialize_intellibase()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar(app)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["❓ Query", "📚 Setup", "📊 Analytics", "🎭 Demo", "🧪 System Test"])
    
    with tab1:
        render_query_section(app)
    
    with tab2:
        render_setup_section(app)
    
    with tab3:
        render_analytics_section(app)
    
    with tab4:
        render_demo_section(app)
    
    with tab5:
        render_system_test_section(app)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🧠 IntelliBase - Powered by Daft, Weaviate, FriendliAI, Hypermode & Arize Phoenix
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 