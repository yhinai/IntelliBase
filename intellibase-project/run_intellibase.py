#!/usr/bin/env python3
"""
IntelliBase Quick Start Script
"""
import subprocess
import sys
import time
from pathlib import Path

def main():
    """Quick start IntelliBase system"""
    
    print("🚀 IntelliBase Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("intellibase_app.py").exists():
        print("❌ Please run this script from the intellibase-project directory")
        return 1
    
    # Create sample data if needed
    if not Path("sample_data").exists():
        print("📊 Creating sample data...")
        try:
            result = subprocess.run([sys.executable, "create_sample_data.py"], 
                                  capture_output=True, text=True, check=True)
            print("✅ Sample data created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create sample data: {e.stderr}")
            return 1
    
    print("\nChoose how to run IntelliBase:")
    print("1. 🌐 Web Interface (Streamlit)")
    print("2. 🧪 Run Tests")
    print("3. 🖥️  Command Line Demo")
    print("4. 📊 Phoenix Dashboard Only")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_streamlit()
    elif choice == "2":
        run_tests()
    elif choice == "3":
        run_cli_demo()
    elif choice == "4":
        run_phoenix_only()
    else:
        print("❌ Invalid choice")
        return 1
    
    return 0

def run_streamlit():
    """Launch Streamlit web interface"""
    
    print("\n🌐 Starting Streamlit web interface...")
    print("📱 Opening at: http://localhost:8501")
    print("🔬 Phoenix available at: http://localhost:6006")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_ui.py", 
                       "--server.port", "8501"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")

def run_tests():
    """Run comprehensive test suite"""
    
    print("\n🧪 Running comprehensive tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              check=True, text=True)
        print("\n✅ All tests completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code: {e.returncode}")

def run_cli_demo():
    """Run command line demo"""
    
    print("\n🖥️  Running command line demo...")
    
    try:
        result = subprocess.run([sys.executable, "intellibase_app.py"], 
                              check=True, text=True)
        print("\n✅ Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Demo failed with exit code: {e.returncode}")

def run_phoenix_only():
    """Start Phoenix dashboard only"""
    
    print("\n🔬 Starting Phoenix observability dashboard...")
    print("📊 Dashboard: http://localhost:6006")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Import and start Phoenix
        import phoenix as px
        session = px.launch_app()
        print(f"✅ Phoenix started at: {session.url}")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Phoenix stopped")
    except ImportError:
        print("❌ Phoenix not available")
    except Exception as e:
        print(f"❌ Error starting Phoenix: {e}")

if __name__ == "__main__":
    exit(main())