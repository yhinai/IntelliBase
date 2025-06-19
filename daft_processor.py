#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - Daft Data Processing
Multi-modal data processing using Daft for PDFs, images, and text
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import time

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    print("Daft not available - using fallback processing")
    DAFT_AVAILABLE = False

import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO

try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    print("PyMuPDF not available - PDF processing disabled")
    PDF_AVAILABLE = False

from observability import trace_data_processing, obs_manager


class DaftProcessor:
    """Multi-modal data processor using Daft for high-performance processing"""
    
    def __init__(self):
        self.daft_available = DAFT_AVAILABLE
        self.pdf_available = PDF_AVAILABLE
        if self.daft_available:
            self.setup_udfs()
        print(f"🔧 DaftProcessor initialized (Daft: {self.daft_available}, PDF: {self.pdf_available})")
    
    def setup_udfs(self):
        """Define User Defined Functions for multimodal processing with Daft"""
        if not self.daft_available:
            return
            
        @daft.udf(return_type=daft.DataType.string())
        def extract_pdf_text(pdf_path):
            """Extract text from PDF files"""
            if not PDF_AVAILABLE:
                return "PDF processing not available"
                
            try:
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except Exception as e:
                return f"Error processing PDF: {str(e)}"
        
        @daft.udf(return_type=daft.DataType.string())
        def extract_image_features(image_path):
            """Extract features from images"""
            try:
                if str(image_path).startswith('http'):
                    response = requests.get(image_path)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(image_path)
                
                # Basic feature extraction
                width, height = img.size
                mode = img.mode
                
                # Calculate basic statistics
                if img.mode == 'RGB':
                    img_array = np.array(img)
                    mean_color = img_array.mean(axis=(0, 1))
                    color_info = f"RGB({mean_color[0]:.0f},{mean_color[1]:.0f},{mean_color[2]:.0f})"
                else:
                    color_info = img.mode
                
                # Convert to description
                description = f"Image: {width}x{height} pixels, Mode: {mode}, Avg Color: {color_info}"
                
                # Try to extract any text from the image filename or context
                if 'diagram' in str(image_path).lower():
                    description += " | Contains: System architecture diagram"
                elif 'flow' in str(image_path).lower():
                    description += " | Contains: Process flow diagram"
                elif 'chart' in str(image_path).lower():
                    description += " | Contains: Chart or graph"
                
                return description
                
            except Exception as e:
                return f"Error processing image: {str(e)}"
        
        @daft.udf(return_type=daft.DataType.string())
        def determine_content_type(file_path):
            """Determine file type from extension"""
            path_str = str(file_path).lower()
            if path_str.endswith(('.pdf')):
                return "pdf"
            elif path_str.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                return "image"
            elif path_str.endswith(('.txt', '.md', '.rst')):
                return "text"
            elif path_str.endswith(('.json', '.yaml', '.yml')):
                return "data"
            else:
                return "unknown"
        
        @daft.udf(return_type=daft.DataType.list(daft.DataType.string()))
        def chunk_text(text, chunk_size=1000):
            """Split text into chunks for vector processing"""
            if not text or len(text) < chunk_size:
                return [text] if text else [""]
            
            chunks = []
            # Split by sentences first, then chunk
            sentences = text.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # Store UDFs for use
        self.extract_pdf_text = extract_pdf_text
        self.extract_image_features = extract_image_features
        self.determine_content_type = determine_content_type
        self.chunk_text = chunk_text
    
    @trace_data_processing(component="daft", operation="process_multimodal")
    def process_multimodal_data(self, data_path: str = "../sample_data/*") -> Any:
        """Process mixed media files using Daft"""
        
        print(f"🔄 Processing multimodal data from: {data_path}")
        
        if self.daft_available:
            return self._process_with_daft(data_path)
        else:
            return self._process_fallback(data_path)
    
    def _process_with_daft(self, data_path: str):
        """Process using Daft engine"""
        try:
            # Load files from glob pattern
            print(f"📁 Loading files from: {data_path}")
            df = daft.from_glob_path(data_path)
            
            # Determine content types
            print("🔍 Determining content types...")
            df = df.with_column("content_type", self.determine_content_type(df["path"]))
            
            # Extract content based on type
            print("📝 Extracting content...")
            df = df.with_column("extracted_content",
                daft.when(df["content_type"] == "pdf")
                .then(self.extract_pdf_text(df["path"]))
                .when(df["content_type"] == "image")
                .then(self.extract_image_features(df["path"]))
                .when(df["content_type"] == "text")
                .then(df["path"].str.read_text())
                .otherwise("Unsupported file type")
            )
            
            # Add metadata
            print("📊 Adding metadata...")
            df = df.with_column("file_size", df["path"].str.stat().size)
            df = df.with_column("processed_at", daft.lit(time.time()))
            
            # Create unique IDs
            df = df.with_column("file_id", 
                df["path"].str.slice(0, 50).str.sha256()
            )
            
            print("✅ Daft processing complete")
            return df
            
        except Exception as e:
            print(f"❌ Daft processing failed: {e}")
            return self._process_fallback(data_path)
    
    def _process_fallback(self, data_path: str):
        """Fallback processing without Daft"""
        print("🔄 Using fallback processing...")
        
        # Convert glob pattern to actual files
        if "*" in data_path:
            base_path = Path(data_path.replace("*", "")).parent
            files = list(base_path.glob("*"))
        else:
            files = [Path(data_path)]
        
        results = []
        for file_path in files:
            if file_path.is_file():
                try:
                    content_type = self._determine_content_type_fallback(file_path)
                    extracted_content = self._extract_content_fallback(file_path, content_type)
                    
                    results.append({
                        "path": str(file_path),
                        "content_type": content_type,
                        "extracted_content": extracted_content,
                        "file_size": file_path.stat().st_size,
                        "processed_at": time.time(),
                        "file_id": hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
                    })
                except Exception as e:
                    print(f"⚠️ Error processing {file_path}: {e}")
        
        return results
    
    def _determine_content_type_fallback(self, file_path: Path) -> str:
        """Fallback content type determination"""
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return "pdf"
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return "image"
        elif suffix in ['.txt', '.md', '.rst']:
            return "text"
        else:
            return "unknown"
    
    def _extract_content_fallback(self, file_path: Path, content_type: str) -> str:
        """Fallback content extraction"""
        try:
            if content_type == "pdf" and PDF_AVAILABLE:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
                
            elif content_type == "image":
                img = Image.open(file_path)
                width, height = img.size
                return f"Image: {width}x{height}, Mode: {img.mode}, File: {file_path.name}"
                
            elif content_type == "text":
                return file_path.read_text(encoding='utf-8')
                
            else:
                return f"Unsupported file type: {content_type}"
                
        except Exception as e:
            return f"Error extracting content: {e}"
    
    @trace_data_processing(component="daft", operation="prepare_for_vector_db")
    def process_for_vector_db(self, processed_data) -> Any:
        """Prepare data for vector database ingestion"""
        
        print("🔄 Preparing data for vector database...")
        
        if self.daft_available and hasattr(processed_data, 'where'):
            return self._prepare_daft_for_vector_db(processed_data)
        else:
            return self._prepare_fallback_for_vector_db(processed_data)
    
    def _prepare_daft_for_vector_db(self, df):
        """Prepare Daft DataFrame for vector DB"""
        try:
            # Filter out failed extractions
            df = df.where(~df["extracted_content"].str.startswith("Error"))
            
            # Apply chunking
            print("✂️ Chunking text content...")
            df = df.with_column("text_chunks", self.chunk_text(df["extracted_content"]))
            
            # Explode chunks into separate rows
            df = df.explode("text_chunks")
            df = df.with_column("content", df["text_chunks"])
            
            # Create unique chunk IDs
            df = df.with_column("chunk_id", 
                df["file_id"].str.cat(df["content"].str.slice(0, 20).str.sha256())
            )
            
            print("✅ Vector DB preparation complete")
            return df
            
        except Exception as e:
            print(f"❌ Vector DB preparation failed: {e}")
            return self._prepare_fallback_for_vector_db(df.collect() if hasattr(df, 'collect') else df)
    
    def _prepare_fallback_for_vector_db(self, data):
        """Fallback preparation for vector DB"""
        if isinstance(data, list):
            results = []
            for item in data:
                if item.get("extracted_content") and not item["extracted_content"].startswith("Error"):
                    chunks = self._chunk_text_fallback(item["extracted_content"])
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            results.append({
                                "content": chunk,
                                "content_type": item["content_type"],
                                "path": item["path"],
                                "file_size": item["file_size"],
                                "processed_at": item["processed_at"],
                                "file_id": item["file_id"],
                                "chunk_id": f"{item['file_id']}_chunk_{i}"
                            })
            return results
        else:
            return data
    
    def _chunk_text_fallback(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Fallback text chunking"""
        if not text or len(text) < chunk_size:
            return [text] if text else [""]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


if __name__ == "__main__":
    # Test the Daft processor
    print("🚀 Testing IntelliBase Daft Data Processing")
    print("=" * 50)
    
    # Initialize processor
    processor = DaftProcessor()
    
    # Process sample data
    print("\n📊 Processing sample data...")
    processed_data = processor.process_multimodal_data("../sample_data/*")
    
    # Prepare for vector DB
    print("\n🔄 Preparing for vector database...")
    vector_ready_data = processor.process_for_vector_db(processed_data)
    
    # Display results
    if hasattr(vector_ready_data, 'collect'):
        # Daft DataFrame
        results = vector_ready_data.collect()
        print(f"\n✅ Processed {len(results)} chunks")
        for i, item in enumerate(results[:3]):  # Show first 3
            print(f"\nChunk {i+1}:")
            print(f"  Content Type: {item.get('content_type', 'N/A')}")
            print(f"  Content Preview: {str(item.get('content', ''))[:100]}...")
            print(f"  Source: {item.get('path', 'N/A')}")
    else:
        # Fallback list
        print(f"\n✅ Processed {len(vector_ready_data)} chunks")
        for i, item in enumerate(vector_ready_data[:3]):  # Show first 3
            print(f"\nChunk {i+1}:")
            print(f"  Content Type: {item.get('content_type', 'N/A')}")
            print(f"  Content Preview: {str(item.get('content', ''))[:100]}...")
            print(f"  Source: {item.get('path', 'N/A')}")
    
    print("\n✅ Daft processing test complete!") 