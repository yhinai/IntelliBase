#!/usr/bin/env python3
"""
IntelliBase Hackathon Project - Simplified Data Processing
Multi-modal data processing with fallback support
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import time
import json

# Core imports that should be available
import pandas as pd
import numpy as np
from PIL import Image

# Optional imports with fallbacks
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    print("PyMuPDF not available - PDF processing disabled")
    PDF_AVAILABLE = False

try:
    import daft
    DAFT_AVAILABLE = True
    print("✅ Daft available for high-performance processing")
except ImportError:
    print("⚠️ Daft not available - using pandas fallback")
    DAFT_AVAILABLE = False

try:
    from observability import trace_data_processing, obs_manager
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    print("⚠️ Observability not available - continuing without tracing")
    OBSERVABILITY_AVAILABLE = False
    
    # Create dummy decorators
    def trace_data_processing(**kwargs):
        def decorator(func):
            return func
        return decorator


class IntelliBaseDataProcessor:
    """Simplified multi-modal data processor for the IntelliBase system"""
    
    def __init__(self):
        self.daft_available = DAFT_AVAILABLE
        self.pdf_available = PDF_AVAILABLE
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "errors": 0
        }
        print(f"🔧 IntelliBaseDataProcessor initialized")
        print(f"   - Daft: {self.daft_available}")
        print(f"   - PDF processing: {self.pdf_available}")
        print(f"   - Observability: {OBSERVABILITY_AVAILABLE}")
    
    @trace_data_processing(component="processor", operation="process_directory")
    def process_directory(self, data_path: str = "../sample_data", exclude_dirs=None) -> List[Dict[str, Any]]:
        """Process all files in a directory recursively, skipping excluded folders"""
        
        print(f"🔄 Processing directory: {data_path}")
        
        if exclude_dirs is None:
            exclude_dirs = ['.git', 'Library', 'node_modules', 'Applications', 'Pictures', 'Movies', 'Music', 'Public', 'Downloads', 'Desktop', 'Documents', 'OneDrive', 'Dropbox', 'Parallels', 'VirtualBox VMs', 'venv', '__pycache__']
        
        base_path = Path(data_path)
        if not base_path.exists():
            print(f"❌ Directory not found: {base_path}")
            return []
        
        results = []
        file_count = 0
        for root, dirs, files in os.walk(base_path):
            # Exclude system/user folders
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    try:
                        file_result = self._process_single_file(file_path)
                        if file_result:
                            results.append(file_result)
                            self.stats["files_processed"] += 1
                            file_count += 1
                            if file_count % 100 == 0:
                                print(f"   Processed {file_count} files so far...")
                    except Exception as e:
                        print(f"⚠️ Error processing {file_path.name}: {e}")
                        self.stats["errors"] += 1
        print(f"✅ Processed {len(results)} files successfully")
        return results
    
    def _process_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single file and extract content"""
        
        content_type = self._determine_content_type(file_path)
        print(f"📝 Processing {file_path.name} ({content_type})")
        
        # Extract content based on file type
        extracted_content = self._extract_content(file_path, content_type)
        
        if not extracted_content or extracted_content.startswith("Error"):
            return None
        
        # Create file metadata
        file_stats = file_path.stat()
        file_id = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
        
        return {
            "file_id": file_id,
            "path": str(file_path),
            "filename": file_path.name,
            "content_type": content_type,
            "extracted_content": extracted_content,
            "file_size": file_stats.st_size,
            "modified_time": file_stats.st_mtime,
            "processed_at": time.time()
        }
    
    def _determine_content_type(self, file_path: Path) -> str:
        """Determine file type from extension"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return "pdf"
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return "image"
        elif suffix in ['.txt', '.md', '.rst']:
            return "text"
        elif suffix in ['.json', '.yaml', '.yml']:
            return "data"
        else:
            return "unknown"
    
    def _extract_content(self, file_path: Path, content_type: str) -> str:
        """Extract content from file based on type"""
        
        try:
            if content_type == "pdf":
                return self._extract_pdf_content(file_path)
            elif content_type == "image":
                return self._extract_image_content(file_path)
            elif content_type == "text":
                return self._extract_text_content(file_path)
            elif content_type == "data":
                return self._extract_data_content(file_path)
            else:
                return f"Unsupported file type: {content_type}"
                
        except Exception as e:
            return f"Error extracting content from {file_path.name}: {e}"
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        if not self.pdf_available:
            return "PDF processing not available - PyMuPDF not installed"
        
        try:
            doc = fitz.open(file_path)
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            doc.close()
            
            return text.strip() if text.strip() else "No text extracted from PDF"
            
        except Exception as e:
            return f"Error processing PDF: {e}"
    
    def _extract_image_content(self, file_path: Path) -> str:
        """Extract metadata and description from images"""
        try:
            img = Image.open(file_path)
            width, height = img.size
            mode = img.mode
            format_type = img.format
            
            # Basic description
            description = f"Image file: {file_path.name}\n"
            description += f"Dimensions: {width}x{height} pixels\n"
            description += f"Color mode: {mode}\n"
            description += f"Format: {format_type}\n"
            
            # Try to get color information
            if mode == 'RGB':
                try:
                    img_array = np.array(img)
                    mean_color = img_array.mean(axis=(0, 1))
                    description += f"Average color: RGB({mean_color[0]:.0f}, {mean_color[1]:.0f}, {mean_color[2]:.0f})\n"
                except:
                    pass
            
            # Contextual information based on filename
            filename_lower = file_path.name.lower()
            if 'diagram' in filename_lower:
                description += "Content type: System architecture diagram"
            elif 'flow' in filename_lower:
                description += "Content type: Process flow diagram"  
            elif 'chart' in filename_lower:
                description += "Content type: Chart or graph"
            elif 'screenshot' in filename_lower:
                description += "Content type: Screenshot or interface capture"
            else:
                description += "Content type: General image"
            
            return description
            
        except Exception as e:
            return f"Error processing image: {e}"
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from text files"""
        try:
            content = file_path.read_text(encoding='utf-8')
            return content
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding='latin-1')
                return content
            except Exception as e:
                return f"Error reading text file: {e}"
        except Exception as e:
            return f"Error processing text file: {e}"
    
    def _extract_data_content(self, file_path: Path) -> str:
        """Extract content from structured data files"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Try to parse and format nicely
            if file_path.suffix.lower() == '.json':
                try:
                    data = json.loads(content)
                    return f"JSON file with {len(data)} top-level items:\n{json.dumps(data, indent=2)[:1000]}..."
                except:
                    return content
            else:
                return content
                
        except Exception as e:
            return f"Error processing data file: {e}"
    
    @trace_data_processing(component="processor", operation="create_chunks")
    def create_chunks(self, processed_files: List[Dict[str, Any]], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Split content into chunks for vector database"""
        
        print(f"✂️ Creating chunks (max size: {chunk_size} chars)")
        
        all_chunks = []
        
        for file_data in processed_files:
            content = file_data.get("extracted_content", "")
            if not content or len(content) < 10:  # Skip empty or very short content
                continue
            
            chunks = self._split_into_chunks(content, chunk_size)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only include non-empty chunks
                    chunk_data = {
                        "chunk_id": f"{file_data['file_id']}_chunk_{i}",
                        "content": chunk.strip(),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content_type": file_data["content_type"],
                        "source_file": file_data["filename"],
                        "source_path": file_data["path"],
                        "file_id": file_data["file_id"],
                        "processed_at": time.time()
                    }
                    all_chunks.append(chunk_data)
                    self.stats["chunks_created"] += 1
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(processed_files)} files")
        return all_chunks
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks, trying to preserve sentence boundaries"""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        
        # Try to split by sentences first
        sentences = text.replace('\n', ' ').split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + sentence + ". "
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Handle cases where individual sentences are too long
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # Force split long chunks
                words = chunk.split()
                current_word_chunk = ""
                for word in words:
                    if len(current_word_chunk + " " + word) <= chunk_size:
                        current_word_chunk += " " + word if current_word_chunk else word
                    else:
                        if current_word_chunk:
                            final_chunks.append(current_word_chunk)
                        current_word_chunk = word
                if current_word_chunk:
                    final_chunks.append(current_word_chunk)
        
        return final_chunks
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()


if __name__ == "__main__":
    # Test the processor
    print("🚀 Testing IntelliBase Data Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = IntelliBaseDataProcessor()
    
    # Process sample data
    print("\n📊 Processing sample data...")
    processed_files = processor.process_directory("../sample_data")
    
    if processed_files:
        print(f"\n📋 Processed {len(processed_files)} files:")
        for file_data in processed_files:
            print(f"  - {file_data['filename']} ({file_data['content_type']}) - {len(file_data['extracted_content'])} chars")
        
        # Create chunks for vector database
        print("\n✂️ Creating chunks for vector database...")
        chunks = processor.create_chunks(processed_files, chunk_size=800)
        
        print(f"\n📦 Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            print(f"\nChunk {i+1} ({chunk['source_file']}):")
            print(f"  Content: {chunk['content'][:100]}...")
            print(f"  Type: {chunk['content_type']}")
            print(f"  Length: {len(chunk['content'])} chars")
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    # Show statistics
    stats = processor.get_processing_stats()
    print(f"\n📊 Processing Statistics:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Errors: {stats['errors']}")
    
    print("\n✅ Data processing test complete!") 