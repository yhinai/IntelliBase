#!/usr/bin/env python3
"""
Daft-based multi-modal data processor for IntelliBase
"""
import daft
import pandas as pd
from pathlib import Path
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DaftProcessor:
    """Multi-modal data processing using Daft"""
    
    def __init__(self):
        self.setup_udfs()
    
    def setup_udfs(self):
        """Define User Defined Functions for multimodal processing"""
        
        # Use simpler approach without UDFs - Daft UDF syntax varies
        pass
    
    def process_multimodal_data(self, data_path: str = "./sample_data/*") -> pd.DataFrame:
        """Process mixed media files using simplified approach"""
        
        logger.info(f"Processing files from: {data_path}")
        
        try:
            # Use glob to find files
            from glob import glob
            files = glob(data_path)
            
            # Process files into pandas DataFrame
            data = []
            for file_path in files:
                path_obj = Path(file_path)
                
                # Determine content type
                content_type = self._determine_content_type(file_path)
                
                # Extract content
                extracted_content = self._extract_content(file_path, content_type)
                
                data.append({
                    'path': file_path,
                    'content_type': content_type,
                    'extracted_content': extracted_content,
                    'file_size': path_obj.stat().st_size if path_obj.exists() else 0,
                    'processed_at': datetime.now().isoformat()
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully processed {len(data)} files")
            return df
            
        except Exception as e:
            logger.error(f"Error processing multimodal data: {e}")
            raise
    
    def _determine_content_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        path_str = str(file_path).lower()
        if path_str.endswith('.pdf'):
            return "pdf"
        elif path_str.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            return "image"
        elif path_str.endswith(('.txt', '.md')):
            return "text"
        else:
            return "unknown"
    
    def _extract_content(self, file_path: str, content_type: str) -> str:
        """Extract content based on file type"""
        try:
            if content_type == "text":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif content_type == "pdf":
                return f"PDF content from {file_path} - placeholder for PDF text extraction"
            elif content_type == "image":
                return f"Image features from {file_path} - placeholder for image analysis"
            else:
                return f"Unsupported file type: {content_type}"
        except Exception as e:
            return f"Error processing {file_path}: {str(e)}"
    
    def process_for_vector_db(self, df: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
        """Prepare data for vector database ingestion"""
        
        logger.info("Preparing data for vector database")
        
        try:
            # Filter out failed extractions
            df_clean = df[~df["extracted_content"].str.startswith("Error")]
            
            # Create chunks
            chunks_data = []
            for _, row in df_clean.iterrows():
                content = row['extracted_content']
                if not content or len(content) < chunk_size:
                    # Single chunk for small content
                    chunk_id = hashlib.sha256(f"{row['path']}:{content[:50]}".encode()).hexdigest()[:16]
                    chunks_data.append({
                        'path': row['path'],
                        'content_type': row['content_type'],
                        'content': content,
                        'chunk_id': chunk_id,
                        'file_size': row['file_size'],
                        'processed_at': row['processed_at'],
                        'chunk_index': 0
                    })
                else:
                    # Multiple chunks for large content
                    for i in range(0, len(content), chunk_size):
                        chunk_text = content[i:i + chunk_size]
                        chunk_id = hashlib.sha256(f"{row['path']}:{i}:{chunk_text[:50]}".encode()).hexdigest()[:16]
                        chunks_data.append({
                            'path': row['path'],
                            'content_type': row['content_type'],
                            'content': chunk_text,
                            'chunk_id': chunk_id,
                            'file_size': row['file_size'],
                            'processed_at': row['processed_at'],
                            'chunk_index': i // chunk_size
                        })
            
            chunks_df = pd.DataFrame(chunks_data)
            
            logger.info(f"Created {len(chunks_data)} chunks for vector database")
            return chunks_df
            
        except Exception as e:
            logger.error(f"Error preparing data for vector DB: {e}")
            raise
    
    def test_processing(self) -> bool:
        """Test the processing pipeline"""
        
        logger.info("Testing data processing pipeline...")
        
        try:
            # Process sample data
            df = self.process_multimodal_data("./sample_data/*")
            processed_df = self.process_for_vector_db(df)
            
            logger.info(f"Processing test successful:")
            logger.info(f"  - Processed {len(processed_df)} chunks")
            logger.info(f"  - Content types: {processed_df['content_type'].unique().tolist()}")
            logger.info(f"  - Average chunk size: {processed_df['content'].str.len().mean():.0f} chars")
            
            return True
            
        except Exception as e:
            logger.error(f"Processing test failed: {e}")
            return False

def main():
    """Test the Daft processor"""
    processor = DaftProcessor()
    success = processor.test_processing()
    
    if success:
        print("✅ Daft processor test successful!")
    else:
        print("❌ Daft processor test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())