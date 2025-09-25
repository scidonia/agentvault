"""Data processing module for downloading and processing Gutenberg texts."""

import requests
import re
from pathlib import Path
from typing import List, Dict
import json
from urllib.parse import urlparse

from bookwyrm import BookWyrm
from .config import GUTENBERG_URLS, RAW_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class GutenbergProcessor:
    """Process Gutenberg texts for RAG application."""
    
    def __init__(self):
        self.bookwyrm = BookWyrm()
        
    def download_text(self, url: str) -> str:
        """Download text from Gutenberg URL."""
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract title and author from Gutenberg text."""
        lines = text.split('\n')[:50]  # Check first 50 lines
        
        title = "Unknown Title"
        author = "Unknown Author"
        
        for line in lines:
            line = line.strip()
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Author:"):
                author = line.replace("Author:", "").strip()
        
        return {"title": title, "author": author}
    
    def clean_text(self, text: str) -> str:
        """Clean Gutenberg text by removing headers and footers."""
        lines = text.split('\n')
        
        # Find start of actual content (after "*** START OF")
        start_idx = 0
        for i, line in enumerate(lines):
            if "*** START OF" in line.upper():
                start_idx = i + 1
                break
        
        # Find end of actual content (before "*** END OF")
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if "*** END OF" in line.upper():
                end_idx = i
                break
        
        # Join the content lines
        content = '\n'.join(lines[start_idx:end_idx])
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        return content.strip()
    
    def process_text(self, text: str, metadata: Dict[str, str]) -> Dict:
        """Process text using BookWyrm phrasal model."""
        print(f"Processing {metadata['title']}...")
        
        # Clean the text
        clean_content = self.clean_text(text)
        
        # Process with BookWyrm
        processed = self.bookwyrm.process_text(
            clean_content,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Add metadata to each chunk
        for chunk in processed['chunks']:
            chunk.update(metadata)
        
        return processed
    
    def save_processed_data(self, processed_data: Dict, filename: str):
        """Save processed data to JSON file."""
        output_path = PROCESSED_DIR / f"{filename}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"Saved processed data to {output_path}")
    
    def process_all_texts(self):
        """Download and process all Gutenberg texts."""
        all_processed = []
        
        for url in GUTENBERG_URLS:
            # Extract filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).stem
            
            # Download text
            raw_text = self.download_text(url)
            
            # Save raw text
            raw_path = RAW_DIR / f"{filename}.txt"
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            
            # Extract metadata
            metadata = self.extract_metadata(raw_text)
            print(f"Found: {metadata['title']} by {metadata['author']}")
            
            # Process text
            processed = self.process_text(raw_text, metadata)
            
            # Save processed data
            self.save_processed_data(processed, filename)
            
            all_processed.append({
                'filename': filename,
                'metadata': metadata,
                'processed': processed
            })
        
        # Create summary
        self.create_summary(all_processed)
        
        return all_processed
    
    def create_summary(self, processed_texts: List[Dict]):
        """Create a summary of all processed texts."""
        summary = {
            "total_texts": len(processed_texts),
            "texts": []
        }
        
        for item in processed_texts:
            metadata = item['metadata']
            processed = item['processed']
            
            text_summary = {
                "filename": item['filename'],
                "title": metadata['title'],
                "author": metadata['author'],
                "total_chunks": len(processed['chunks']),
                "total_characters": sum(len(chunk['text']) for chunk in processed['chunks']),
                "embedding_dimension": len(processed['chunks'][0]['embedding']) if processed['chunks'] else 0
            }
            
            summary["texts"].append(text_summary)
        
        # Save summary
        summary_path = PROCESSED_DIR / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary saved to {summary_path}")
        print(f"Processed {summary['total_texts']} texts with {sum(t['total_chunks'] for t in summary['texts'])} total chunks")


if __name__ == "__main__":
    processor = GutenbergProcessor()
    processor.process_all_texts()
