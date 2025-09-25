"""Main CLI interface for the BookWyrm RAG Agent."""

import argparse
import sys
from pathlib import Path

from .data_processor import GutenbergProcessor
from .rag_agent import RAGAgent
from .config import PROCESSED_DIR


def process_texts():
    """Download and process Gutenberg texts."""
    print("Starting text processing...")
    processor = GutenbergProcessor()
    processor.process_all_texts()
    print("Text processing completed!")


def run_agent():
    """Run the interactive RAG agent."""
    # Check if processed data exists
    if not any(PROCESSED_DIR.glob("*.json")) or not (PROCESSED_DIR / "summary.json").exists():
        print("No processed data found. Please run text processing first:")
        print("  python -m agentvault.main --process")
        return
    
    print("Initializing RAG Agent...")
    agent = RAGAgent()
    
    print("\nBookWyrm RAG Agent Ready!")
    print("Ask questions about the processed texts. Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nSearching and generating answer...\n")
            result = agent.query(question)
            
            print("Answer:")
            print("-" * 60)
            print(result['answer'])
            print("-" * 60)
            print(f"(Used {result['search_results']} search results)")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


def show_summary():
    """Show summary of processed texts."""
    summary_path = PROCESSED_DIR / "summary.json"
    
    if not summary_path.exists():
        print("No summary found. Please run text processing first:")
        print("  python -m agentvault.main --process")
        return
    
    import json
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print("Processed Texts Summary:")
    print("=" * 50)
    print(f"Total texts: {summary['total_texts']}")
    print(f"Total chunks: {sum(t['total_chunks'] for t in summary['texts'])}")
    print()
    
    for text in summary['texts']:
        print(f"📖 {text['title']}")
        print(f"   Author: {text['author']}")
        print(f"   Chunks: {text['total_chunks']}")
        print(f"   Characters: {text['total_characters']:,}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BookWyrm RAG Agent - Ask questions about literary texts"
    )
    
    parser.add_argument(
        '--process', 
        action='store_true',
        help='Download and process Gutenberg texts'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true', 
        help='Show summary of processed texts'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Ask a single question and exit'
    )
    
    args = parser.parse_args()
    
    if args.process:
        process_texts()
    elif args.summary:
        show_summary()
    elif args.query:
        # Check if processed data exists
        if not any(PROCESSED_DIR.glob("*.json")):
            print("No processed data found. Please run text processing first:")
            print("  python -m agentvault.main --process")
            return
        
        agent = RAGAgent()
        result = agent.query(args.query)
        print(result['answer'])
    else:
        # Interactive mode
        run_agent()


if __name__ == "__main__":
    main()
