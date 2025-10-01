"""Interactive query agent using LangGraph for title card search and citation-based answers."""

import logging
from typing import Dict, List, Any, Optional, TypedDict
from pathlib import Path

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    print(f"❌ Missing LangGraph dependencies: {e}")
    print("Please install: uv add langgraph langchain-core")
    raise

import pandas as pd
import numpy as np

from .config import DATA_DIR, LANCEDB_URI, TITLES_TABLE
from .openai_client import OpenAIClient
from .google_drive_processor import GoogleDriveProcessor

logger = logging.getLogger(__name__)


class QueryState(TypedDict):
    """State for the query processing graph."""
    user_question: str
    search_results: List[Dict[str, Any]]
    relevant_phrases: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    final_answer: str
    search_threshold: float
    error: Optional[str]


class TitleCardQueryAgent:
    """Agent for querying title cards and generating citation-based answers."""

    def __init__(self):
        self.processor = GoogleDriveProcessor()
        self.openai_client = self.processor.openai_client
        self.lancedb_client = self.processor.lancedb_client
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(QueryState)

        # Add nodes
        workflow.add_node("search_titles", self.search_title_cards_node)
        workflow.add_node("gather_phrases", self.gather_phrases_node)
        workflow.add_node("generate_answer", self.generate_answer_node)

        # Define the flow
        workflow.set_entry_point("search_titles")
        workflow.add_edge("search_titles", "gather_phrases")
        workflow.add_edge("gather_phrases", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def search_title_cards_node(self, state: QueryState) -> QueryState:
        """Node 1: Search for relevant title cards using vector similarity."""
        import sys
        import os
        
        def update_status(message):
            # Get terminal width, default to 80 if not available
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            # Create the full message
            full_message = f"🔍 [Stage 1/3] {message}"
            
            # Pad with spaces to clear any remaining characters
            padded_message = full_message.ljust(terminal_width)
            
            print(f"\r{padded_message}", end="", flush=True)
        
        update_status("Searching for relevant documents...")
        logger.info(f"Searching title cards for: {state['user_question']}")

        try:
            if not self.lancedb_client:
                state["error"] = "LanceDB client not available"
                return state

            if not self.openai_client:
                state["error"] = "OpenAI client not available for embeddings"
                return state

            update_status("Converting question to searchable format...")
            # Generate embedding for the user question
            question_embedding = self.openai_client.get_embedding(state["user_question"])
            
            update_status("Searching document database...")
            # Search LanceDB for similar title cards
            table = self.lancedb_client.open_table(TITLES_TABLE)
            
            update_status("Analyzing similarity scores...")
            # Perform vector search with top-k=15
            search_results = (
                table.search(question_embedding)
                .limit(15)
                .to_pandas()
            )

            if search_results.empty:
                # Get terminal width for proper clearing
                try:
                    terminal_width = os.get_terminal_size().columns
                except:
                    terminal_width = 80
            
                message = "🔍 [Stage 1/3] ⚠️  No relevant documents found"
                padded_message = message.ljust(terminal_width)
                print(f"\r{padded_message}")
                logger.warning("No title cards found in search")
                state["search_results"] = []
                state["search_threshold"] = 0.0
                return state

            update_status("Ranking documents by relevance...")
            # Convert to list of dictionaries and add similarity scores
            results = []
            similarities = []
            
            for _, row in search_results.iterrows():
                # Calculate cosine similarity (LanceDB returns distance, we want similarity)
                distance = row.get("_distance", 1.0)
                similarity = 1.0 - distance  # Convert distance to similarity
                similarities.append(similarity)
                
                result = {
                    "file_hash": row["file_hash"],
                    "title": row["title"],
                    "author": row["author"],
                    "file_name": row["file_name"],
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "summary_length": row["summary_length"],
                    "phrase_count": row["phrase_count"],
                    "path": row["path"],
                    "similarity_score": similarity,
                    "text": row["text"]  # The combined text used for embedding
                }
                results.append(result)

            # Calculate threshold (similarity of the 15th result)
            threshold = min(similarities) if similarities else 0.0

            state["search_results"] = results
            state["search_threshold"] = threshold

            # Get terminal width for proper clearing
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            message = f"🔍 [Stage 1/3] ✅ Found {len(results)} documents (threshold: {threshold:.4f})"
            padded_message = message.ljust(terminal_width)
            print(f"\r{padded_message}")
            logger.info(f"Found {len(results)} title cards, threshold: {threshold:.4f}")
            
            return state

        except Exception as e:
            # Get terminal width for proper clearing
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            message = f"🔍 [Stage 1/3] ❌ Search failed: {str(e)}"
            padded_message = message.ljust(terminal_width)
            print(f"\r{padded_message}")
            logger.error(f"Error in search_title_cards_node: {e}")
            state["error"] = f"Search error: {str(e)}"
            return state

    def gather_phrases_node(self, state: QueryState) -> QueryState:
        """Node 2: Gather phrasal information for the relevant title cards."""
        import sys
        import os
        
        def update_status(message):
            # Get terminal width, default to 80 if not available
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            # Create the full message
            full_message = f"📚 [Stage 2/3] {message}"
            
            # Pad with spaces to clear any remaining characters
            padded_message = full_message.ljust(terminal_width)
            
            print(f"\r{padded_message}", end="", flush=True)
        
        update_status("Gathering detailed content...")
        logger.info("Gathering phrases for relevant title cards")

        try:
            if state.get("error"):
                return state

            if not state["search_results"]:
                # Get terminal width for proper clearing
                try:
                    terminal_width = os.get_terminal_size().columns
                except:
                    terminal_width = 80
            
                message = "📚 [Stage 2/3] ⚠️  No search results to process"
                padded_message = message.ljust(terminal_width)
                print(f"\r{padded_message}")
                logger.warning("No search results to gather phrases for")
                state["relevant_phrases"] = []
                return state

            update_status("Looking for detailed phrase data...")
            # Load phrases data
            phrases_file = DATA_DIR / "content_phrases.parquet"
            if not phrases_file.exists():
                # Get terminal width for proper clearing
                try:
                    terminal_width = os.get_terminal_size().columns
                except:
                    terminal_width = 80
            
                message = "📚 [Stage 2/3] ⚠️  Using document summaries (no phrase data)"
                padded_message = message.ljust(terminal_width)
                print(f"\r{padded_message}")
                logger.warning("Phrases file not found, using summaries from title cards")
                state["relevant_phrases"] = []
                return state

            update_status("Loading phrase database...")
            df_phrases = pd.read_parquet(phrases_file)
            
            update_status("Filtering phrases for relevant documents...")
            # Get file hashes from search results
            relevant_file_hashes = [result["file_hash"] for result in state["search_results"]]
            
            # Filter phrases for relevant files
            relevant_phrases_df = df_phrases[
                df_phrases["file_hash"].isin(relevant_file_hashes)
            ].copy()

            if relevant_phrases_df.empty:
                # Get terminal width for proper clearing
                try:
                    terminal_width = os.get_terminal_size().columns
                except:
                    terminal_width = 80
            
                message = "📚 [Stage 2/3] ⚠️  No phrases found for relevant documents"
                padded_message = message.ljust(terminal_width)
                print(f"\r{padded_message}")
                logger.warning("No phrases found for relevant title cards")
                state["relevant_phrases"] = []
                return state

            update_status("Organizing phrases by document...")
            # Convert to list of dictionaries and group by file
            phrases_by_file = {}
            for _, row in relevant_phrases_df.iterrows():
                file_hash = row["file_hash"]
                if file_hash not in phrases_by_file:
                    phrases_by_file[file_hash] = []
                
                phrases_by_file[file_hash].append({
                    "phrase": row["phrase"],
                    "start_char": row.get("start_char", 0),
                    "end_char": row.get("end_char", 0),
                    "phrase_count": row.get("phrase_count", 0)
                })

            update_status("Selecting most relevant excerpts...")
            # Sort phrases within each file by phrase_count
            for file_hash in phrases_by_file:
                phrases_by_file[file_hash].sort(key=lambda x: x["phrase_count"])

            # Create structured phrases data
            relevant_phrases = []
            for result in state["search_results"]:
                file_hash = result["file_hash"]
                if file_hash in phrases_by_file:
                    relevant_phrases.append({
                        "file_hash": file_hash,
                        "title": result["title"],
                        "author": result["author"],
                        "similarity_score": result["similarity_score"],
                        "phrases": phrases_by_file[file_hash][:20]  # Limit to first 20 phrases per file
                    })

            state["relevant_phrases"] = relevant_phrases
            # Get terminal width for proper clearing
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            message = f"📚 [Stage 2/3] ✅ Gathered content from {len(relevant_phrases)} documents"
            padded_message = message.ljust(terminal_width)
            print(f"\r{padded_message}")
            logger.info(f"Gathered phrases from {len(relevant_phrases)} files")

            return state

        except Exception as e:
            # Get terminal width for proper clearing
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            message = f"📚 [Stage 2/3] ❌ Content gathering failed: {str(e)}"
            padded_message = message.ljust(terminal_width)
            print(f"\r{padded_message}")
            logger.error(f"Error in gather_phrases_node: {e}")
            state["error"] = f"Phrase gathering error: {str(e)}"
            return state

    def generate_answer_node(self, state: QueryState) -> QueryState:
        """Node 3: Generate citations and answer based on the gathered information."""
        import sys
        import os
        
        def update_status(message):
            # Get terminal width, default to 80 if not available
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            # Create the full message
            full_message = f"🤖 [Stage 3/3] {message}"
            
            # Pad with spaces to clear any remaining characters
            padded_message = full_message.ljust(terminal_width)
            
            print(f"\r{padded_message}", end="", flush=True)
        
        update_status("Generating comprehensive answer...")
        logger.info("Generating answer with citations")

        try:
            if state.get("error"):
                return state

            if not state["search_results"]:
                # Get terminal width for proper clearing
                try:
                    terminal_width = os.get_terminal_size().columns
                except:
                    terminal_width = 80
            
                message = "🤖 [Stage 3/3] ⚠️  No relevant documents to answer question"
                padded_message = message.ljust(terminal_width)
                print(f"\r{padded_message}")
                state["final_answer"] = "I couldn't find any relevant documents to answer your question."
                state["citations"] = []
                return state

            update_status("Preparing citations from top documents...")
            # Create citations from search results and phrases
            citations = []
            context_parts = []

            for i, result in enumerate(state["search_results"][:5], 1):  # Use top 5 for answer
                file_hash = result["file_hash"]
                
                update_status(f"Processing document {i}/5...")
                
                # Find corresponding phrases
                phrases_text = ""
                phrases_data = next(
                    (p for p in state["relevant_phrases"] if p["file_hash"] == file_hash),
                    None
                )
                
                if phrases_data and phrases_data["phrases"]:
                    # Use first few phrases as context
                    phrase_texts = [p["phrase"] for p in phrases_data["phrases"][:5]]
                    phrases_text = " ".join(phrase_texts)
                else:
                    # Fall back to the embedded text from title card
                    phrases_text = result["text"][:500]  # Limit length

                # Get the full summary from the title card for context
                title_card_summary = ""
                # The "text" field in search results contains the combined title+author+summary
                # We need to extract just the summary part, or use a reasonable portion
                full_text = result.get("text", "")
                if full_text:
                    # Try to extract summary portion (everything after title and author)
                    title = result.get("title", "")
                    author = result.get("author", "")
                    
                    # Remove title and author from the beginning to get summary
                    summary_start = full_text
                    if title:
                        summary_start = summary_start.replace(title, "", 1).strip()
                    if author:
                        summary_start = summary_start.replace(author, "", 1).strip()
                    
                    title_card_summary = summary_start[:800] if summary_start else full_text[:800]

                citation = {
                    "id": i,
                    "title": result["title"],
                    "author": result["author"] or "Unknown",
                    "file_name": result["file_name"],
                    "similarity_score": result["similarity_score"],
                    "excerpt": phrases_text[:300] + "..." if len(phrases_text) > 300 else phrases_text,
                    "summary": title_card_summary,
                    "category": result.get("category", "Unknown"),
                    "subcategory": result.get("subcategory", "Unknown")
                }
                citations.append(citation)

                # Add to context for answer generation - include both summary and phrases
                context_parts.append(
                    f"[{i}] {result['title']} by {result['author'] or 'Unknown'}\n"
                    f"Summary: {title_card_summary}\n"
                    f"Relevant excerpts: {phrases_text}"
                )

            state["citations"] = citations

            update_status("Analyzing document content and context...")
            # Generate answer using OpenAI
            if not self.openai_client:
                # Get terminal width for proper clearing
                try:
                    terminal_width = os.get_terminal_size().columns
                except:
                    terminal_width = 80
            
                message = "🤖 [Stage 3/3] ❌ AI assistant not available"
                padded_message = message.ljust(terminal_width)
                print(f"\r{padded_message}")
                state["final_answer"] = "OpenAI client not available for answer generation."
                return state

            update_status("Crafting comprehensive answer with citations...")
            context = "\n\n".join(context_parts)
            
            prompt = f"""Based on the following documents, please answer the user's question. Use the citation numbers [1], [2], etc. to reference specific documents in your answer.

User Question: {state['user_question']}

Available Documents:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Uses specific citations [1], [2], etc. when referencing information
3. Acknowledges if the available documents don't fully answer the question
4. Is clear and well-structured

Answer:"""

            messages = [
                {"role": "user", "content": prompt}
            ]

            update_status("Generating final response...")
            response = self.openai_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )

            # Handle different response types from OpenAI client
            if hasattr(response, 'choices'):
                # Standard OpenAI API response object
                state["final_answer"] = response.choices[0].message.content
            elif isinstance(response, str):
                # Direct string response
                state["final_answer"] = response
            else:
                # Fallback - try to extract content
                state["final_answer"] = str(response)

            # Get terminal width for proper clearing
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            message = "🤖 [Stage 3/3] ✅ Answer generated with citations"
            padded_message = message.ljust(terminal_width)
            print(f"\r{padded_message}")
            logger.info("Generated answer with citations")
            return state

        except Exception as e:
            # Get terminal width for proper clearing
            try:
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80
            
            message = f"🤖 [Stage 3/3] ❌ Answer generation failed: {str(e)}"
            padded_message = message.ljust(terminal_width)
            print(f"\r{padded_message}")
            logger.error(f"Error in generate_answer_node: {e}")
            state["error"] = f"Answer generation error: {str(e)}"
            state["final_answer"] = "I encountered an error while generating the answer."
            return state

    def query(self, question: str) -> Dict[str, Any]:
        """Process a user query through the graph."""
        logger.info(f"Processing query: {question}")

        # Initialize state
        initial_state = QueryState(
            user_question=question,
            search_results=[],
            relevant_phrases=[],
            citations=[],
            final_answer="",
            search_threshold=0.0,
            error=None
        )

        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)

            # Format response
            response = {
                "question": question,
                "answer": final_state["final_answer"],
                "citations": final_state["citations"],
                "search_results_count": len(final_state["search_results"]),
                "search_threshold": final_state["search_threshold"],
                "error": final_state.get("error")
            }

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": "I encountered an error while processing your question.",
                "citations": [],
                "search_results_count": 0,
                "search_threshold": 0.0,
                "error": str(e)
            }


def main():
    """Interactive CLI for the query agent."""
    import sys
    import readline
    import atexit
    from pathlib import Path
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    
    # Set up history file
    history_file = Path("data/query_history.txt")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing history
    if history_file.exists():
        try:
            readline.read_history_file(str(history_file))
            console.print(f"📚 Loaded {readline.get_current_history_length()} previous queries", style="dim")
        except Exception as e:
            console.print(f"⚠️  Could not load history: {e}", style="dim")
    
    # Set up history saving on exit
    def save_history():
        try:
            readline.write_history_file(str(history_file))
        except Exception as e:
            console.print(f"⚠️  Could not save history: {e}", style="dim")
    
    atexit.register(save_history)
    
    # Configure readline
    readline.set_history_length(1000)  # Keep last 1000 queries
    
    console.print(Panel.fit("🤖 Title Card Query Agent", style="bold blue"))
    console.print("Ask questions about your indexed documents. Type 'quit' to exit.")
    console.print("💡 Use ↑/↓ arrow keys to navigate through previous questions.\n", style="dim")

    # Initialize agent
    try:
        agent = TitleCardQueryAgent()
        
        # Check prerequisites
        if not agent.lancedb_client:
            console.print("❌ LanceDB client not available", style="red")
            console.print("Please run: agentvault index-titles", style="yellow")
            sys.exit(1)
            
        if not agent.openai_client:
            console.print("❌ OpenAI client not available", style="red")
            console.print("Please set OPENAI_API_KEY environment variable", style="yellow")
            sys.exit(1)
            
        console.print("✅ Agent initialized successfully", style="green")
        
    except Exception as e:
        console.print(f"❌ Failed to initialize agent: {e}", style="red")
        sys.exit(1)

    # Interactive loop
    while True:
        try:
            # Use input() instead of Prompt.ask() to get readline support
            question = input("\n🔍 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("👋 Goodbye!", style="blue")
                break
                
            if not question:
                continue

            print("\n🚀 Starting query processing...")
            
            # Process query
            response = agent.query(question)
            
            # Clear the status line and add some space
            print("\n")
            
            if response.get("error"):
                console.print(f"❌ Error: {response['error']}", style="red")
                continue

            # Display results
            console.print(f"\n📊 Found {response['search_results_count']} relevant documents", style="dim")
            console.print(f"🎯 Search threshold: {response['search_threshold']:.4f}", style="dim")
            
            # Show answer
            console.print(Panel(response["answer"], title="🤖 Answer", border_style="green"))
            
            # Show citations with summaries
            if response["citations"]:
                citations_table = Table(title="📚 Citations", show_header=True, header_style="bold magenta")
                citations_table.add_column("ID", style="cyan", width=4)
                citations_table.add_column("Title", style="green", width=25)
                citations_table.add_column("Author", style="yellow", width=15)
                citations_table.add_column("Category", style="blue", width=12)
                citations_table.add_column("Relevance", style="blue", width=8)
                citations_table.add_column("Summary", style="dim", width=50)
                
                for citation in response["citations"]:
                    citations_table.add_row(
                        f"[{citation['id']}]",
                        citation["title"][:25] + "..." if len(citation["title"]) > 25 else citation["title"],
                        citation["author"][:15] + "..." if len(citation["author"]) > 15 else citation["author"],
                        citation.get("category", "Unknown")[:12],
                        f"{citation['similarity_score']:.3f}",
                        citation.get("summary", "")[:50] + "..." if len(citation.get("summary", "")) > 50 else citation.get("summary", "")
                    )
                
                console.print(citations_table)

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="blue")
            break
        except EOFError:
            console.print("\n👋 Goodbye!", style="blue")
            break
        except Exception as e:
            console.print(f"❌ Unexpected error: {e}", style="red")


if __name__ == "__main__":
    main()
