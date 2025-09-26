"""Main CLI interface for the BookWyrm RAG Agent."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .data_processor import GutenbergProcessor
from .google_drive_processor import GoogleDriveProcessor
from .rag_agent import RAGAgent
from .config import PROCESSED_DIR, DATA_DIR, GOOGLE_DRIVE_INDEX_FILE

app = typer.Typer(
    name="agentvault",
    help="BookWyrm RAG Agent - Ask questions about literary texts",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def process(
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing even if data exists")
):
    """Download and process Gutenberg texts with progress monitoring."""
    
    # Check if data already exists
    if not force and any(PROCESSED_DIR.glob("*.json")):
        if not typer.confirm("Processed data already exists. Reprocess anyway?"):
            console.print("❌ Processing cancelled", style="yellow")
            raise typer.Exit()
    
    console.print(Panel.fit("🚀 Starting Text Processing", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress:
        
        # Create main task
        main_task = progress.add_task("Processing texts...", total=100)
        
        try:
            processor = GutenbergProcessor()
            
            # Update progress as we go
            progress.update(main_task, advance=20, description="Initializing processor...")
            
            # This would need to be modified in GutenbergProcessor to support progress callbacks
            processor.process_all_texts()
            
            progress.update(main_task, advance=80, description="Finalizing...")
            
        except Exception as e:
            console.print(f"❌ Error during processing: {e}", style="red")
            raise typer.Exit(1)
    
    console.print("✅ Text processing completed!", style="green bold")


@app.command()
def chat():
    """Run the interactive RAG agent chat interface."""
    
    # Check if processed data exists
    if not any(PROCESSED_DIR.glob("*.json")) or not (PROCESSED_DIR / "summary.json").exists():
        console.print("❌ No processed data found.", style="red")
        console.print("Please run text processing first:", style="yellow")
        console.print("  [bold]agentvault process[/bold]")
        raise typer.Exit(1)
    
    console.print(Panel.fit("🤖 Initializing RAG Agent", style="bold blue"))
    
    with console.status("[bold green]Loading knowledge base..."):
        try:
            agent = RAGAgent()
        except Exception as e:
            console.print(f"❌ Error initializing agent: {e}", style="red")
            raise typer.Exit(1)
    
    console.print("✅ BookWyrm RAG Agent Ready!", style="green bold")
    console.print("Ask questions about the processed texts. Type [bold red]'quit'[/bold red] to exit.\n")
    
    while True:
        try:
            question = Prompt.ask("🤔 [bold blue]Question[/bold blue]").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("👋 Goodbye!", style="bold blue")
                break
            
            if not question:
                continue
            
            # Show progress while processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("🔍 Searching and generating answer...", total=None)
                
                try:
                    result = agent.query(question)
                except Exception as e:
                    console.print(f"❌ Error processing question: {e}", style="red")
                    continue
            
            # Display answer in a nice panel
            answer_panel = Panel(
                result['answer'],
                title="💡 Answer",
                border_style="green",
                padding=(1, 2)
            )
            console.print(answer_panel)
            
            # Show metadata
            console.print(f"📊 Used {result['search_results']} search results", style="dim")
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold blue")
            break
        except Exception as e:
            console.print(f"❌ Unexpected error: {e}", style="red")
            console.print("Please try again.\n", style="yellow")


@app.command()
def summary():
    """Show summary of processed texts in a beautiful table."""
    summary_path = PROCESSED_DIR / "summary.json"
    
    if not summary_path.exists():
        console.print("❌ No summary found.", style="red")
        console.print("Please run text processing first:", style="yellow")
        console.print("  [bold]agentvault process[/bold]")
        raise typer.Exit(1)
    
    import json
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    # Create summary panel
    total_chunks = sum(t['total_chunks'] for t in summary_data['texts'])
    total_chars = sum(t['total_characters'] for t in summary_data['texts'])
    
    summary_text = f"""
📚 Total texts: [bold blue]{summary_data['total_texts']}[/bold blue]
🧩 Total chunks: [bold green]{total_chunks:,}[/bold green]
📝 Total characters: [bold yellow]{total_chars:,}[/bold yellow]
    """.strip()
    
    console.print(Panel(summary_text, title="📊 Processing Summary", border_style="blue"))
    
    # Create detailed table
    table = Table(title="📖 Processed Texts Details", show_header=True, header_style="bold magenta")
    table.add_column("Title", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Author", style="green")
    table.add_column("Chunks", justify="right", style="blue")
    table.add_column("Characters", justify="right", style="yellow")
    
    for text in summary_data['texts']:
        table.add_row(
            text['title'],
            text['author'],
            f"{text['total_chunks']:,}",
            f"{text['total_characters']:,}"
        )
    
    console.print(table)


@app.command()
def index_drive(
    output_file: str = typer.Option(GOOGLE_DRIVE_INDEX_FILE, "--output", "-o", help="Output parquet file name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reindexing even if index exists")
):
    """Index Google Drive files and create searchable database."""
    
    output_path = DATA_DIR / output_file
    
    # Check if index already exists
    if not force and output_path.exists():
        if not typer.confirm(f"Index file {output_file} already exists. Reindex anyway?"):
            console.print("❌ Indexing cancelled", style="yellow")
            raise typer.Exit()
    
    console.print(Panel.fit("🔍 Starting Google Drive Indexing", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        task = progress.add_task("Authenticating with Google Drive...", total=None)
        
        try:
            processor = GoogleDriveProcessor()
            
            progress.update(task, description="🔐 Authenticating...")
            if not processor.authenticate():
                console.print("❌ Failed to authenticate with Google Drive", style="red")
                console.print("Please ensure you have valid credentials in the secret/ directory", style="yellow")
                raise typer.Exit(1)
            
            progress.update(task, description="📁 Scanning Drive structure...")
            success = processor.process_drive(output_file)
            
            if not success:
                console.print("❌ Failed to index Google Drive", style="red")
                raise typer.Exit(1)
                
        except Exception as e:
            console.print(f"❌ Error during indexing: {e}", style="red")
            raise typer.Exit(1)
    
    console.print("✅ Google Drive indexing completed!", style="green bold")
    
    # Show summary
    summary = processor.create_summary(output_file)
    if "error" not in summary:
        console.print(f"📊 Indexed {summary['total_documents']} documents in {summary['total_folders']} folders")
        console.print(f"💾 Total size: {summary['total_size_bytes']:,} bytes")


@app.command()
def drive_summary(
    index_file: str = typer.Option(GOOGLE_DRIVE_INDEX_FILE, "--file", "-f", help="Index file to summarize")
):
    """Show summary of Google Drive index."""
    
    index_path = DATA_DIR / index_file
    if not index_path.exists():
        console.print(f"❌ Index file {index_file} not found.", style="red")
        console.print("Please run Google Drive indexing first:", style="yellow")
        console.print("  [bold]agentvault index-drive[/bold]")
        raise typer.Exit(1)
    
    processor = GoogleDriveProcessor()
    summary = processor.create_summary(index_file)
    
    if "error" in summary:
        console.print(f"❌ Error reading index: {summary['error']}", style="red")
        raise typer.Exit(1)
    
    # Create summary panel
    summary_text = f"""
📁 Total folders: [bold blue]{summary['total_folders']}[/bold blue]
📄 Total documents: [bold green]{summary['total_documents']}[/bold green]  
💾 Total size: [bold yellow]{summary['total_size_bytes']:,} bytes[/bold yellow]
    """.strip()
    
    console.print(Panel(summary_text, title="📊 Google Drive Index Summary", border_style="blue"))
    
    # Categories table
    if summary['categories']:
        cat_table = Table(title="📂 File Categories", show_header=True, header_style="bold magenta")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right", style="green")
        
        for category, count in summary['categories'].items():
            cat_table.add_row(category, str(count))
        
        console.print(cat_table)
    
    # Largest files table
    if summary['largest_files']:
        files_table = Table(title="📈 Largest Files", show_header=True, header_style="bold magenta")
        files_table.add_column("Name", style="cyan", max_width=40)
        files_table.add_column("Size", justify="right", style="yellow")
        files_table.add_column("Category", style="green")
        
        for file_info in summary['largest_files']:
            files_table.add_row(
                file_info['name'],
                f"{file_info['size']:,} bytes",
                file_info['category']
            )
        
        console.print(files_table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask about the texts"),
    show_citations: bool = typer.Option(False, "--citations", "-c", help="Show detailed citations")
):
    """Ask a single question and get an answer."""
    
    # Check if processed data exists
    if not any(PROCESSED_DIR.glob("*.json")):
        console.print("❌ No processed data found.", style="red")
        console.print("Please run text processing first:", style="yellow")
        console.print("  [bold]agentvault process[/bold]")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("🤖 Initializing agent...", total=None)
        
        try:
            agent = RAGAgent()
            progress.update(task, description="🔍 Processing question...")
            result = agent.query(question)
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")
            raise typer.Exit(1)
    
    # Display the answer
    answer_panel = Panel(
        result['answer'],
        title=f"💡 Answer to: '{question}'",
        border_style="green",
        padding=(1, 2)
    )
    console.print(answer_panel)
    
    if show_citations and result.get('citations'):
        console.print("\n📚 [bold]Citations:[/bold]")
        for i, citation in enumerate(result['citations'], 1):
            console.print(f"  {i}. [cyan]{citation['title']}[/cyan] by [green]{citation['author']}[/green]")
            console.print(f"     Relevance: {citation['similarity']:.2f}")


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", "-v", help="Show version and exit")
):
    """
    🤖 BookWyrm RAG Agent
    
    A powerful tool for asking questions about literary texts using advanced AI.
    """
    if version:
        console.print("agentvault version 0.1.0", style="bold blue")
        raise typer.Exit()
    
    # If no command is provided, default to chat
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


if __name__ == "__main__":
    app()
