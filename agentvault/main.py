"""Main CLI interface for the BookWyrm RAG Agent."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
import fnmatch
import re

from .google_drive_processor import GoogleDriveProcessor
from .rag_agent import RAGAgent
from .config import DATA_DIR, GOOGLE_DRIVE_INDEX_FILE

app = typer.Typer(
    name="agentvault",
    help="BookWyrm RAG Agent - Ask questions about literary texts",
    rich_markup_mode="rich",
)
console = Console()





@app.command("index-drive")
def index_drive(
    output_file: str = typer.Option(
        GOOGLE_DRIVE_INDEX_FILE, "--output", "-o", help="Output parquet file name"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reindexing even if index exists"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed progress information"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show debug information"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Process only N files"
    ),
    skip: int = typer.Option(
        0, "--skip", "-s", help="Skip first M files (start from Mth file)"
    ),
):
    """Index Google Drive files and create searchable database with detailed progress tracking."""
    
    output_path = DATA_DIR / output_file

    # Check if index already exists
    if not force and output_path.exists():
        if not typer.confirm(
            f"Index file {output_file} already exists. Reindex anyway?"
        ):
            console.print("❌ Indexing cancelled", style="yellow")
            raise typer.Exit()

    console.print(Panel.fit("🔍 Starting Google Drive Indexing", style="bold blue"))
    
    # Enhanced progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("[bold blue]{task.fields[files]} files"),
        TextColumn("[bold green]{task.fields[folders]} folders"),
        TextColumn("[bold yellow]{task.fields[docs]} docs"),
        console=console,
        transient=False,
    ) as progress:

        # Authentication task
        auth_task = progress.add_task(
            "🔐 Authenticating with Google Drive...",
            total=100,
            files=0,
            folders=0,
            docs=0,
        )

        try:
            processor = GoogleDriveProcessor()

            # Enhanced authentication with detailed feedback
            def auth_progress_callback(message: str):
                progress.update(auth_task, description=message)
                if verbose:
                    console.print(f"  {message}", style="dim")

            if not processor.authenticate(progress_callback=auth_progress_callback, debug=debug):
                console.print(
                    "❌ Failed to authenticate with Google Drive", style="red"
                )
                console.print("\n📋 [bold]Authentication Setup Instructions:[/bold]", style="yellow")
                console.print("1. Go to https://console.cloud.google.com/", style="yellow")
                console.print("2. Create a new project or select existing one", style="yellow")
                console.print("3. Enable the Google Drive API", style="yellow")
                console.print("4. Create credentials (OAuth 2.0 Client ID)", style="yellow")
                console.print("5. Download the JSON file and place it in the secret/ directory", style="yellow")
                raise typer.Exit(1)

            progress.update(
                auth_task, advance=100, description="✅ Authenticated successfully"
            )

            # Indexing task
            index_task = progress.add_task(
                "📁 Scanning Drive structure...", total=None, files=0, folders=0, docs=0
            )

            # Progress callback function
            def update_progress(stats):
                phase = stats.get("phase", "scanning")
                current_file = stats.get("current_file", "")
                current_path = stats.get("current_path", "")

                if phase == "saving":
                    description = "💾 Saving index to Parquet file..."
                else:
                    if verbose and current_file:
                        description = f"📄 Processing: {current_file}"
                    else:
                        description = (
                            f"📁 Scanning files... ({stats['total_files']} processed)"
                        )

                progress.update(
                    index_task,
                    description=description,
                    files=stats["total_files"],
                    folders=stats["folders"],
                    docs=stats["documents"],
                )
                
                # Show additional stats in verbose mode
                if verbose and "processed" in stats:
                    console.print(f"  📊 Processed: {stats['processed']}, Skipped: {stats.get('skipped', 0)}, Duplicates: {stats.get('duplicates', 0)}", style="dim")

                # Show current path in verbose mode
                if verbose and current_path and phase != "saving":
                    console.print(f"  📂 {current_path}", style="dim")

            success = processor.process_drive(
                output_file, progress_callback=update_progress, limit=limit, skip=skip
            )

            if not success:
                console.print("❌ Failed to index Google Drive", style="red")
                raise typer.Exit(1)

            progress.update(
                index_task, description="✅ Indexing completed successfully"
            )

        except KeyboardInterrupt:
            console.print("\n⚠️  Indexing interrupted by user", style="yellow")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"❌ Error during indexing: {e}", style="red")
            raise typer.Exit(1)

    console.print("✅ Google Drive indexing completed!", style="green bold")

    # Show detailed summary
    summary = processor.create_summary(output_file)
    if "error" not in summary:
        # Create a beautiful summary table
        summary_table = Table(
            title="📊 Indexing Results", show_header=True, header_style="bold magenta"
        )
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")

        summary_table.add_row("📁 Total Folders", f"{summary['total_folders']:,}")
        summary_table.add_row("📄 Total Documents", f"{summary['total_documents']:,}")
        summary_table.add_row("💾 Total Size", f"{summary['total_size_bytes']:,} bytes")
        summary_table.add_row("📂 Categories Found", f"{len(summary['categories'])}")
        summary_table.add_row("🗂️  File Types", f"{len(summary['mime_types'])}")

        console.print(summary_table)

        # Show top categories
        if summary["categories"]:
            console.print("\n🏷️  [bold]Top File Categories:[/bold]")
            for category, count in list(summary["categories"].items())[:5]:
                console.print(
                    f"  • [cyan]{category}[/cyan]: [green]{count}[/green] files"
                )

        console.print(
            f"\n💡 Use [bold]agentvault drive-summary[/bold] to see detailed statistics"
        )
        console.print(f"📁 Index saved to: [bold blue]{output_path}[/bold blue]")


@app.command("drive-summary")
def drive_summary(
    index_file: str = typer.Option(
        GOOGLE_DRIVE_INDEX_FILE, "--file", "-f", help="Index file to summarize"
    )
):
    """Show summary of Google Drive index."""

    console.print(f"🔍 Looking for index file: {index_file}", style="dim")
    index_path = DATA_DIR / index_file
    console.print(f"📁 Full path: {index_path}", style="dim")

    if not index_path.exists():
        console.print(
            f"❌ Index file {index_file} not found at {index_path}", style="red"
        )
        console.print("Please run Google Drive indexing first:", style="yellow")
        console.print("  [bold]agentvault index-drive[/bold]")

        # Show what files do exist in the data directory
        if DATA_DIR.exists():
            existing_files = list(DATA_DIR.glob("*"))
            if existing_files:
                console.print(f"\n📂 Files found in {DATA_DIR}:", style="blue")
                for file in existing_files:
                    console.print(f"  • {file.name}", style="dim")
            else:
                console.print(f"\n📂 {DATA_DIR} directory is empty", style="dim")
        else:
            console.print(f"\n📂 {DATA_DIR} directory doesn't exist yet", style="dim")

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

    console.print(
        Panel(summary_text, title="📊 Google Drive Index Summary", border_style="blue")
    )

    # Categories table
    if summary["categories"]:
        cat_table = Table(
            title="📂 File Categories", show_header=True, header_style="bold magenta"
        )
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right", style="green")

        for category, count in summary["categories"].items():
            cat_table.add_row(category, str(count))

        console.print(cat_table)

    # Largest files table
    if summary["largest_files"]:
        files_table = Table(
            title="📈 Largest Files", show_header=True, header_style="bold magenta"
        )
        files_table.add_column("Name", style="cyan", max_width=40)
        files_table.add_column("Size", justify="right", style="yellow")
        files_table.add_column("Category", style="green")

        for file_info in summary["largest_files"]:
            files_table.add_row(
                file_info["name"], f"{file_info['size']:,} bytes", file_info["category"]
            )

        console.print(files_table)



@app.callback()
def main():
    """
    🤖 BookWyrm RAG Agent

    A powerful tool for processing and analyzing documents using advanced AI.
    """
    pass


@app.command("test")
def test():
    """Test command to verify typer is working."""
    console.print("✅ Test command works!", style="green")


@app.command("list")
def list_files(
    pattern: Optional[str] = typer.Argument(None, help="Pattern to match file names/paths (supports wildcards)"),
    index_file: str = typer.Option(
        GOOGLE_DRIVE_INDEX_FILE, "--file", "-f", help="Index file to read from"
    ),
    # Column toggles
    name: bool = typer.Option(True, "--name/--no-name", "-n", help="Show file names"),
    path: bool = typer.Option(False, "--path/--no-path", "-p", help="Show full paths"),
    category: bool = typer.Option(True, "--category/--no-category", "-c", help="Show categories"),
    language: bool = typer.Option(False, "--language/--no-language", "-l", help="Show detected language"),
    size: bool = typer.Option(True, "--size/--no-size", "-s", help="Show file sizes"),
    mime: bool = typer.Option(False, "--mime/--no-mime", "-m", help="Show MIME types"),
    modified: bool = typer.Option(False, "--modified/--no-modified", "-t", help="Show modification times"),
    confidence: bool = typer.Option(False, "--confidence/--no-confidence", help="Show classification confidence"),
    # Filters
    filter_category: Optional[str] = typer.Option(None, "--filter-category", help="Filter by category"),
    filter_language: Optional[str] = typer.Option(None, "--filter-language", help="Filter by language"),
    filter_mime: Optional[str] = typer.Option(None, "--filter-mime", help="Filter by MIME type"),
    folders_only: bool = typer.Option(False, "--folders-only", help="Show only folders"),
    files_only: bool = typer.Option(False, "--files-only", help="Show only files (not folders)"),
    # Display options
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of results"),
    sort_by: str = typer.Option("name", "--sort", help="Sort by: name, size, modified, category"),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Reverse sort order"),
):
    """List files from the Google Drive index with filtering and customizable columns."""
    
    index_path = DATA_DIR / index_file
    
    if not index_path.exists():
        console.print(f"❌ Index file {index_file} not found at {index_path}", style="red")
        console.print("Please run Google Drive indexing first:", style="yellow")
        console.print("  [bold]agentvault index-drive[/bold]")
        raise typer.Exit(1)
    
    try:
        import pandas as pd
        df = pd.read_parquet(index_path)
    except Exception as e:
        console.print(f"❌ Error reading index file: {e}", style="red")
        raise typer.Exit(1)
    
    if df.empty:
        console.print("📂 No files found in index", style="yellow")
        return
    
    # Apply filters
    filtered_df = df.copy()
    
    # Pattern matching
    if pattern:
        if path:
            # Match against full path
            mask = filtered_df['path'].str.contains(pattern.replace('*', '.*').replace('?', '.'), 
                                                   case=False, regex=True, na=False)
        else:
            # Match against file name
            mask = filtered_df['name'].str.contains(pattern.replace('*', '.*').replace('?', '.'), 
                                                   case=False, regex=True, na=False)
        filtered_df = filtered_df[mask]
    
    # Category filter
    if filter_category:
        filtered_df = filtered_df[filtered_df['category'].str.contains(filter_category, case=False, na=False)]
    
    # Language filter
    if filter_language:
        filtered_df = filtered_df[filtered_df['language'].str.contains(filter_language, case=False, na=False)]
    
    # MIME type filter
    if filter_mime:
        filtered_df = filtered_df[filtered_df['mime_type'].str.contains(filter_mime, case=False, na=False)]
    
    # Folder/file filters
    if folders_only:
        filtered_df = filtered_df[filtered_df['is_folder'] == True]
    elif files_only:
        filtered_df = filtered_df[filtered_df['is_folder'] == False]
    
    if filtered_df.empty:
        console.print("📂 No files match the specified criteria", style="yellow")
        return
    
    # Sort the results
    sort_column_map = {
        'name': 'name',
        'size': 'size',
        'modified': 'modified_time',
        'category': 'category'
    }
    
    if sort_by in sort_column_map:
        sort_col = sort_column_map[sort_by]
        if sort_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(sort_col, ascending=not reverse)
    
    # Apply limit
    if limit:
        filtered_df = filtered_df.head(limit)
    
    # Create table with requested columns
    table = Table(title=f"📁 Files in Agent Vault ({len(filtered_df)} results)")
    
    # Add columns based on flags
    if name:
        table.add_column("Name", style="cyan", no_wrap=False, max_width=40)
    if path:
        table.add_column("Path", style="blue", no_wrap=False, max_width=50)
    if category:
        table.add_column("Category", style="green", max_width=20)
    if language:
        table.add_column("Language", style="yellow", max_width=10)
    if size:
        table.add_column("Size", justify="right", style="magenta")
    if mime:
        table.add_column("MIME Type", style="dim", max_width=25)
    if modified:
        table.add_column("Modified", style="dim", max_width=20)
    if confidence:
        table.add_column("Confidence", justify="right", style="bright_blue")
    
    # Add rows
    for _, row in filtered_df.iterrows():
        row_data = []
        
        if name:
            file_name = row.get('name', 'Unknown')
            if row.get('is_folder', False):
                file_name = f"📁 {file_name}"
            else:
                file_name = f"📄 {file_name}"
            row_data.append(file_name)
        
        if path:
            row_data.append(row.get('path', ''))
        
        if category:
            cat = row.get('category', 'Unknown')
            subcat = row.get('subcategory', '')
            if subcat and subcat != 'Unknown':
                row_data.append(f"{cat}/{subcat}")
            else:
                row_data.append(cat)
        
        if language:
            lang = row.get('language', 'unknown')
            row_data.append(lang if lang != 'unknown' else '-')
        
        if size:
            file_size = row.get('size', 0)
            if file_size > 0:
                if file_size >= 1024*1024:
                    row_data.append(f"{file_size/(1024*1024):.1f} MB")
                elif file_size >= 1024:
                    row_data.append(f"{file_size/1024:.1f} KB")
                else:
                    row_data.append(f"{file_size} B")
            else:
                row_data.append("-")
        
        if mime:
            row_data.append(row.get('mime_type', 'unknown'))
        
        if modified:
            mod_time = row.get('modified_time', '')
            if mod_time:
                # Format the timestamp nicely
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(mod_time.replace('Z', '+00:00'))
                    row_data.append(dt.strftime('%Y-%m-%d %H:%M'))
                except:
                    row_data.append(mod_time[:16])  # Fallback to first 16 chars
            else:
                row_data.append('-')
        
        if confidence:
            conf = row.get('classification_confidence', 0.0)
            if conf > 0:
                row_data.append(f"{conf:.2f}")
            else:
                row_data.append("-")
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Show summary
    total_files = len(filtered_df[filtered_df['is_folder'] == False])
    total_folders = len(filtered_df[filtered_df['is_folder'] == True])
    total_size = filtered_df['size'].sum()
    
    summary_text = f"📊 Summary: {total_files} files, {total_folders} folders"
    if total_size > 0:
        if total_size >= 1024*1024*1024:
            summary_text += f", {total_size/(1024*1024*1024):.1f} GB total"
        elif total_size >= 1024*1024:
            summary_text += f", {total_size/(1024*1024):.1f} MB total"
        else:
            summary_text += f", {total_size/1024:.1f} KB total"
    
    console.print(f"\n{summary_text}", style="dim")


@app.command("extract-pdfs")
def extract_pdfs(
    index_file: str = typer.Option(
        GOOGLE_DRIVE_INDEX_FILE, "--index", "-i", help="Google Drive index file to read from"
    ),
    output_file: str = typer.Option(
        "pdf_extractions.parquet", "--output", "-o", help="Output file for PDF extractions"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of PDFs to process"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-extraction even if extractions exist"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed progress information"
    ),
):
    """Extract full text from PDF files using BookWyrm API."""
    
    index_path = DATA_DIR / index_file
    output_path = DATA_DIR / output_file
    
    if not index_path.exists():
        console.print(f"❌ Index file {index_file} not found at {index_path}", style="red")
        console.print("Please run Google Drive indexing first:", style="yellow")
        console.print("  [bold]agentvault index-drive[/bold]")
        raise typer.Exit(1)
    
    # Check if output already exists
    if not force and output_path.exists():
        if not typer.confirm(f"Extractions file {output_file} already exists. Continue anyway?"):
            console.print("❌ Extraction cancelled", style="yellow")
            raise typer.Exit()
    
    # Check prerequisites before starting
    processor = GoogleDriveProcessor()
    
    if not processor.bookwyrm_client:
        console.print("❌ BookWyrm API key not found", style="red")
        console.print("\n📋 [bold]Setup Instructions:[/bold]", style="yellow")
        console.print("1. Get an API key from https://api.bookwyrm.ai", style="yellow")
        console.print("2. Set environment variable:", style="yellow")
        console.print("   [cyan]export BOOKWYRM_API_KEY='your-api-key-here'[/cyan]", style="yellow")
        console.print("3. Run the command again", style="yellow")
        raise typer.Exit(1)
    
    # Import the HAS_PDF_SUPPORT flag
    from .google_drive_processor import HAS_PDF_SUPPORT
    
    if not HAS_PDF_SUPPORT:
        console.print("❌ PDF extraction not supported in current BookWyrm client", style="red")
        console.print("\n📋 [bold]Upgrade Instructions:[/bold]", style="yellow")
        console.print("1. Upgrade BookWyrm client:", style="yellow")
        console.print("   [cyan]pip install --upgrade bookwyrm-client[/cyan]", style="yellow")
        console.print("2. Or install from source if needed", style="yellow")
        console.print("3. Run the command again", style="yellow")
        raise typer.Exit(1)
    
    console.print(Panel.fit("📄 Starting PDF Text Extraction", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold blue]{task.fields[processed]} processed"),
        TextColumn("[bold yellow]{task.fields[skipped]} skipped"),
        TextColumn("[bold red]{task.fields[errors]} errors"),
        console=console,
        transient=False,
    ) as progress:
        
        # Authentication task
        auth_task = progress.add_task(
            "🔐 Authenticating...", total=100, processed=0, skipped=0, errors=0
        )
        
        try:
            # processor already created above for prerequisite checks
            if not processor.service:
                if not processor.authenticate():
                    console.print("❌ Failed to authenticate with Google Drive", style="red")
                    raise typer.Exit(1)
            
            progress.update(auth_task, advance=100, description="✅ Authenticated")
            
            # Extraction task
            extract_task = progress.add_task(
                "📄 Extracting PDF text...", total=None, processed=0, skipped=0, errors=0
            )
            
            def update_progress(stats):
                phase = stats.get('phase', 'processing')
                current_file = stats.get('current_file', '')
                
                if phase == 'saving':
                    description = "💾 Saving extractions..."
                else:
                    if verbose and current_file:
                        description = f"📄 Processing: {current_file}"
                    else:
                        description = f"📄 Extracting text... ({stats['processed']} processed)"
                
                progress.update(
                    extract_task,
                    description=description,
                    total=stats.get('total', None),
                    completed=stats['processed'],
                    processed=stats['processed'],
                    skipped=stats['skipped'],
                    errors=stats['errors']
                )
                
                if verbose and current_file and phase != 'saving':
                    console.print(f"  📄 {current_file}", style="dim")
            
            success = processor.process_pdf_extractions(
                index_file=index_file,
                output_file=output_file,
                progress_callback=update_progress,
                limit=limit
            )
            
            if not success:
                console.print("❌ PDF extraction failed", style="red")
                raise typer.Exit(1)
            
            progress.update(extract_task, description="✅ Extraction completed")
            
        except KeyboardInterrupt:
            console.print("\n⚠️  Extraction interrupted by user", style="yellow")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"❌ Error during extraction: {e}", style="red")
            raise typer.Exit(1)
    
    console.print("✅ PDF text extraction completed!", style="green bold")
    console.print(f"📁 Extractions saved to: [bold blue]{output_path}[/bold blue]")


@app.command("process-phrases")
def process_phrases(
    index_file: str = typer.Option(
        GOOGLE_DRIVE_INDEX_FILE, "--index", "-i", help="Google Drive index file to read from"
    ),
    pdf_extractions_file: str = typer.Option(
        "pdf_extractions.parquet", "--pdf-extractions", help="PDF extractions file to use"
    ),
    output_file: str = typer.Option(
        "content_phrases.parquet", "--output", "-o", help="Output file for phrases"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of files to process"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if phrases exist"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed progress information"
    ),
):
    """Process all text content into phrases using BookWyrm phrasal API."""
    
    index_path = DATA_DIR / index_file
    output_path = DATA_DIR / output_file
    
    if not index_path.exists():
        console.print(f"❌ Index file {index_file} not found at {index_path}", style="red")
        console.print("Please run Google Drive indexing first:", style="yellow")
        console.print("  [bold]agentvault index-drive[/bold]")
        raise typer.Exit(1)
    
    # Check if output already exists
    if not force and output_path.exists():
        if not typer.confirm(f"Phrases file {output_file} already exists. Continue anyway?"):
            console.print("❌ Processing cancelled", style="yellow")
            raise typer.Exit()
    
    # Check prerequisites before starting
    processor = GoogleDriveProcessor()
    
    if not processor.bookwyrm_client:
        console.print("❌ BookWyrm API key not found", style="red")
        console.print("\n📋 [bold]Setup Instructions:[/bold]", style="yellow")
        console.print("1. Get an API key from https://api.bookwyrm.ai", style="yellow")
        console.print("2. Set environment variable:", style="yellow")
        console.print("   [cyan]export BOOKWYRM_API_KEY='your-api-key-here'[/cyan]", style="yellow")
        console.print("3. Run the command again", style="yellow")
        raise typer.Exit(1)
    
    # Import the HAS_PDF_SUPPORT flag
    from .google_drive_processor import HAS_PDF_SUPPORT
    
    if not HAS_PDF_SUPPORT:
        console.print("❌ Phrasal processing not supported in current BookWyrm client", style="red")
        console.print("\n📋 [bold]Upgrade Instructions:[/bold]", style="yellow")
        console.print("1. Upgrade BookWyrm client:", style="yellow")
        console.print("   [cyan]pip install --upgrade bookwyrm-client[/cyan]", style="yellow")
        console.print("2. Or install from source if needed", style="yellow")
        console.print("3. Run the command again", style="yellow")
        raise typer.Exit(1)
    
    console.print(Panel.fit("🔤 Starting Phrasal Processing", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold blue]{task.fields[processed]} processed"),
        TextColumn("[bold yellow]{task.fields[skipped]} skipped"),
        TextColumn("[bold red]{task.fields[errors]} errors"),
        console=console,
        transient=False,
    ) as progress:
        
        try:
            # processor already created above for prerequisite checks
            
            # Processing task
            process_task = progress.add_task(
                "🔤 Processing phrases...", total=None, processed=0, skipped=0, errors=0
            )
            
            def update_progress(stats):
                phase = stats.get('phase', 'processing')
                current_file = stats.get('current_file', '')
                
                if phase == 'saving':
                    description = "💾 Saving phrases..."
                else:
                    if verbose and current_file:
                        description = f"🔤 Processing: {current_file}"
                    else:
                        description = f"🔤 Creating phrases... ({stats['processed']} processed)"
                
                progress.update(
                    process_task,
                    description=description,
                    total=stats.get('total', None),
                    completed=stats['processed'],
                    processed=stats['processed'],
                    skipped=stats['skipped'],
                    errors=stats['errors']
                )
                
                if verbose and current_file and phase != 'saving':
                    console.print(f"  🔤 {current_file}", style="dim")
            
            success = processor.process_phrases_from_all_content(
                index_file=index_file,
                pdf_extractions_file=pdf_extractions_file,
                output_file=output_file,
                progress_callback=update_progress,
                limit=limit
            )
            
            if not success:
                console.print("❌ Phrasal processing failed", style="red")
                raise typer.Exit(1)
            
            progress.update(process_task, description="✅ Processing completed")
            
        except KeyboardInterrupt:
            console.print("\n⚠️  Processing interrupted by user", style="yellow")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"❌ Error during processing: {e}", style="red")
            raise typer.Exit(1)
    
    console.print("✅ Phrasal processing completed!", style="green bold")
    console.print(f"📁 Phrases saved to: [bold blue]{output_path}[/bold blue]")


@app.command("debug-phrases")
def debug_phrases(
    phrases_file: str = typer.Option(
        "content_phrases.parquet", "--phrases", "-p", help="Phrases file to debug"
    ),
    limit: int = typer.Option(
        3, "--limit", "-n", help="Number of files to show"
    ),
):
    """Debug phrases file to see what JSONL content would be created."""
    
    phrases_path = DATA_DIR / phrases_file
    
    if not phrases_path.exists():
        console.print(f"❌ Phrases file {phrases_file} not found at {phrases_path}", style="red")
        raise typer.Exit(1)
    
    try:
        import pandas as pd
        df_phrases = pd.read_parquet(phrases_path)
        
        console.print(f"📊 Loaded {len(df_phrases)} phrase records", style="blue")
        console.print(f"📊 Columns: {list(df_phrases.columns)}", style="blue")
        console.print(f"📊 Unique files: {df_phrases['file_hash'].nunique()}", style="blue")
        
        # Show sample data
        console.print("\n📋 Sample phrase records:", style="bold")
        sample_df = df_phrases.head(5)
        for _, row in sample_df.iterrows():
            console.print(f"  File: {row.get('file_name', 'Unknown')}")
            console.print(f"  Phrase: {row.get('phrase', 'No phrase')[:100]}...")
            console.print(f"  Start: {row.get('start_char', 'None')}, End: {row.get('end_char', 'None')}")
            console.print("")
        
        # Group by file and show JSONL that would be created
        console.print("\n📝 JSONL content that would be created:", style="bold")
        
        count = 0
        for file_hash, group in df_phrases.groupby('file_hash'):
            if count >= limit:
                break
                
            sorted_phrases = group.sort_values('phrase_count')
            file_name = sorted_phrases.iloc[0]['file_name']
            
            console.print(f"\n🔍 File: {file_name} ({len(sorted_phrases)} phrases)")
            
            # Create JSONL content
            jsonl_lines = []
            for _, phrase_row in sorted_phrases.iterrows():
                jsonl_line = {
                    "text": phrase_row['phrase'],
                    "start_char": phrase_row.get('start_char'),
                    "end_char": phrase_row.get('end_char')
                }
                jsonl_lines.append(json.dumps(jsonl_line))
            
            jsonl_content = '\n'.join(jsonl_lines)
            
            console.print(f"📏 JSONL length: {len(jsonl_content)} chars")
            console.print(f"📄 First 3 lines:")
            for i, line in enumerate(jsonl_lines[:3]):
                console.print(f"  {i+1}: {line[:120]}...")
            
            if len(jsonl_lines) > 3:
                console.print(f"  ... and {len(jsonl_lines) - 3} more lines")
            
            count += 1
            
    except Exception as e:
        console.print(f"❌ Error reading phrases file: {e}", style="red")
        raise typer.Exit(1)


@app.command("create-summaries")
def create_summaries(
    phrases_file: str = typer.Option(
        "content_phrases.parquet", "--phrases", "-p", help="Phrases file to summarize from"
    ),
    output_file: str = typer.Option(
        "content_summaries.parquet", "--output", "-o", help="Output file for summaries"
    ),
    max_tokens: int = typer.Option(
        10000, "--max-tokens", help="Maximum tokens per summary chunk (max: 131,072)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of files to process"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-processing even if summaries exist"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed progress information"
    ),
):
    """Create summaries from phrasal content using BookWyrm summarization API."""
    
    phrases_path = DATA_DIR / phrases_file
    output_path = DATA_DIR / output_file
    
    if not phrases_path.exists():
        console.print(f"❌ Phrases file {phrases_file} not found at {phrases_path}", style="red")
        console.print("Please run phrasal processing first:", style="yellow")
        console.print("  [bold]agentvault process-phrases[/bold]")
        raise typer.Exit(1)
    
    # Validate max_tokens
    if max_tokens > 131072:
        console.print(f"❌ max_tokens cannot exceed 131,072 (got {max_tokens})", style="red")
        raise typer.Exit(1)
    if max_tokens < 1:
        console.print(f"❌ max_tokens must be at least 1 (got {max_tokens})", style="red")
        raise typer.Exit(1)
    
    # Check if output already exists
    if not force and output_path.exists():
        if not typer.confirm(f"Summaries file {output_file} already exists. Continue anyway?"):
            console.print("❌ Processing cancelled", style="yellow")
            raise typer.Exit()
    
    # Check prerequisites before starting
    processor = GoogleDriveProcessor()
    
    if not processor.bookwyrm_client:
        console.print("❌ BookWyrm API key not found", style="red")
        console.print("\n📋 [bold]Setup Instructions:[/bold]", style="yellow")
        console.print("1. Get an API key from https://api.bookwyrm.ai", style="yellow")
        console.print("2. Set environment variable:", style="yellow")
        console.print("   [cyan]export BOOKWYRM_API_KEY='your-api-key-here'[/cyan]", style="yellow")
        console.print("3. Run the command again", style="yellow")
        raise typer.Exit(1)
    
    # Import the HAS_PDF_SUPPORT flag
    from .google_drive_processor import HAS_PDF_SUPPORT
    
    if not HAS_PDF_SUPPORT:
        console.print("❌ Summarization not supported in current BookWyrm client", style="red")
        console.print("\n📋 [bold]Upgrade Instructions:[/bold]", style="yellow")
        console.print("1. Upgrade BookWyrm client:", style="yellow")
        console.print("   [cyan]pip install --upgrade bookwyrm-client[/cyan]", style="yellow")
        console.print("2. Or install from source if needed", style="yellow")
        console.print("3. Run the command again", style="yellow")
        raise typer.Exit(1)
    
    console.print(Panel.fit("📝 Starting Content Summarization", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold blue]{task.fields[processed]} processed"),
        TextColumn("[bold yellow]{task.fields[skipped]} skipped"),
        TextColumn("[bold red]{task.fields[errors]} errors"),
        console=console,
        transient=False,
    ) as progress:
        
        try:
            # Processing task
            process_task = progress.add_task(
                "📝 Creating summaries...", total=None, processed=0, skipped=0, errors=0
            )
            
            def update_progress(stats):
                phase = stats.get('phase', 'processing')
                current_file = stats.get('current_file', '')
                
                if phase == 'saving':
                    description = "💾 Saving summaries..."
                else:
                    if verbose and current_file:
                        description = f"📝 Processing: {current_file}"
                    else:
                        description = f"📝 Creating summaries... ({stats['processed']} processed)"
                
                progress.update(
                    process_task,
                    description=description,
                    total=stats.get('total', None),
                    completed=stats['processed'],
                    processed=stats['processed'],
                    skipped=stats['skipped'],
                    errors=stats['errors']
                )
                
                if verbose and current_file and phase != 'saving':
                    console.print(f"  📝 {current_file}", style="dim")
            
            success = processor.process_summaries_from_phrases(
                phrases_file=phrases_file,
                output_file=output_file,
                progress_callback=update_progress,
                limit=limit,
                max_tokens=max_tokens
            )
            
            if not success:
                console.print("❌ Summarization failed", style="red")
                raise typer.Exit(1)
            
            progress.update(process_task, description="✅ Summarization completed")
            
        except KeyboardInterrupt:
            console.print("\n⚠️  Summarization interrupted by user", style="yellow")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"❌ Error during summarization: {e}", style="red")
            raise typer.Exit(1)
    
    console.print("✅ Content summarization completed!", style="green bold")
    console.print(f"📁 Summaries saved to: [bold blue]{output_path}[/bold blue]")


@app.command("version")
def version():
    """Show version information."""
    console.print("🤖 agentvault version 0.1.0", style="bold blue")
    console.print(
        "BookWyrm RAG Agent - Process and analyze documents with AI", style="dim"
    )


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    app()
