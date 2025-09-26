# AgentVault

A powerful RAG (Retrieval-Augmented Generation) application that indexes and queries Google Drive files using BookWyrm AI for intelligent document processing and question answering.

## Features

- 🔍 **Google Drive Integration**: Recursively index all files in your Google Drive
- 📊 **Smart Classification**: Automatically categorize files using BookWyrm AI
- 🔄 **Incremental Indexing**: Hash-based deduplication for efficient updates
- 📄 **Multiple File Types**: Support for Google Docs, Sheets, PDFs, text files, and more
- 🎯 **Pagination Support**: Process files in batches with skip/limit options
- 💬 **Interactive Chat**: Ask questions about your indexed documents
- 📈 **Rich Progress Tracking**: Beautiful CLI with real-time progress bars
- 🗃️ **Parquet Storage**: Efficient columnar storage for fast queries

## Installation

### Prerequisites

- Python 3.8+
- Google Drive API credentials
- BookWyrm API access

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agentvault
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up Google Drive API credentials:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Drive API
   - Create OAuth 2.0 Client ID credentials
   - Download the JSON file and place it in the `secret/` directory

## Usage

### Google Drive Indexing

Index your entire Google Drive:
```bash
agentvault index-drive
```

Index with options:
```bash
# Process only first 100 files
agentvault index-drive --limit 100

# Skip first 50 files, process next 100
agentvault index-drive --skip 50 --limit 100

# Verbose output with debug information
agentvault index-drive --verbose --debug

# Force reindexing (overwrite existing index)
agentvault index-drive --force
```

### View Index Summary

```bash
# Show summary of indexed files
agentvault drive-summary

# Summary of specific index file
agentvault drive-summary --file my_index.parquet
```

### Interactive Chat

Ask questions about your indexed documents:
```bash
agentvault chat
```

### Single Query

Ask a single question:
```bash
agentvault query "What are the main topics in my documents?"

# Show detailed citations
agentvault query "What are the main topics?" --citations
```

### Process Gutenberg Texts

Process sample texts from Project Gutenberg:
```bash
agentvault process
```

### View Processing Summary

```bash
agentvault summary
```

## Configuration

The application uses several configuration files:

- `agentvault/config.py`: Main configuration settings
- `secret/`: Directory for Google Drive API credentials
- `data/`: Directory for processed data and indexes

### Key Settings

- **CHUNK_SIZE**: Size of text chunks for processing (default: 512)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 50)
- **GOOGLE_DRIVE_INDEX_FILE**: Default output file name
- **SUPPORTED_MIME_TYPES**: File types to process

## File Structure

```
agentvault/
├── agentvault/
│   ├── __init__.py
│   ├── main.py              # CLI interface
│   ├── config.py            # Configuration settings
│   ├── google_drive_processor.py  # Google Drive integration
│   ├── data_processor.py    # Text processing
│   └── rag_agent.py         # RAG agent implementation
├── data/
│   ├── raw/                 # Raw downloaded files
│   └── processed/           # Processed data
├── secret/                  # API credentials (not in git)
├── GoogleDriveIndexStrategy.md
└── README.md
```

## How It Works

### 1. Authentication
- Uses OAuth 2.0 to authenticate with Google Drive
- Stores credentials securely for future use
- Tests connection and displays connected user

### 2. File Discovery
- Recursively traverses Google Drive structure
- Extracts metadata (name, size, type, modified date)
- Calculates file hashes for deduplication

### 3. Content Processing
- Downloads supported file types
- Extracts text content from various formats
- Classifies files using BookWyrm AI
- Generates embeddings for semantic search

### 4. Storage
- Saves data in efficient Parquet format
- Supports incremental updates
- Hash-based deduplication prevents duplicates

### 5. Querying
- Uses RAG (Retrieval-Augmented Generation)
- Semantic search with embeddings
- Provides citations and sources

## Supported File Types

- **Google Workspace**: Docs, Sheets, Presentations
- **Text Files**: .txt, .csv, .json
- **Documents**: PDF, Word (planned)
- **Images**: Basic metadata extraction

## CLI Commands

| Command | Description |
|---------|-------------|
| `index-drive` | Index Google Drive files |
| `drive-summary` | Show indexing summary |
| `chat` | Interactive Q&A session |
| `query` | Single question query |
| `process` | Process Gutenberg texts |
| `summary` | Show processing summary |
| `test` | Test CLI functionality |
| `version` | Show version info |

## Advanced Usage

### Incremental Indexing

The system supports incremental indexing - you can run the same command multiple times and it will only process new or changed files:

```bash
# First run - indexes everything
agentvault index-drive

# Later runs - only new/changed files
agentvault index-drive
```

### Batch Processing

Process files in batches for large drives:

```bash
# Process files 0-999
agentvault index-drive --limit 1000

# Process files 1000-1999
agentvault index-drive --skip 1000 --limit 1000

# Process files 2000-2999
agentvault index-drive --skip 2000 --limit 1000
```

### Custom Output Files

Use different index files for different purposes:

```bash
# Create separate indexes
agentvault index-drive --output work_docs.parquet --limit 500
agentvault index-drive --output personal_docs.parquet --skip 500

# Query specific indexes
agentvault drive-summary --file work_docs.parquet
```

## Troubleshooting

### Authentication Issues

1. Ensure credentials file is in `secret/` directory
2. Check that Google Drive API is enabled
3. Verify OAuth consent screen is configured
4. Try deleting `secret/token.pickle` to force re-authentication

### Performance Issues

1. Use `--limit` to process files in smaller batches
2. Check network connectivity for large files
3. Monitor disk space for Parquet files
4. Use `--debug` flag to identify bottlenecks

### File Processing Errors

1. Check file permissions in Google Drive
2. Verify supported file types
3. Review logs for specific error messages
4. Use `--verbose` for detailed progress information

## Development

### Running Tests

```bash
agentvault test
```

### Adding New File Types

1. Update `SUPPORTED_MIME_TYPES` in `config.py`
2. Add processing logic in `download_file_content()`
3. Update classification rules in `classify_file()`

### Extending the RAG Agent

1. Modify `rag_agent.py` for custom query processing
2. Add new embedding models in `config.py`
3. Customize chunking strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs with `--debug` flag
- Open an issue on GitHub
