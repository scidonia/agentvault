# Google Drive Index Strategy

## Overview

This document outlines the process for indexing Google Drive files and creating a RAG (Retrieval-Augmented Generation) application that can answer questions about the content.

## Process Flow

### 1. Authentication & Setup

- Load Google Drive API credentials from `secret/` directory (first JSON file found)
- Initialize Google Drive API client using OAuth2 flow
- Set up required scopes: `https://www.googleapis.com/auth/drive.readonly`

### 2. Drive Traversal

- Start from root directory of Google Drive
- Recursively traverse all folders and files
- For each file encountered:
  - Extract metadata (name, path, size, type, modified date)
  - Download file content (if supported file type)
  - Skip system files and unsupported formats

### 3. File Classification

- Use BookWyrm categorization API to classify each file
- Categories may include:
  - Document type (PDF, Word, Text, etc.)
  - Content category (Technical, Literature, Business, etc.)
  - Subject matter classification
  - Language detection

### 4. Data Storage

- Create a Parquet file in `data/` directory with schema:
  ```
  - file_id: string (Google Drive file ID)
  - name: string (file name)
  - path: string (full path in Drive)
  - size: int64 (file size in bytes)
  - mime_type: string (MIME type)
  - modified_time: timestamp
  - category: string (BookWyrm classification)
  - subcategory: string (detailed classification)
  - content_preview: string (first 500 chars of content)
  - embedding: array<float> (vector embedding for RAG)
  ```

### 5. Content Processing

- Extract text content from supported file types:
  - PDF files using PyPDF2 or similar
  - Word documents using python-docx
  - Plain text files directly
  - Google Docs via export API
- Chunk large documents for better RAG performance
- Generate embeddings using configured model

### 6. Index Creation

- Build searchable index from processed content
- Store embeddings for semantic search
- Create metadata index for filtering

## Implementation Components

### GoogleDriveProcessor Class

- `authenticate()`: Handle OAuth2 authentication
- `traverse_drive()`: Recursively scan Drive structure
- `classify_file()`: Use BookWyrm API for categorization
- `extract_content()`: Get text content from files
- `process_file()`: Complete processing pipeline for single file
- `create_index()`: Build final Parquet dataset

### Configuration Updates

- Add Google Drive API settings
- Configure supported file types
- Set classification parameters
- Define output schema

### CLI Integration

- New `index-drive` command in main CLI
- Progress tracking for long-running operations
- Resume capability for interrupted indexing
- Summary reporting

## Error Handling

- Rate limiting for API calls
- Retry logic for failed downloads
- Skip corrupted or inaccessible files
- Log all errors for review

## Security Considerations

- Store credentials securely
- Use read-only Drive access
- Respect file permissions
- Handle sensitive content appropriately

## Performance Optimizations

- Batch API requests where possible
- Parallel processing for file operations
- Incremental updates (only process changed files)
- Caching of classifications and embeddings
