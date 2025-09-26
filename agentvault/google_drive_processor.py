"""Google Drive processor for RAG application."""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import logging

import pandas as pd
import numpy as np
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests

from .config import DATA_DIR, EMBEDDING_MODEL

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

logger = logging.getLogger(__name__)


class GoogleDriveProcessor:
    """Process Google Drive files for RAG application."""
    
    def __init__(self):
        self.service = None
        self.credentials = None
        self.bookwyrm_api_url = "https://api.bookwyrm.ai/classify"  # Placeholder URL
        
    def authenticate(self) -> bool:
        """Authenticate with Google Drive API."""
        creds = None
        token_path = Path("secret/token.pickle")
        
        # Load existing token
        if token_path.exists():
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Find first JSON credentials file in secret directory
                secret_dir = Path("secret")
                creds_files = list(secret_dir.glob("*.json"))
                
                if not creds_files:
                    logger.error("No credentials file found in secret/ directory")
                    return False
                
                creds_file = creds_files[0]
                logger.info(f"Using credentials file: {creds_file}")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_file), SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self.credentials = creds
        self.service = build('drive', 'v3', credentials=creds)
        return True
    
    def list_files(self, folder_id: str = 'root', query: str = None) -> Iterator[Dict[str, Any]]:
        """List files in a Google Drive folder."""
        page_token = None
        
        while True:
            try:
                # Build query
                full_query = f"'{folder_id}' in parents and trashed=false"
                if query:
                    full_query += f" and {query}"
                
                results = self.service.files().list(
                    q=full_query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents, webViewLink)",
                    pageToken=page_token
                ).execute()
                
                items = results.get('files', [])
                for item in items:
                    yield item
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
                    
            except HttpError as error:
                logger.error(f"Error listing files: {error}")
                break
    
    def get_file_path(self, file_id: str, file_name: str) -> str:
        """Get full path of a file in Google Drive."""
        path_parts = [file_name]
        current_id = file_id
        
        try:
            # Get file metadata to find parents
            file_metadata = self.service.files().get(
                fileId=current_id,
                fields="parents"
            ).execute()
            
            parents = file_metadata.get('parents', [])
            
            # Traverse up the parent chain
            while parents and parents[0] != 'root':
                parent_id = parents[0]
                parent_metadata = self.service.files().get(
                    fileId=parent_id,
                    fields="name, parents"
                ).execute()
                
                path_parts.insert(0, parent_metadata.get('name', 'Unknown'))
                parents = parent_metadata.get('parents', [])
        
        except HttpError as error:
            logger.warning(f"Could not get full path for {file_name}: {error}")
        
        return "/" + "/".join(path_parts)
    
    def download_file_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """Download and extract text content from a file."""
        try:
            # Handle Google Workspace files
            if mime_type == 'application/vnd.google-apps.document':
                # Export Google Doc as plain text
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                )
                content = request.execute()
                return content.decode('utf-8')
            
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                # Export Google Sheet as CSV
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType='text/csv'
                )
                content = request.execute()
                return content.decode('utf-8')
            
            elif mime_type in ['text/plain', 'text/csv', 'application/json']:
                # Download plain text files
                request = self.service.files().get_media(fileId=file_id)
                content = request.execute()
                return content.decode('utf-8')
            
            # For other file types, we'd need additional libraries
            # (PyPDF2 for PDFs, python-docx for Word docs, etc.)
            else:
                logger.debug(f"Unsupported file type for content extraction: {mime_type}")
                return None
                
        except HttpError as error:
            logger.error(f"Error downloading file content: {error}")
            return None
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file content for file_id: {file_id}")
            return None
    
    def classify_file(self, file_name: str, content: Optional[str], mime_type: str) -> Dict[str, str]:
        """Classify file using BookWyrm API."""
        # Placeholder implementation - replace with actual BookWyrm API call
        try:
            # Prepare data for classification
            classification_data = {
                "filename": file_name,
                "mime_type": mime_type,
                "content_preview": content[:500] if content else "",
            }
            
            # Make API call to BookWyrm (placeholder)
            # response = requests.post(self.bookwyrm_api_url, json=classification_data)
            # result = response.json()
            
            # For now, return basic classification based on mime type
            if mime_type.startswith('text/'):
                category = "Text Document"
                subcategory = "Plain Text"
            elif 'document' in mime_type:
                category = "Document"
                subcategory = "Word Processing"
            elif 'spreadsheet' in mime_type:
                category = "Data"
                subcategory = "Spreadsheet"
            elif 'presentation' in mime_type:
                category = "Presentation"
                subcategory = "Slides"
            elif mime_type.startswith('image/'):
                category = "Media"
                subcategory = "Image"
            else:
                category = "Other"
                subcategory = "Unknown"
            
            return {
                "category": category,
                "subcategory": subcategory
            }
            
        except Exception as error:
            logger.error(f"Error classifying file: {error}")
            return {
                "category": "Unknown",
                "subcategory": "Error"
            }
    
    def traverse_drive(self, folder_id: str = 'root', path: str = "/") -> Iterator[Dict[str, Any]]:
        """Recursively traverse Google Drive and yield file information."""
        for item in self.list_files(folder_id):
            file_info = {
                'file_id': item['id'],
                'name': item['name'],
                'mime_type': item['mimeType'],
                'size': int(item.get('size', 0)) if item.get('size') else 0,
                'modified_time': item['modifiedTime'],
                'web_view_link': item.get('webViewLink', ''),
                'path': path + item['name']
            }
            
            # If it's a folder, recurse into it
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                file_info['is_folder'] = True
                yield file_info
                
                # Recursively process folder contents
                yield from self.traverse_drive(
                    folder_id=item['id'],
                    path=path + item['name'] + "/"
                )
            else:
                file_info['is_folder'] = False
                
                # Download content and classify
                content = self.download_file_content(item['id'], item['mimeType'])
                classification = self.classify_file(item['name'], content, item['mimeType'])
                
                file_info.update({
                    'content_preview': content[:500] if content else "",
                    'category': classification['category'],
                    'subcategory': classification['subcategory']
                })
                
                yield file_info
    
    def process_drive(self, output_file: str = "google_drive_index.parquet") -> bool:
        """Process entire Google Drive and create Parquet index."""
        if not self.authenticate():
            logger.error("Failed to authenticate with Google Drive")
            return False
        
        logger.info("Starting Google Drive indexing...")
        
        # Collect all file information
        files_data = []
        file_count = 0
        
        try:
            for file_info in self.traverse_drive():
                files_data.append(file_info)
                file_count += 1
                
                if file_count % 100 == 0:
                    logger.info(f"Processed {file_count} files...")
        
        except Exception as error:
            logger.error(f"Error during drive traversal: {error}")
            return False
        
        if not files_data:
            logger.warning("No files found in Google Drive")
            return False
        
        # Create DataFrame and save as Parquet
        df = pd.DataFrame(files_data)
        output_path = DATA_DIR / output_file
        
        try:
            df.to_parquet(output_path, index=False)
            logger.info(f"Successfully created index with {len(files_data)} files")
            logger.info(f"Index saved to: {output_path}")
            return True
            
        except Exception as error:
            logger.error(f"Error saving Parquet file: {error}")
            return False
    
    def create_summary(self, parquet_file: str = "google_drive_index.parquet") -> Dict[str, Any]:
        """Create summary statistics from the processed index."""
        parquet_path = DATA_DIR / parquet_file
        
        if not parquet_path.exists():
            return {"error": "Index file not found"}
        
        try:
            df = pd.read_parquet(parquet_path)
            
            summary = {
                "total_files": len(df),
                "total_folders": len(df[df['is_folder'] == True]),
                "total_documents": len(df[df['is_folder'] == False]),
                "total_size_bytes": df['size'].sum(),
                "categories": df['category'].value_counts().to_dict(),
                "mime_types": df['mime_type'].value_counts().head(10).to_dict(),
                "largest_files": df.nlargest(5, 'size')[['name', 'size', 'category']].to_dict('records')
            }
            
            return summary
            
        except Exception as error:
            logger.error(f"Error creating summary: {error}")
            return {"error": str(error)}
