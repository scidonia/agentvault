"""Google Drive processor for RAG application."""

import json
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import logging

try:
    import pandas as pd
    import numpy as np
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import requests
    import base64
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Please install Google API dependencies:")
    print(
        "  uv add google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pyarrow"
    )
    raise

# Import BookWyrm client for classification and PDF extraction
try:
    from bookwyrm.client import BookWyrmClient, BookWyrmAPIError
    from bookwyrm.models import ClassifyRequest

    # Try to import PDF and phrasal processing models
    try:
        from bookwyrm.models import (
            PDFExtractRequest,
            ProcessTextRequest,
            ResponseFormat,
            SummarizeRequest,
        )

        HAS_PDF_SUPPORT = True
    except ImportError:
        print(
            "⚠️  PDF extraction and phrasal processing not available in this BookWyrm client version"
        )
        HAS_PDF_SUPPORT = False
        PDFExtractRequest = None
        ProcessTextRequest = None
        ResponseFormat = None
        SummarizeRequest = None

except ImportError as e:
    print(f"❌ Missing BookWyrm client: {e}")
    print("Please ensure bookwyrm-client is available")
    print("Try: pip install bookwyrm-client")
    raise

# Import LanceDB and embedding libraries
import lancedb

# Import OpenAI client
try:
    from .openai_client import OpenAIClient, HAS_OPENAI_SUPPORT
except ImportError:
    print("⚠️  OpenAI client not available")
    HAS_OPENAI_SUPPORT = False
    OpenAIClient = None

from .config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    LANCEDB_URI,
    TITLES_TABLE,
    EMBEDDING_DIMENSION,
    OPENAI_API_KEY,
)

# Google Drive API scopes
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

logger = logging.getLogger(__name__)


class GoogleDriveProcessor:
    """Process Google Drive files for RAG application."""

    def __init__(self):
        self.service = None
        self.credentials = None
        self.bookwyrm_client = None
        self.lancedb_client = None
        self.openai_client = None
        self._init_bookwyrm_client()
        self._init_lancedb_client()
        self._init_openai_client()

    def _init_bookwyrm_client(self):
        """Initialize BookWyrm client for classification."""
        try:
            # Get API settings from environment or config
            base_url = os.getenv("BOOKWYRM_API_URL", "https://api.bookwyrm.ai:443")
            api_key = os.getenv("BOOKWYRM_API_KEY")

            if api_key:
                self.bookwyrm_client = BookWyrmClient(
                    base_url=base_url, api_key=api_key
                )
                logger.info("BookWyrm client initialized successfully")
            else:
                logger.warning(
                    "No BOOKWYRM_API_KEY found - classification will use fallback method"
                )
                self.bookwyrm_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize BookWyrm client: {e}")
            self.bookwyrm_client = None

    def authenticate(self, progress_callback=None, debug=False) -> bool:
        """Authenticate with Google Drive API with detailed feedback."""
        creds = None
        token_path = Path("secret/token.pickle")

        def update_progress(message: str):
            if progress_callback:
                progress_callback(message)
            logger.info(message)

        update_progress("🔍 Checking for existing authentication...")

        # Load existing token
        if token_path.exists():
            update_progress("📄 Found existing token file, loading...")
            try:
                with open(token_path, "rb") as token:
                    creds = pickle.load(token)
                update_progress("✅ Successfully loaded existing credentials")
            except Exception as e:
                update_progress(f"⚠️  Error loading token file: {e}")
                creds = None

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                update_progress("🔄 Refreshing expired credentials...")
                try:
                    creds.refresh(Request())
                    update_progress("✅ Successfully refreshed credentials")
                except Exception as e:
                    update_progress(f"❌ Failed to refresh credentials: {e}")
                    creds = None

            if not creds or not creds.valid:
                update_progress(
                    "🔑 Need new credentials, looking for client secrets..."
                )

                # Find first JSON credentials file in secret directory
                secret_dir = Path("secret")
                if not secret_dir.exists():
                    update_progress("❌ Secret directory doesn't exist")
                    logger.error("Secret directory not found")
                    return False

                creds_files = list(secret_dir.glob("*.json"))

                if debug:
                    print(f"DEBUG: Found {len(creds_files)} JSON files in secret/")
                    for f in creds_files:
                        print(f"DEBUG: - {f.name}")

                if not creds_files:
                    update_progress(
                        "❌ No JSON credentials file found in secret/ directory"
                    )
                    update_progress(
                        "💡 Please download your Google Drive API credentials and place them in the secret/ folder"
                    )
                    update_progress(
                        "📖 Visit: https://console.cloud.google.com/apis/credentials"
                    )
                    logger.error("No credentials file found in secret/ directory")
                    return False

                creds_file = creds_files[0]
                update_progress(f"📋 Using credentials file: {creds_file.name}")

                try:
                    update_progress("🌐 Starting OAuth2 flow...")
                    update_progress("🔗 Your browser will open for authentication")

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_file), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    update_progress("✅ Authentication successful!")

                except Exception as e:
                    update_progress(f"❌ Authentication failed: {e}")
                    logger.error(f"OAuth flow failed: {e}")
                    return False

            # Save credentials for next run
            update_progress("💾 Saving credentials for future use...")
            try:
                with open(token_path, "wb") as token:
                    pickle.dump(creds, token)
                update_progress("✅ Credentials saved successfully")
            except Exception as e:
                update_progress(f"⚠️  Warning: Could not save credentials: {e}")

        update_progress("🔧 Building Google Drive service...")
        try:
            if debug:
                print("DEBUG: Setting credentials...")
            self.credentials = creds
            if debug:
                print("DEBUG: Building service...")
            self.service = build("drive", "v3", credentials=creds)
            if debug:
                print("DEBUG: Service built successfully")
            update_progress("✅ Google Drive service ready!")

            # Test the connection
            if debug:
                print("DEBUG: About to test connection...")
            update_progress("🧪 Testing connection...")
            if debug:
                print("DEBUG: Making API call...")
            about = self.service.about().get(fields="user").execute()
            if debug:
                print("DEBUG: API call completed")
            user_email = about.get("user", {}).get("emailAddress", "Unknown")
            update_progress(f"👤 Connected as: {user_email}")
            if debug:
                print(f"DEBUG: Connected as {user_email}")

            return True

        except Exception as e:
            if debug:
                print(f"DEBUG: Exception in service building: {e}")
            update_progress(f"❌ Failed to build Google Drive service: {e}")
            logger.error(f"Service build failed: {e}")
            return False

    def list_files(
        self, folder_id: str = "root", query: str = None
    ) -> Iterator[Dict[str, Any]]:
        """List files in a Google Drive folder."""
        page_token = None

        while True:
            try:
                # Build query
                full_query = f"'{folder_id}' in parents and trashed=false"
                if query:
                    full_query += f" and {query}"

                results = (
                    self.service.files()
                    .list(
                        q=full_query,
                        pageSize=100,
                        fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents, webViewLink)",
                        pageToken=page_token,
                    )
                    .execute()
                )

                items = results.get("files", [])
                for item in items:
                    yield item

                page_token = results.get("nextPageToken")
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
            file_metadata = (
                self.service.files().get(fileId=current_id, fields="parents").execute()
            )

            parents = file_metadata.get("parents", [])

            # Traverse up the parent chain
            while parents and parents[0] != "root":
                parent_id = parents[0]
                parent_metadata = (
                    self.service.files()
                    .get(fileId=parent_id, fields="name, parents")
                    .execute()
                )

                path_parts.insert(0, parent_metadata.get("name", "Unknown"))
                parents = parent_metadata.get("parents", [])

        except HttpError as error:
            logger.warning(f"Could not get full path for {file_name}: {error}")

        return "/" + "/".join(path_parts)

    def download_file_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """Download and extract text content from a file."""
        try:
            # Handle Google Workspace files
            if mime_type == "application/vnd.google-apps.document":
                # Export Google Doc as plain text
                request = self.service.files().export_media(
                    fileId=file_id, mimeType="text/plain"
                )
                content = request.execute()
                return content.decode("utf-8")

            elif mime_type == "application/vnd.google-apps.spreadsheet":
                # Export Google Sheet as CSV
                request = self.service.files().export_media(
                    fileId=file_id, mimeType="text/csv"
                )
                content = request.execute()
                return content.decode("utf-8")

            elif mime_type in ["text/plain", "text/csv", "application/json"]:
                # Download plain text files
                request = self.service.files().get_media(fileId=file_id)
                content = request.execute()
                return content.decode("utf-8")

            # For other file types, we'd need additional libraries
            # (PyPDF2 for PDFs, python-docx for Word docs, etc.)
            else:
                logger.debug(
                    f"Unsupported file type for content extraction: {mime_type}"
                )
                return None

        except HttpError as error:
            logger.error(f"Error downloading file content: {error}")
            return None
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file content for file_id: {file_id}")
            return None

    def classify_file(
        self, file_name: str, content: Optional[str], mime_type: str
    ) -> Dict[str, str]:
        """Classify file using BookWyrm API."""
        try:
            # Try BookWyrm API classification first
            if self.bookwyrm_client and content:
                try:
                    # Encode content as base64 for BookWyrm API
                    content_bytes = content.encode("utf-8")
                    content_b64 = base64.b64encode(content_bytes).decode("ascii")

                    # Create classification request
                    request = ClassifyRequest(
                        content=content_b64,
                        filename=file_name,
                        content_encoding="base64",
                    )

                    # Make API call
                    response = self.bookwyrm_client.classify(request)

                    # Extract classification results
                    classification = response.classification

                    return {
                        "category": classification.format_type,
                        "subcategory": classification.content_type,
                        "language": getattr(classification, "language", "unknown"),
                        "confidence": classification.confidence,
                        "mime_type_detected": classification.mime_type,
                    }

                except BookWyrmAPIError as e:
                    logger.warning(f"BookWyrm API error for {file_name}: {e}")
                except Exception as e:
                    logger.warning(
                        f"BookWyrm classification failed for {file_name}: {e}"
                    )

            # Fallback to basic classification based on mime type
            if mime_type.startswith("text/"):
                category = "Text Document"
                subcategory = "Plain Text"
            elif "document" in mime_type:
                category = "Document"
                subcategory = "Word Processing"
            elif "spreadsheet" in mime_type:
                category = "Data"
                subcategory = "Spreadsheet"
            elif "presentation" in mime_type:
                category = "Presentation"
                subcategory = "Slides"
            elif mime_type.startswith("image/"):
                category = "Media"
                subcategory = "Image"
            else:
                category = "Other"
                subcategory = "Unknown"

            return {
                "category": category,
                "subcategory": subcategory,
                "language": "unknown",
                "confidence": 0.0,
                "mime_type_detected": mime_type,
            }

        except Exception as error:
            logger.error(f"Error classifying file: {error}")
            return {
                "category": "Unknown",
                "subcategory": "Error",
                "language": "unknown",
                "confidence": 0.0,
                "mime_type_detected": mime_type,
            }

    def traverse_drive(
        self, folder_id: str = "root", path: str = "/"
    ) -> Iterator[Dict[str, Any]]:
        """Recursively traverse Google Drive and yield file information."""
        for item in self.list_files(folder_id):
            file_info = {
                "file_id": item["id"],
                "name": item["name"],
                "mime_type": item["mimeType"],
                "size": int(item.get("size", 0)) if item.get("size") else 0,
                "modified_time": item["modifiedTime"],
                "web_view_link": item.get("webViewLink", ""),
                "path": path + item["name"],
            }

            # If it's a folder, recurse into it
            if item["mimeType"] == "application/vnd.google-apps.folder":
                file_info["is_folder"] = True
                yield file_info

                # Recursively process folder contents
                yield from self.traverse_drive(
                    folder_id=item["id"], path=path + item["name"] + "/"
                )
            else:
                file_info["is_folder"] = False

                # Download content and classify
                content = self.download_file_content(item["id"], item["mimeType"])
                classification = self.classify_file(
                    item["name"], content, item["mimeType"]
                )

                file_info.update(
                    {
                        "content_preview": content[:500] if content else "",
                        "category": classification["category"],
                        "subcategory": classification["subcategory"],
                        "language": classification["language"],
                        "classification_confidence": classification["confidence"],
                        "detected_mime_type": classification["mime_type_detected"],
                    }
                )

                yield file_info

    def process_drive(
        self,
        output_file: str = "google_drive_index.parquet",
        progress_callback=None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> bool:
        """Process entire Google Drive and create Parquet index."""
        # Don't re-authenticate if we're already authenticated
        if not self.service:
            if not self.authenticate(progress_callback):
                logger.error("Failed to authenticate with Google Drive")
                return False

        logger.info("Starting Google Drive indexing...")

        # Load existing index to check for duplicates
        output_path = DATA_DIR / output_file
        existing_hashes = set()
        existing_data = []

        if output_path.exists():
            try:
                existing_df = pd.read_parquet(output_path)
                existing_hashes = set(existing_df.get("file_hash", []))
                existing_data = existing_df.to_dict("records")
                logger.info(
                    f"Loaded {len(existing_data)} existing records with {len(existing_hashes)} hashes"
                )
            except Exception as e:
                logger.warning(f"Could not load existing parquet file: {e}")

        # Collect all file information
        files_data = []
        file_count = 0
        folder_count = 0
        error_count = 0
        processed_count = 0
        skipped_count = 0
        duplicate_count = 0

        try:
            for file_info in self.traverse_drive():
                # Apply skip logic
                if file_count < skip:
                    file_count += 1
                    skipped_count += 1
                    continue

                # Apply limit logic
                if limit is not None and processed_count >= limit:
                    break

                # Calculate file hash for deduplication
                file_hash = self._calculate_file_hash(file_info)
                file_info["file_hash"] = file_hash

                # Check if file already exists in index
                if file_hash in existing_hashes:
                    duplicate_count += 1
                    logger.debug(
                        f"Skipping duplicate file: {file_info.get('name', 'Unknown')}"
                    )
                    file_count += 1
                    continue

                files_data.append(file_info)
                file_count += 1
                processed_count += 1

                if file_info.get("is_folder", False):
                    folder_count += 1

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        {
                            "total_files": file_count,
                            "folders": folder_count,
                            "documents": file_count - folder_count,
                            "processed": processed_count,
                            "skipped": skipped_count,
                            "duplicates": duplicate_count,
                            "current_file": file_info.get("name", "Unknown"),
                            "current_path": file_info.get("path", ""),
                            "errors": error_count,
                        }
                    )

                if processed_count % 50 == 0:
                    logger.info(
                        f"Processed {processed_count} new files (total seen: {file_count}, folders: {folder_count}, duplicates: {duplicate_count})..."
                    )

        except Exception as error:
            logger.error(f"Error during drive traversal: {error}")
            error_count += 1
            return False

        if not files_data and not existing_data:
            logger.warning("No files found in Google Drive and no existing data")
            return False

        # Final progress update
        if progress_callback:
            progress_callback(
                {
                    "total_files": file_count,
                    "folders": folder_count,
                    "documents": file_count - folder_count,
                    "processed": processed_count,
                    "skipped": skipped_count,
                    "duplicates": duplicate_count,
                    "current_file": "Saving to Parquet...",
                    "current_path": "",
                    "errors": error_count,
                    "phase": "saving",
                }
            )

        # Combine existing data with new data
        all_data = existing_data + files_data

        # Create DataFrame and save as Parquet
        df = pd.DataFrame(all_data)

        try:
            df.to_parquet(output_path, index=False)
            logger.info(
                f"Successfully updated index: {len(existing_data)} existing + {len(files_data)} new = {len(all_data)} total files"
            )
            logger.info(f"Index saved to: {output_path}")
            return True

        except Exception as error:
            logger.error(f"Error saving Parquet file: {error}")
            return False

    def create_summary(
        self, parquet_file: str = "google_drive_index.parquet"
    ) -> Dict[str, Any]:
        """Create summary statistics from the processed index."""
        parquet_path = DATA_DIR / parquet_file

        if not parquet_path.exists():
            return {"error": "Index file not found"}

        try:
            df = pd.read_parquet(parquet_path)

            summary = {
                "total_files": len(df),
                "total_folders": len(df[df["is_folder"] == True]),
                "total_documents": len(df[df["is_folder"] == False]),
                "total_size_bytes": df["size"].sum(),
                "categories": df["category"].value_counts().to_dict(),
                "mime_types": df["mime_type"].value_counts().head(10).to_dict(),
                "largest_files": df.nlargest(5, "size")[
                    ["name", "size", "category"]
                ].to_dict("records"),
            }

            return summary

        except Exception as error:
            logger.error(f"Error creating summary: {error}")
            return {"error": str(error)}

    def _calculate_file_hash(self, file_info: Dict[str, Any]) -> str:
        """Calculate a hash for the file based on its metadata."""
        # Use file ID, name, size, and modified time to create a unique hash
        hash_data = {
            "file_id": file_info.get("file_id", ""),
            "name": file_info.get("name", ""),
            "size": file_info.get("size", 0),
            "modified_time": file_info.get("modified_time", ""),
            "mime_type": file_info.get("mime_type", ""),
        }

        # Create a consistent string representation
        hash_string = json.dumps(hash_data, sort_keys=True)

        # Generate SHA-256 hash
        return hashlib.sha256(hash_string.encode("utf-8")).hexdigest()

    def extract_pdf_text(self, file_id: str, file_name: str) -> Optional[str]:
        """Extract full text from PDF using BookWyrm API."""
        if not self.bookwyrm_client:
            logger.warning("BookWyrm client not available for PDF extraction")
            return None

        if not HAS_PDF_SUPPORT:
            logger.warning(
                "PDF extraction not supported in this BookWyrm client version"
            )
            return None

        try:
            # Download PDF content as binary
            request = self.service.files().get_media(fileId=file_id)
            pdf_content = request.execute()

            # Encode as base64 for BookWyrm API
            pdf_b64 = base64.b64encode(pdf_content).decode("ascii")

            # Create PDF extraction request
            extract_request = PDFExtractRequest(pdf_content=pdf_b64, filename=file_name)

            # Make API call
            response = self.bookwyrm_client.extract_pdf(extract_request)

            # Combine all text from all pages
            full_text = []
            for page in response.pages:
                page_text = []
                for text_block in page.text_blocks:
                    page_text.append(text_block.text)
                if page_text:
                    full_text.append(" ".join(page_text))

            return "\n\n".join(full_text) if full_text else None

        except BookWyrmAPIError as e:
            logger.error(f"BookWyrm PDF extraction error for {file_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_name}: {e}")
            return None

    def process_pdf_extractions(
        self,
        index_file: str = "google_drive_index.parquet",
        output_file: str = "pdf_extractions.parquet",
        progress_callback=None,
        limit: Optional[int] = None,
    ) -> bool:
        """Extract text from all PDFs in the index and save to parquet."""

        # Check prerequisites first
        if not self.bookwyrm_client:
            logger.error(
                "BookWyrm client not available. Please set BOOKWYRM_API_KEY environment variable."
            )
            return False

        if not HAS_PDF_SUPPORT:
            logger.error(
                "PDF extraction not supported in this BookWyrm client version. Please upgrade bookwyrm-client."
            )
            return False

        index_path = DATA_DIR / index_file
        if not index_path.exists():
            logger.error(f"Index file not found: {index_path}")
            return False

        try:
            # Load the index
            df = pd.read_parquet(index_path)

            # Filter for PDF files only
            pdf_files = df[
                (df["mime_type"] == "application/pdf") & (df["is_folder"] == False)
            ].copy()

            if pdf_files.empty:
                logger.warning("No PDF files found in index")
                return False

            logger.info(f"Found {len(pdf_files)} PDF files to process")

            # Apply limit if specified
            if limit:
                pdf_files = pdf_files.head(limit)
                logger.info(f"Processing limited to {len(pdf_files)} files")

            # Check for existing extractions
            output_path = DATA_DIR / output_file
            existing_hashes = set()
            existing_data = []

            if output_path.exists():
                try:
                    existing_df = pd.read_parquet(output_path)
                    existing_hashes = set(existing_df["file_hash"])
                    existing_data = existing_df.to_dict("records")
                    logger.info(f"Found {len(existing_data)} existing extractions")
                except Exception as e:
                    logger.warning(f"Could not load existing extractions: {e}")

            # Process PDFs
            extractions = []
            processed_count = 0
            skipped_count = 0
            error_count = 0

            for _, row in pdf_files.iterrows():
                file_hash = row["file_hash"]
                file_name = row["name"]
                file_id = row["file_id"]

                # Skip if already processed
                if file_hash in existing_hashes:
                    skipped_count += 1
                    continue

                # Update progress
                if progress_callback:
                    progress_callback(
                        {
                            "current_file": file_name,
                            "processed": processed_count,
                            "total": len(pdf_files),
                            "skipped": skipped_count,
                            "errors": error_count,
                        }
                    )

                # Extract text
                extracted_text = self.extract_pdf_text(file_id, file_name)

                if extracted_text:
                    extractions.append(
                        {
                            "file_hash": file_hash,
                            "file_name": file_name,
                            "file_id": file_id,
                            "extracted_text": extracted_text,
                            "text_length": len(extracted_text),
                            "extraction_timestamp": pd.Timestamp.now().isoformat(),
                        }
                    )
                    processed_count += 1
                    logger.info(
                        f"Extracted {len(extracted_text)} characters from {file_name}"
                    )
                else:
                    error_count += 1
                    logger.warning(f"Failed to extract text from {file_name}")

            # Final progress update
            if progress_callback:
                progress_callback(
                    {
                        "current_file": "Saving extractions...",
                        "processed": processed_count,
                        "total": len(pdf_files),
                        "skipped": skipped_count,
                        "errors": error_count,
                        "phase": "saving",
                    }
                )

            # Combine with existing data and save
            all_extractions = existing_data + extractions

            if all_extractions:
                df_extractions = pd.DataFrame(all_extractions)
                df_extractions.to_parquet(output_path, index=False)
                logger.info(
                    f"Saved {len(all_extractions)} total extractions to {output_path}"
                )
                return True
            else:
                logger.warning("No extractions to save")
                return False

        except Exception as e:
            logger.error(f"Error processing PDF extractions: {e}")
            return False

    def process_phrases_from_all_content(
        self,
        index_file: str = "google_drive_index.parquet",
        pdf_extractions_file: str = "pdf_extractions.parquet",
        output_file: str = "content_phrases.parquet",
        progress_callback=None,
        limit: Optional[int] = None,
    ) -> bool:
        """Process all text content (PDF extractions + raw text) into phrases using BookWyrm phrasal API."""

        if not self.bookwyrm_client:
            logger.error("BookWyrm client not available for phrasal processing")
            return False

        if not HAS_PDF_SUPPORT:
            logger.error(
                "Phrasal processing not supported in this BookWyrm client version"
            )
            return False

        index_path = DATA_DIR / index_file
        if not index_path.exists():
            logger.error(f"Index file not found: {index_path}")
            return False

        try:
            # Load the main index
            df_index = pd.read_parquet(index_path)

            # Load PDF extractions if available
            pdf_extractions = {}
            pdf_extractions_path = DATA_DIR / pdf_extractions_file
            if pdf_extractions_path.exists():
                try:
                    df_pdf = pd.read_parquet(pdf_extractions_path)
                    pdf_extractions = dict(
                        zip(df_pdf["file_hash"], df_pdf["extracted_text"])
                    )
                    logger.info(f"Loaded {len(pdf_extractions)} PDF extractions")
                except Exception as e:
                    logger.warning(f"Could not load PDF extractions: {e}")

            # Filter for files with content (either raw content or PDF extractions)
            content_files = []

            for _, row in df_index.iterrows():
                file_hash = row["file_hash"]
                file_name = row["name"]

                # Skip folders
                if row.get("is_folder", False):
                    continue

                # Get content from either raw content preview or PDF extraction
                content = None
                content_source = None

                # Check for PDF extraction first (more complete)
                if file_hash in pdf_extractions:
                    content = pdf_extractions[file_hash]
                    content_source = "pdf_extraction"
                # Fall back to raw content preview
                elif (
                    row.get("content_preview")
                    and len(row["content_preview"].strip()) > 100
                ):
                    content = row["content_preview"]
                    content_source = "raw_content"

                if content:
                    content_files.append(
                        {
                            "file_hash": file_hash,
                            "file_name": file_name,
                            "content": content,
                            "content_source": content_source,
                            "mime_type": row.get("mime_type", ""),
                            "category": row.get("category", ""),
                            "language": row.get("language", "unknown"),
                        }
                    )

            if not content_files:
                logger.warning("No files with content found for phrasal processing")
                return False

            logger.info(
                f"Found {len(content_files)} files with content to process into phrases"
            )

            # Apply limit if specified
            if limit:
                content_files = content_files[:limit]
                logger.info(f"Processing limited to {len(content_files)} files")

            # Check for existing phrases
            output_path = DATA_DIR / output_file
            existing_hashes = set()
            existing_data = []

            if output_path.exists():
                try:
                    existing_df = pd.read_parquet(output_path)
                    existing_hashes = set(existing_df["file_hash"])
                    existing_data = existing_df.to_dict("records")
                    logger.info(f"Found {len(existing_data)} existing phrase records")
                except Exception as e:
                    logger.warning(f"Could not load existing phrases: {e}")

            # Process each file for phrasal processing
            all_phrases = []
            processed_count = 0
            skipped_count = 0
            error_count = 0

            for file_info in content_files:
                file_hash = file_info["file_hash"]
                file_name = file_info["file_name"]
                content = file_info["content"]

                # Skip if already processed
                if file_hash in existing_hashes:
                    skipped_count += 1
                    continue

                # Update progress
                if progress_callback:
                    progress_callback(
                        {
                            "current_file": file_name,
                            "processed": processed_count,
                            "total": len(content_files),
                            "skipped": skipped_count,
                            "errors": error_count,
                        }
                    )

                try:
                    # Create phrasal processing request
                    request = ProcessTextRequest(
                        text=content,
                        response_format=ResponseFormat.WITH_OFFSETS,
                        spacy_model="en_core_web_sm",
                    )

                    # Process text into phrases
                    phrases = []
                    phrase_count = 0

                    for response in self.bookwyrm_client.process_text(request):
                        if hasattr(response, "text"):  # PhraseResult
                            phrases.append(
                                {
                                    "file_hash": file_hash,
                                    "file_name": file_name,
                                    "content_source": file_info["content_source"],
                                    "phrase_count": phrase_count,
                                    "phrase": response.text,
                                    "start_char": response.start_char,
                                    "end_char": response.end_char,
                                    "mime_type": file_info["mime_type"],
                                    "category": file_info["category"],
                                    "language": file_info["language"],
                                }
                            )
                            phrase_count += 1

                    all_phrases.extend(phrases)
                    processed_count += 1
                    logger.info(f"Processed {len(phrases)} phrases from {file_name}")

                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to process phrases for {file_name}: {e}")

            # Final progress update
            if progress_callback:
                progress_callback(
                    {
                        "current_file": "Saving phrases...",
                        "processed": processed_count,
                        "total": len(content_files),
                        "skipped": skipped_count,
                        "errors": error_count,
                        "phase": "saving",
                    }
                )

            # Combine with existing data and save
            all_phrase_data = existing_data + all_phrases

            if all_phrase_data:
                df_phrases = pd.DataFrame(all_phrase_data)
                df_phrases.to_parquet(output_path, index=False)
                logger.info(
                    f"Saved {len(all_phrase_data)} total phrase records to {output_path}"
                )
                return True
            else:
                logger.warning("No phrases to save")
                return False

        except Exception as e:
            logger.error(f"Error processing phrases: {e}")
            return False

    def process_summaries_from_phrases(
        self,
        phrases_file: str = "content_phrases.parquet",
        output_file: str = "content_summaries.parquet",
        progress_callback=None,
        limit: Optional[int] = None,
        max_tokens: int = 10000,
    ) -> bool:
        """Create summaries from phrasal content using BookWyrm API."""

        if not self.bookwyrm_client:
            logger.error("BookWyrm client not available for summarization")
            return False

        if not HAS_PDF_SUPPORT or not SummarizeRequest:
            logger.error("Summarization not supported in this BookWyrm client version")
            return False

        phrases_path = DATA_DIR / phrases_file
        if not phrases_path.exists():
            logger.error(f"Phrases file not found: {phrases_path}")
            return False

        try:
            # Load phrases
            df_phrases = pd.read_parquet(phrases_path)

            if df_phrases.empty:
                logger.warning("No phrases found for summarization")
                return False

            # Group phrases by file_hash to create phrases array per file
            files_to_summarize = []

            for file_hash, group in df_phrases.groupby("file_hash"):
                # Sort phrases by phrase_count to maintain order
                sorted_phrases = group.sort_values("phrase_count")

                # Create phrases array from phrases (as per BookWyrm API spec)
                phrases_array = []
                for _, phrase_row in sorted_phrases.iterrows():
                    phrase_obj = {
                        "text": phrase_row["phrase"],
                        "start_char": phrase_row.get("start_char"),
                        "end_char": phrase_row.get("end_char"),
                    }
                    phrases_array.append(phrase_obj)

                # Get metadata from first phrase record
                first_record = sorted_phrases.iloc[0]

                files_to_summarize.append(
                    {
                        "file_hash": file_hash,
                        "file_name": first_record["file_name"],
                        "phrases": phrases_array,
                        "content_source": first_record["content_source"],
                        "mime_type": first_record["mime_type"],
                        "category": first_record["category"],
                        "language": first_record["language"],
                        "phrase_count": len(sorted_phrases),
                    }
                )

            logger.info(
                f"Found {len(files_to_summarize)} files with phrases to summarize"
            )

            # Apply limit if specified
            if limit:
                files_to_summarize = files_to_summarize[:limit]
                logger.info(f"Processing limited to {len(files_to_summarize)} files")

            # Check for existing summaries
            output_path = DATA_DIR / output_file
            existing_hashes = set()
            existing_data = []

            if output_path.exists():
                try:
                    existing_df = pd.read_parquet(output_path)
                    existing_hashes = set(existing_df["file_hash"])
                    existing_data = existing_df.to_dict("records")
                    logger.info(f"Found {len(existing_data)} existing summaries")
                except Exception as e:
                    logger.warning(f"Could not load existing summaries: {e}")

            # Process each file for summarization
            summaries = []
            processed_count = 0
            skipped_count = 0
            error_count = 0

            for file_info in files_to_summarize:
                file_hash = file_info["file_hash"]
                file_name = file_info["file_name"]
                phrases = file_info["phrases"]

                # Skip if already processed
                if file_hash in existing_hashes:
                    skipped_count += 1
                    continue

                # Update progress
                if progress_callback:
                    progress_callback(
                        {
                            "current_file": file_name,
                            "processed": processed_count,
                            "total": len(files_to_summarize),
                            "skipped": skipped_count,
                            "errors": error_count,
                        }
                    )

                try:
                    # Validate phrases array
                    if not phrases or len(phrases) == 0:
                        logger.warning(f"Skipping {file_name}: No phrases found")
                        error_count += 1
                        continue

                    # Clean up phrases - ensure all required fields are present
                    clean_phrases = []
                    for phrase in phrases:
                        if phrase.get("text") and len(phrase["text"].strip()) > 0:
                            clean_phrase = {
                                "text": phrase["text"],
                                "start_char": (
                                    phrase.get("start_char")
                                    if phrase.get("start_char") is not None
                                    else 0
                                ),
                                "end_char": (
                                    phrase.get("end_char")
                                    if phrase.get("end_char") is not None
                                    else len(phrase["text"])
                                ),
                            }
                            clean_phrases.append(clean_phrase)

                    if not clean_phrases:
                        logger.warning(f"Skipping {file_name}: No valid phrases found")
                        error_count += 1
                        continue

                    logger.info(
                        f"Sending {len(clean_phrases)} phrases for summarization of {file_name}"
                    )

                    # Create summarization request with phrases array
                    request = SummarizeRequest(
                        phrases=clean_phrases, max_tokens=max_tokens, debug=False
                    )

                    # Make API call for summarization
                    response = self.bookwyrm_client.summarize(request)

                    # Calculate original length from phrases
                    original_length = sum(len(p["text"]) for p in clean_phrases)

                    summaries.append(
                        {
                            "file_hash": file_hash,
                            "file_name": file_name,
                            "summary": response.summary,
                            "content_source": file_info["content_source"],
                            "original_length": original_length,
                            "summary_length": len(response.summary),
                            "phrase_count": file_info["phrase_count"],
                            "subsummary_count": response.subsummary_count,
                            "levels_used": response.levels_used,
                            "total_tokens": response.total_tokens,
                            "mime_type": file_info["mime_type"],
                            "category": file_info["category"],
                            "language": file_info["language"],
                            "summarization_timestamp": pd.Timestamp.now().isoformat(),
                        }
                    )

                    processed_count += 1
                    logger.info(
                        f"Created summary for {file_name} ({len(response.summary)} chars from {file_info['phrase_count']} phrases)"
                    )

                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to create summary for {file_name}: {e}")
                    # Log more details for debugging
                    if hasattr(e, "status_code"):
                        logger.error(f"HTTP Status: {e.status_code}")

            # Final progress update
            if progress_callback:
                progress_callback(
                    {
                        "current_file": "Saving summaries...",
                        "processed": processed_count,
                        "total": len(files_to_summarize),
                        "skipped": skipped_count,
                        "errors": error_count,
                        "phase": "saving",
                    }
                )

            # Combine with existing data and save
            all_summaries = existing_data + summaries

            if all_summaries:
                df_summaries = pd.DataFrame(all_summaries)
                df_summaries.to_parquet(output_path, index=False)
                logger.info(
                    f"Saved {len(all_summaries)} total summaries to {output_path}"
                )
                return True
            else:
                logger.warning("No summaries to save")
                return False

        except Exception as e:
            logger.error(f"Error processing summaries: {e}")
            return False

    def process_summaries_via_endpoint(
        self,
        phrases_file: str = "content_phrases.parquet",
        output_file: str = "content_summaries.parquet",
        progress_callback=None,
        limit: Optional[int] = None,
        max_tokens: int = 10000,
        endpoint_url: str = "https://api.bookwyrm.ai:443",
        api_token: str = None,
    ) -> bool:
        """Create summaries from phrasal content using the summarize-endpoint service."""

        if not api_token:
            logger.error("API token required for summarize endpoint")
            return False

        phrases_path = DATA_DIR / phrases_file
        if not phrases_path.exists():
            logger.error(f"Phrases file not found: {phrases_path}")
            return False

        try:
            import httpx
            import json

            # Load phrases
            df_phrases = pd.read_parquet(phrases_path)

            if df_phrases.empty:
                logger.warning("No phrases found for summarization")
                return False

            # Group phrases by file_hash to create phrases array per file
            files_to_summarize = []

            for file_hash, group in df_phrases.groupby("file_hash"):
                # Sort phrases by phrase_count to maintain order
                sorted_phrases = group.sort_values("phrase_count")

                # Create phrases array from phrases
                phrases_array = []
                for _, phrase_row in sorted_phrases.iterrows():
                    phrase_obj = {
                        "text": phrase_row["phrase"],
                        "start_char": phrase_row.get("start_char"),
                        "end_char": phrase_row.get("end_char"),
                    }
                    phrases_array.append(phrase_obj)

                # Get metadata from first phrase record
                first_record = sorted_phrases.iloc[0]

                files_to_summarize.append(
                    {
                        "file_hash": file_hash,
                        "file_name": first_record["file_name"],
                        "phrases": phrases_array,
                        "content_source": first_record["content_source"],
                        "mime_type": first_record["mime_type"],
                        "category": first_record["category"],
                        "language": first_record["language"],
                        "phrase_count": len(sorted_phrases),
                    }
                )

            logger.info(
                f"Found {len(files_to_summarize)} files with phrases to summarize"
            )

            # Apply limit if specified
            if limit:
                files_to_summarize = files_to_summarize[:limit]
                logger.info(f"Processing limited to {len(files_to_summarize)} files")

            # Check for existing summaries
            output_path = DATA_DIR / output_file
            existing_hashes = set()
            existing_data = []

            if output_path.exists():
                try:
                    existing_df = pd.read_parquet(output_path)
                    existing_hashes = set(existing_df["file_hash"])
                    existing_data = existing_df.to_dict("records")
                    logger.info(f"Found {len(existing_data)} existing summaries")
                except Exception as e:
                    logger.warning(f"Could not load existing summaries: {e}")

            # Process each file for summarization
            summaries = []
            processed_count = 0
            skipped_count = 0
            error_count = 0

            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            }

            for file_info in files_to_summarize:
                file_hash = file_info["file_hash"]
                file_name = file_info["file_name"]
                phrases = file_info["phrases"]

                # Skip if already processed
                if file_hash in existing_hashes:
                    skipped_count += 1
                    continue

                # Update progress
                if progress_callback:
                    progress_callback(
                        {
                            "current_file": file_name,
                            "processed": processed_count,
                            "total": len(files_to_summarize),
                            "skipped": skipped_count,
                            "errors": error_count,
                        }
                    )

                try:
                    # Validate phrases array
                    if not phrases or len(phrases) == 0:
                        logger.warning(f"Skipping {file_name}: No phrases found")
                        error_count += 1
                        continue

                    # Clean up phrases - ensure all required fields are present
                    clean_phrases = []
                    for phrase in phrases:
                        if phrase.get("text") and len(phrase["text"].strip()) > 0:
                            clean_phrase = {
                                "text": phrase["text"],
                                "start_char": (
                                    phrase.get("start_char")
                                    if phrase.get("start_char") is not None
                                    else 0
                                ),
                                "end_char": (
                                    phrase.get("end_char")
                                    if phrase.get("end_char") is not None
                                    else len(phrase["text"])
                                ),
                            }
                            clean_phrases.append(clean_phrase)

                    if not clean_phrases:
                        logger.warning(f"Skipping {file_name}: No valid phrases found")
                        error_count += 1
                        continue

                    logger.info(
                        f"Sending {len(clean_phrases)} phrases for summarization of {file_name}"
                    )

                    # Prepare request payload
                    request_data = {
                        "phrases": clean_phrases,
                        "max_tokens": max_tokens,
                        "debug": False,
                    }

                    # Make HTTP request to summarize endpoint
                    with httpx.Client(timeout=300.0) as client:
                        response = client.post(
                            f"{endpoint_url}/summarize",
                            json=request_data,
                            headers=headers,
                        )

                    if response.status_code == 200:
                        # Parse streaming response
                        lines = response.text.strip().split("\n")
                        final_summary = None

                        for line in lines:
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(
                                        line[6:]
                                    )  # Remove 'data: ' prefix
                                    if data.get("type") == "summary":
                                        final_summary = data
                                        break
                                except json.JSONDecodeError:
                                    continue

                        if final_summary:
                            # Calculate original length from phrases
                            original_length = sum(len(p["text"]) for p in clean_phrases)

                            summaries.append(
                                {
                                    "file_hash": file_hash,
                                    "file_name": file_name,
                                    "summary": final_summary["summary"],
                                    "content_source": file_info["content_source"],
                                    "original_length": original_length,
                                    "summary_length": len(final_summary["summary"]),
                                    "phrase_count": file_info["phrase_count"],
                                    "subsummary_count": final_summary[
                                        "subsummary_count"
                                    ],
                                    "levels_used": final_summary["levels_used"],
                                    "total_tokens": final_summary["total_tokens"],
                                    "mime_type": file_info["mime_type"],
                                    "category": file_info["category"],
                                    "language": file_info["language"],
                                    "summarization_timestamp": pd.Timestamp.now().isoformat(),
                                }
                            )

                            processed_count += 1
                            logger.info(
                                f"Created summary for {file_name} ({len(final_summary['summary'])} chars from {file_info['phrase_count']} phrases)"
                            )
                        else:
                            error_count += 1
                            logger.error(f"No final summary received for {file_name}")
                    else:
                        error_count += 1
                        logger.error(
                            f"HTTP {response.status_code} error for {file_name}: {response.text[:200]}"
                        )

                except httpx.TimeoutException:
                    error_count += 1
                    logger.error(f"Timeout error for {file_name}")
                except httpx.ConnectError:
                    error_count += 1
                    logger.error(
                        f"Connection error for {file_name} - is the endpoint running?"
                    )
                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to create summary for {file_name}: {e}")

            # Final progress update
            if progress_callback:
                progress_callback(
                    {
                        "current_file": "Saving summaries...",
                        "processed": processed_count,
                        "total": len(files_to_summarize),
                        "skipped": skipped_count,
                        "errors": error_count,
                        "phase": "saving",
                    }
                )

            # Combine with existing data and save
            all_summaries = existing_data + summaries

            if all_summaries:
                df_summaries = pd.DataFrame(all_summaries)
                df_summaries.to_parquet(output_path, index=False)
                logger.info(
                    f"Saved {len(all_summaries)} total summaries to {output_path}"
                )
                return True
            else:
                logger.warning("No summaries to save")
                return False

        except Exception as e:
            logger.error(f"Error processing summaries via endpoint: {e}")
            return False

    def extract_title_and_author(
        self, file_name: str, summary: str, content_preview: str = ""
    ) -> Dict[str, str]:
        """Extract title and author information from file name, summary, and content."""
        import re

        # Initialize result
        result = {"title": "", "author": "", "extracted_from": ""}

        # Clean up file name (remove extension, common prefixes)
        clean_name = file_name
        if "." in clean_name:
            clean_name = clean_name.rsplit(".", 1)[0]

        # Remove common prefixes/suffixes
        clean_name = re.sub(
            r"^(copy of |draft |final |v\d+|version \d+)",
            "",
            clean_name,
            flags=re.IGNORECASE,
        )
        clean_name = re.sub(
            r"(draft|final|v\d+|version \d+)$", "", clean_name, flags=re.IGNORECASE
        )
        clean_name = clean_name.strip()

        # Try to extract from summary first (most reliable)
        if summary:
            # Look for title patterns in summary
            title_patterns = [
                r'"([^"]+)"',  # Quoted titles
                r'titled "([^"]+)"',
                r'title[:\s]+"([^"]+)"',
                r"work titled ([^,\n\.]+)",
                r"document titled ([^,\n\.]+)",
                r"book titled ([^,\n\.]+)",
                r"story titled ([^,\n\.]+)",
            ]

            for pattern in title_patterns:
                match = re.search(pattern, summary, re.IGNORECASE)
                if match:
                    result["title"] = match.group(1).strip()
                    result["extracted_from"] = "summary"
                    break

            # Look for author patterns in summary
            author_patterns = [
                r"by ([A-Z][a-z]+ [A-Z][a-z]+)",  # by FirstName LastName
                r"author ([A-Z][a-z]+ [A-Z][a-z]+)",
                r"written by ([A-Z][a-z]+ [A-Z][a-z]+)",
                r"([A-Z][a-z]+ [A-Z][a-z]+)\'s",  # Author's work
            ]

            for pattern in author_patterns:
                match = re.search(pattern, summary, re.IGNORECASE)
                if match:
                    result["author"] = match.group(1).strip()
                    break

        # Try to extract from content preview if no title found
        if not result["title"] and content_preview:
            # Look for title at the beginning of content
            lines = content_preview.split("\n")[:5]  # First 5 lines
            for line in lines:
                line = line.strip()
                if len(line) > 5 and len(line) < 100:  # Reasonable title length
                    # Check if it looks like a title (not too many common words)
                    common_words = [
                        "the",
                        "and",
                        "or",
                        "but",
                        "in",
                        "on",
                        "at",
                        "to",
                        "for",
                        "of",
                        "with",
                        "by",
                    ]
                    words = line.lower().split()
                    if (
                        len(words) > 0
                        and sum(1 for w in words if w in common_words) / len(words)
                        < 0.5
                    ):
                        result["title"] = line
                        result["extracted_from"] = "content"
                        break

        # Fall back to cleaned file name if no title found
        if not result["title"]:
            result["title"] = clean_name
            result["extracted_from"] = "filename"

        return result

    def create_title_cards(
        self,
        summaries_file: str = "content_summaries.parquet",
        index_file: str = "google_drive_index.parquet",
        output_file: str = "title_cards.parquet",
        progress_callback=None,
        limit: Optional[int] = None,
    ) -> bool:
        """Create title cards from summaries with extracted metadata."""

        summaries_path = DATA_DIR / summaries_file
        index_path = DATA_DIR / index_file

        if not summaries_path.exists():
            logger.error(f"Summaries file not found: {summaries_path}")
            return False

        if not index_path.exists():
            logger.error(f"Index file not found: {index_path}")
            return False

        try:
            # Load summaries and index
            df_summaries = pd.read_parquet(summaries_path)
            df_index = pd.read_parquet(index_path)

            if df_summaries.empty:
                logger.warning("No summaries found for title card creation")
                return False

            logger.info(
                f"Found {len(df_summaries)} summaries to process into title cards"
            )

            # Apply limit if specified
            if limit:
                df_summaries = df_summaries.head(limit)
                logger.info(f"Processing limited to {len(df_summaries)} summaries")

            # Create a lookup for index data by file_hash
            index_lookup = df_index.set_index("file_hash").to_dict("index")

            # Check for existing title cards
            output_path = DATA_DIR / output_file
            existing_hashes = set()
            existing_data = []

            if output_path.exists():
                try:
                    existing_df = pd.read_parquet(output_path)
                    existing_hashes = set(existing_df["file_hash"])
                    existing_data = existing_df.to_dict("records")
                    logger.info(f"Found {len(existing_data)} existing title cards")
                except Exception as e:
                    logger.warning(f"Could not load existing title cards: {e}")

            # Process each summary into a title card
            title_cards = []
            processed_count = 0
            skipped_count = 0
            error_count = 0

            for _, summary_row in df_summaries.iterrows():
                file_hash = summary_row["file_hash"]
                file_name = summary_row["file_name"]

                # Skip if already processed
                if file_hash in existing_hashes:
                    skipped_count += 1
                    continue

                # Update progress
                if progress_callback:
                    progress_callback(
                        {
                            "current_file": file_name,
                            "processed": processed_count,
                            "total": len(df_summaries),
                            "skipped": skipped_count,
                            "errors": error_count,
                        }
                    )

                try:
                    # Get additional metadata from index
                    index_data = index_lookup.get(file_hash, {})
                    content_preview = index_data.get("content_preview", "")

                    # Extract title and author
                    title_info = self.extract_title_and_author(
                        file_name, summary_row["summary"], content_preview
                    )

                    # Create title card
                    title_card = {
                        "file_hash": file_hash,
                        "title": title_info["title"],
                        "author": title_info["author"],
                        "title_extracted_from": title_info["extracted_from"],
                        "file_name": file_name,
                        "summary": summary_row["summary"],
                        "category": summary_row.get("category", "Unknown"),
                        "subcategory": index_data.get("subcategory", "Unknown"),
                        "language": summary_row.get("language", "unknown"),
                        "mime_type": summary_row.get("mime_type", ""),
                        "content_source": summary_row.get("content_source", ""),
                        "original_length": summary_row.get("original_length", 0),
                        "summary_length": summary_row.get("summary_length", 0),
                        "phrase_count": summary_row.get("phrase_count", 0),
                        "levels_used": summary_row.get("levels_used", 0),
                        "total_tokens": summary_row.get("total_tokens", 0),
                        "file_size": index_data.get("size", 0),
                        "modified_time": index_data.get("modified_time", ""),
                        "web_view_link": index_data.get("web_view_link", ""),
                        "path": index_data.get("path", ""),
                        "classification_confidence": index_data.get(
                            "classification_confidence", 0.0
                        ),
                        "summarization_timestamp": summary_row.get(
                            "summarization_timestamp", ""
                        ),
                        "title_card_timestamp": pd.Timestamp.now().isoformat(),
                    }

                    title_cards.append(title_card)
                    processed_count += 1
                    logger.info(
                        f"Created title card for '{title_info['title']}' by {title_info['author'] or 'Unknown'}"
                    )

                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to create title card for {file_name}: {e}")

            # Final progress update
            if progress_callback:
                progress_callback(
                    {
                        "current_file": "Saving title cards...",
                        "processed": processed_count,
                        "total": len(df_summaries),
                        "skipped": skipped_count,
                        "errors": error_count,
                        "phase": "saving",
                    }
                )

            # Combine with existing data and save
            all_title_cards = existing_data + title_cards

            if all_title_cards:
                df_title_cards = pd.DataFrame(all_title_cards)
                df_title_cards.to_parquet(output_path, index=False)
                logger.info(
                    f"Saved {len(all_title_cards)} total title cards to {output_path}"
                )
                return True
            else:
                logger.warning("No title cards to save")
                return False

        except Exception as e:
            logger.error(f"Error creating title cards: {e}")
            return False

    def _init_lancedb_client(self):
        """Initialize LanceDB client."""
        try:
            # Initialize LanceDB - create directory if it doesn't exist
            from pathlib import Path
            db_path = Path(LANCEDB_URI)
            
            # Ensure the parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to LanceDB (this will create the database if it doesn't exist)
            self.lancedb_client = lancedb.connect(str(db_path))
            logger.info(f"LanceDB client initialized successfully at {db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB client: {e}")
            self.lancedb_client = None

    def _init_openai_client(self):
        """Initialize OpenAI client for embeddings."""
        if not HAS_OPENAI_SUPPORT:
            logger.warning("OpenAI support not available - will use fallback embedding")
            return

        if not OPENAI_API_KEY:
            logger.warning("No OPENAI_API_KEY found - will use fallback embedding")
            return

        try:
            self.openai_client = OpenAIClient()
            logger.info("OpenAI client initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None

    def _ensure_lancedb_table(self):
        """Ensure LanceDB table exists, create it if it doesn't."""
        if self.lancedb_client is None:
            return False

        try:
            # Check if table exists
            existing_tables = self.lancedb_client.table_names()

            if TITLES_TABLE not in existing_tables:
                logger.info(f"Creating LanceDB table: {TITLES_TABLE}")

                # Create a sample record to define schema
                sample_data = [
                    {
                        "id": "sample",
                        "text": "sample text",
                        "vector": [0.0] * EMBEDDING_DIMENSION,
                        "file_hash": "sample",
                        "title": "sample",
                        "author": "sample",
                        "file_name": "sample",
                        "category": "sample",
                        "subcategory": "sample",
                        "language": "sample",
                        "mime_type": "sample",
                        "content_source": "sample",
                        "summary_length": 0,
                        "phrase_count": 0,
                        "file_size": 0,
                        "path": "sample",
                        "web_view_link": "sample",
                        "title_extracted_from": "sample",
                        "classification_confidence": 0.0,
                        "type": "title_card",
                    }
                ]

                # Create table with sample data
                table = self.lancedb_client.create_table(TITLES_TABLE, sample_data)

                # Delete the sample record
                table.delete("id = 'sample'")

                logger.info(f"Created LanceDB table: {TITLES_TABLE}")

            return True

        except Exception as e:
            logger.error(f"Error ensuring LanceDB table: {e}")
            return False

    def index_title_cards_in_lancedb(
        self,
        title_cards_file: str = "title_cards.parquet",
        progress_callback=None,
        limit: Optional[int] = None,
        batch_size: int = 100,
    ) -> bool:
        """Index title cards in LanceDB with vector embeddings."""

        if self.lancedb_client is None:
            logger.error("LanceDB client not initialized")
            return False

        if not self.openai_client:
            logger.error("OpenAI client not initialized - cannot generate embeddings")
            return False

        title_cards_path = DATA_DIR / title_cards_file
        if not title_cards_path.exists():
            logger.error(f"Title cards file not found: {title_cards_path}")
            return False

        try:
            # Ensure table exists
            if not self._ensure_lancedb_table():
                logger.error("Failed to ensure LanceDB table exists")
                return False

            # Load title cards
            df_title_cards = pd.read_parquet(title_cards_path)

            if df_title_cards.empty:
                logger.warning("No title cards found for indexing")
                return False

            logger.info(f"Found {len(df_title_cards)} title cards to index")

            # Apply limit if specified
            if limit:
                df_title_cards = df_title_cards.head(limit)
                logger.info(f"Processing limited to {len(df_title_cards)} title cards")

            # Get LanceDB table
            table = self.lancedb_client.open_table(TITLES_TABLE)

            # Process title cards in batches
            processed_count = 0
            error_count = 0

            for i in range(0, len(df_title_cards), batch_size):
                batch = df_title_cards.iloc[i : i + batch_size]
                batch_records = []

                for _, row in batch.iterrows():
                    file_hash = row["file_hash"]

                    # Update progress
                    if progress_callback:
                        progress_callback(
                            {
                                "current_file": row.get("title", "Unknown"),
                                "processed": processed_count,
                                "total": len(df_title_cards),
                                "errors": error_count,
                            }
                        )

                    try:
                        # Combine text for embedding
                        text_parts = []
                        if row.get("title"):
                            text_parts.append(row["title"])
                        if row.get("author"):
                            text_parts.append(row["author"])
                        if row.get("summary"):
                            text_parts.append(row["summary"])

                        combined_text = " ".join(text_parts)

                        if not combined_text.strip():
                            logger.warning(f"Skipping empty text for {file_hash}")
                            error_count += 1
                            continue

                        # Generate embedding using OpenAI
                        embedding = self.openai_client.get_embedding(combined_text)

                        # Create record for LanceDB
                        record = {
                            "id": file_hash,
                            "text": combined_text,
                            "vector": embedding,
                            "file_hash": file_hash,
                            "title": row.get("title", ""),
                            "author": row.get("author", ""),
                            "file_name": row.get("file_name", ""),
                            "category": row.get("category", ""),
                            "subcategory": row.get("subcategory", ""),
                            "language": row.get("language", ""),
                            "mime_type": row.get("mime_type", ""),
                            "content_source": row.get("content_source", ""),
                            "summary_length": int(row.get("summary_length", 0)),
                            "phrase_count": int(row.get("phrase_count", 0)),
                            "file_size": int(row.get("file_size", 0)),
                            "path": row.get("path", ""),
                            "web_view_link": row.get("web_view_link", ""),
                            "title_extracted_from": row.get("title_extracted_from", ""),
                            "classification_confidence": float(
                                row.get("classification_confidence", 0.0)
                            ),
                            "type": "title_card",
                        }
                        batch_records.append(record)

                        processed_count += 1

                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing title card {file_hash}: {e}")

                # Add batch to LanceDB
                try:
                    if batch_records:
                        table.add(batch_records)
                        logger.info(f"Added {len(batch_records)} records to LanceDB")

                except Exception as e:
                    logger.error(f"Error adding batch to LanceDB: {e}")
                    error_count += len(batch_records)

            # Final progress update
            if progress_callback:
                progress_callback(
                    {
                        "current_file": "Indexing completed",
                        "processed": processed_count,
                        "total": len(df_title_cards),
                        "errors": error_count,
                        "phase": "completed",
                    }
                )

            logger.info(
                f"Indexing completed: {processed_count} processed, {error_count} errors"
            )
            return error_count == 0

        except Exception as e:
            logger.error(f"Error indexing title cards in LanceDB: {e}")
            return False

    def clear_lancedb_table(self) -> bool:
        """Clear and recreate LanceDB table."""
        if self.lancedb_client is None:
            logger.error("LanceDB client not initialized")
            return False

        try:
            # Drop existing table if it exists
            existing_tables = self.lancedb_client.table_names()
            if TITLES_TABLE in existing_tables:
                logger.info(f"Dropping existing table: {TITLES_TABLE}")
                self.lancedb_client.drop_table(TITLES_TABLE)

            # Recreate table
            logger.info("Recreating LanceDB table...")
            return self._ensure_lancedb_table()

        except Exception as e:
            logger.error(f"Error clearing LanceDB table: {e}")
            return False
