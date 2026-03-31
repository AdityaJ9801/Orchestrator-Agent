import os
import logging
from typing import Optional
from azure.storage.blob import BlobServiceClient
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class StorageProvider:
    def __init__(self):
        self.connection_string = settings.azure_storage_connection_string
        self.container_name = settings.azure_storage_container_name
        self.use_azure = bool(self.connection_string)
        
        if self.use_azure:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                self.container_client = self.blob_service_client.get_container_client(self.container_name)
                logger.info("Azure Blob Storage initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Blob Storage: {e}")
                self.use_azure = False

    async def save_file(self, content: bytes, filename: str) -> str:
        """Saves file and returns its path or URL."""
        if self.use_azure:
            blob_client = self.container_client.get_blob_client(filename)
            blob_client.upload_blob(content, overwrite=True)
            return blob_client.url
        else:
            # Fallback to local
            upload_dir = os.path.join(os.getcwd(), "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            with open(file_path, "wb") as f:
                f.write(content)
            return file_path

    async def delete_file(self, filename: str):
        if self.use_azure:
            blob_client = self.container_client.get_blob_client(filename)
            blob_client.delete_blob()
        else:
            file_path = os.path.join(os.getcwd(), "uploads", filename)
            if os.path.exists(file_path):
                os.remove(file_path)

storage = StorageProvider()
