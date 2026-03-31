import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class StorageProvider:
    def __init__(self):
        self.connection_string = settings.azure_storage_connection_string
        self.container_name = settings.azure_storage_container_name
        self.use_azure = bool(self.connection_string)
        self._account_name: str = ""
        self._account_key: str = ""
        
        if self.use_azure:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                self.container_client = self.blob_service_client.get_container_client(self.container_name)
                # Extract account name and key from connection string for SAS generation
                for part in self.connection_string.split(";"):
                    if part.startswith("AccountName="):
                        self._account_name = part.split("=", 1)[1]
                    elif part.startswith("AccountKey="):
                        self._account_key = part.split("=", 1)[1]
                logger.info("Azure Blob Storage initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Blob Storage: {e}")
                self.use_azure = False

    def generate_sas_url(self, blob_name: str, expiry_hours: int = 2) -> str:
        """Generate a time-limited SAS URL for a private blob."""
        sas_token = generate_blob_sas(
            account_name=self._account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self._account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(tz=timezone.utc) + timedelta(hours=expiry_hours),
        )
        return f"https://{self._account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"

    async def save_file(self, content: bytes, filename: str) -> str:
        """Saves file and returns a publicly accessible SAS URL (Azure) or local path."""
        if self.use_azure:
            blob_client = self.container_client.get_blob_client(filename)
            blob_client.upload_blob(content, overwrite=True)
            # Return SAS URL so Context Agent can access it without auth headers
            if self._account_key:
                return self.generate_sas_url(filename)
            return blob_client.url  # fallback if key not parsed
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

