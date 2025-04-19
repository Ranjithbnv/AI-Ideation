#Run using below command
# python upload_to_blob.py \
#   --account-name your_account_name \
#   --account-key your_account_key \
#   --container-name your_container_name \
#   --folder Resumes



import os
import click
from azure.storage.blob import BlobServiceClient, BlobClient
from pathlib import Path
from datetime import datetime, UTC  # Python 3.12+

@click.command()
@click.option('--account-name', required=True, help='Azure Storage Account Name')
@click.option('--account-key', required=True, help='Azure Storage Account Key')
@click.option('--container-name', required=True, help='Azure Blob Container Name')
@click.option('--folder', default='Resumes', help='Local folder to upload from (default: Resumes)')
def upload_files(account_name, account_key, container_name, folder):
    """Uploads files from a local folder to Azure Blob Storage with metadata."""
    
    # Construct connection string
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix=core.windows.net"
    )

    # Create the BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Path to the local folder
    local_folder_path = Path(folder)

    if not local_folder_path.exists():
        click.echo(f"Folder '{folder}' does not exist.")
        return

    # Loop through files and upload
    for file_path in local_folder_path.glob("*.*"):
        if file_path.is_file():
            blob_name = f"{local_folder_path.name}/{file_path.name}"

            # UTC timestamp without colons
           

            metadata = {
                "filename": file_path.name,
                "uploaded": datetime.now(UTC).isoformat(),
                "DocType" : "docx",
                "DocCategory" : "Resume",
            }

            try:
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True, metadata=metadata)

                click.echo(f" Uploaded: '{file_path}' → '{blob_name}'")

            except Exception as e:
                click.echo(f" Error uploading '{file_path}': {e}")

if __name__ == '__main__':
    upload_files()
