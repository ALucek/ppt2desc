import requests
from pathlib import Path
import logging

from .exceptions import ConversionError

logger = logging.getLogger(__name__)

def convert_pptx_via_docker(
    ppt_file: Path,
    container_url: str,
    temp_dir: Path
) -> Path:
    """
    Convert a PPT/PPTX file to PDF by sending it to the Docker container at container_url.
    e.g., container_url="http://localhost:2002"
    
    :param ppt_file: Path to the local PPT/PPTX file
    :param container_url: Base URL of the container (without trailing slash)
    :param temp_dir: Directory to store the resulting PDF
    :return: Path to the newly-created PDF file
    :raises ConversionError: if the container fails or file can't be saved
    """
    endpoint = f"{container_url.rstrip('/')}/convert/ppt-to-pdf"
    logger.info(f"Calling Docker LibreOffice at {endpoint} for {ppt_file}")

    try:
        # 1) Prepare the file for upload (using context manager to ensure file is closed)
        with ppt_file.open("rb") as file_handle:
            files = {
                "file": (ppt_file.name, file_handle, "application/vnd.ms-powerpoint")
            }

            # 2) Make a POST request (file will be closed automatically after request completes)
            resp = requests.post(endpoint, files=files, timeout=300)
            resp.raise_for_status()

        # 3) Save the returned PDF to temp_dir (file handle is now closed)
        pdf_filename = ppt_file.stem + ".pdf"
        pdf_path = temp_dir / pdf_filename
        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        if not pdf_path.exists():
            raise ConversionError("PDF file not created after Docker-based conversion.")
        logger.info(f"Created PDF {pdf_path} via Docker container.")
        return pdf_path

    except Exception as e:
        logger.error(f"Error converting {ppt_file} via Docker: {e}")
        raise ConversionError(f"Error converting {ppt_file}: {str(e)}")
