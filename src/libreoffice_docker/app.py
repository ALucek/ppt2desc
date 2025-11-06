from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import Response
import subprocess
from pathlib import Path
import tempfile
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Conversion Service")
LIBREOFFICE_PATH = Path("/usr/bin/libreoffice")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/convert/ppt-to-pdf")
async def convert_pptx_to_pdf(file: UploadFile):
    """Convert uploaded PPTX file to PDF"""
    if not file.filename or not file.filename.lower().endswith(('.pptx', '.ppt')):
        raise HTTPException(status_code=400, detail="File must be a .pptx or .ppt")

    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = Path(temp_dir)
        input_path = temp_path / file.filename
        pdf_path = temp_path / f"{Path(file.filename).stem}.pdf"

        # Save uploaded file
        with input_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert to PDF
        result = subprocess.run(
            [
                str(LIBREOFFICE_PATH),
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', str(temp_path),
                str(input_path)
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 or not pdf_path.exists():
            logger.error(f"Conversion failed: {result.stderr}")
            raise HTTPException(status_code=500, detail="PDF conversion failed")

        # Read file content before cleanup
        pdf_content = pdf_path.read_bytes()
        
        return Response(
            content=pdf_content,
            media_type='application/pdf',
            headers={"Content-Disposition": f'attachment; filename="{pdf_path.name}"'}
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
