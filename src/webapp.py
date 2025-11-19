from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile
import shutil
import logging
import sys
from typing import Optional
import json

from llm.google_unified import GoogleUnifiedClient
from llm.openai import OpenAIClient
from llm.anthropic import AnthropicClient
from llm.azure import AzureClient
from llm.aws import AWSClient
from processor import process_input_path
from prompt import BASE_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPT2Desc Web Service")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PPT to Description Converter</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            padding: 40px;
        }

        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }

        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }

        input[type="text"],
        input[type="password"],
        input[type="number"],
        select,
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        input[type="text"]:focus,
        input[type="password"]:focus,
        input[type="number"]:focus,
        select:focus,
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px dashed #e1e8ed;
            border-radius: 6px;
            cursor: pointer;
            background: #f8f9fa;
        }

        input[type="file"]:hover {
            border-color: #667eea;
            background: #f0f2ff;
        }

        .checkbox-group {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }

        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        button:active {
            transform: translateY(0);
        }

        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        #status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 6px;
            display: none;
        }

        #status.info {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            color: #1976d2;
        }

        #status.success {
            background: #e8f5e9;
            border: 1px solid #4caf50;
            color: #2e7d32;
        }

        #status.error {
            background: #ffebee;
            border: 1px solid #f44336;
            color: #c62828;
        }

        #result {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e1e8ed;
            display: none;
        }

        #result h3 {
            margin-bottom: 15px;
            color: #333;
        }

        .slide-content {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }

        .slide-number {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
        }

        .slide-text {
            color: #555;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .client-specific {
            display: none;
            padding: 15px;
            background: #f0f2ff;
            border-radius: 6px;
            margin-top: 10px;
        }

        .client-specific.active {
            display: block;
        }

        small {
            color: #666;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 PPT to Description</h1>
        <p class="subtitle">Convert PowerPoint presentations into semantic descriptions using AI</p>

        <form id="uploadForm">
            <div class="form-group">
                <label for="file">PowerPoint File (.ppt or .pptx) *</label>
                <input type="file" id="file" name="file" accept=".ppt,.pptx" required>
            </div>

            <div class="form-group">
                <label for="client">AI Model Provider *</label>
                <select id="client" name="client" required onchange="updateClientFields()">
                    <option value="">Select a provider...</option>
                    <option value="gemini">Google Gemini API</option>
                    <option value="vertexai">Google Vertex AI</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic Claude</option>
                    <option value="azure">Azure OpenAI</option>
                    <option value="aws">AWS Bedrock</option>
                </select>
            </div>

            <div class="form-group">
                <label for="model">Model Name</label>
                <input type="text" id="model" name="model" placeholder="e.g., gemini-2.5-flash, gpt-4o, claude-3-5-sonnet-20241022">
                <small>Leave blank to use default model for selected provider</small>
            </div>

            <!-- Generic API Key -->
            <div class="form-group" id="api-key-group">
                <label for="api_key">API Key</label>
                <input type="password" id="api_key" name="api_key" placeholder="Your API key">
                <small>Required for gemini, openai, and anthropic providers</small>
            </div>

            <!-- Vertex AI specific -->
            <div class="client-specific" id="vertexai-fields">
                <div class="form-group">
                    <label for="gcp_project_id">GCP Project ID *</label>
                    <input type="text" id="gcp_project_id" name="gcp_project_id">
                </div>
                <div class="form-group">
                    <label for="gcp_region">GCP Region</label>
                    <input type="text" id="gcp_region" name="gcp_region" placeholder="e.g., us-central1">
                </div>
                <div class="form-group">
                    <label for="gcp_application_credentials">GCP Credentials Path *</label>
                    <input type="text" id="gcp_application_credentials" name="gcp_application_credentials" placeholder="/path/to/service-account.json">
                </div>
            </div>

            <!-- Azure specific -->
            <div class="client-specific" id="azure-fields">
                <div class="form-group">
                    <label for="azure_openai_api_key">Azure OpenAI API Key *</label>
                    <input type="password" id="azure_openai_api_key" name="azure_openai_api_key">
                </div>
                <div class="form-group">
                    <label for="azure_openai_endpoint">Azure OpenAI Endpoint *</label>
                    <input type="text" id="azure_openai_endpoint" name="azure_openai_endpoint" placeholder="https://example.openai.azure.com/">
                </div>
                <div class="form-group">
                    <label for="azure_deployment_name">Deployment Name *</label>
                    <input type="text" id="azure_deployment_name" name="azure_deployment_name">
                </div>
                <div class="form-group">
                    <label for="azure_api_version">API Version</label>
                    <input type="text" id="azure_api_version" name="azure_api_version" value="2023-12-01-preview">
                </div>
            </div>

            <!-- AWS specific -->
            <div class="client-specific" id="aws-fields">
                <div class="form-group">
                    <label for="aws_access_key_id">AWS Access Key ID *</label>
                    <input type="password" id="aws_access_key_id" name="aws_access_key_id">
                </div>
                <div class="form-group">
                    <label for="aws_secret_access_key">AWS Secret Access Key *</label>
                    <input type="password" id="aws_secret_access_key" name="aws_secret_access_key">
                </div>
                <div class="form-group">
                    <label for="aws_region">AWS Region</label>
                    <input type="text" id="aws_region" name="aws_region" value="us-east-1">
                </div>
            </div>

            <div class="form-group">
                <label for="instructions">Additional Instructions (Optional)</label>
                <textarea id="instructions" name="instructions" rows="3" placeholder="E.g., Focus on extracting numerical data from charts"></textarea>
            </div>

            <div class="form-group">
                <label for="libreoffice_url">LibreOffice URL (Optional)</label>
                <input type="text" id="libreoffice_url" name="libreoffice_url" value="http://libreoffice-converter:2002" placeholder="http://localhost:2002">
                <small>Use Docker-based LibreOffice converter. Leave blank to use local installation.</small>
            </div>

            <div class="form-group">
                <label for="rate_limit">Rate Limit (API calls per minute)</label>
                <input type="number" id="rate_limit" name="rate_limit" value="60" min="1">
            </div>

            <div class="form-group">
                <div class="checkbox-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="save_pdf" name="save_pdf">
                        <span>Save PDF</span>
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" id="save_images" name="save_images">
                        <span>Save Images</span>
                    </label>
                </div>
            </div>

            <button type="submit" id="submitBtn">
                <span id="btnText">Convert Presentation</span>
            </button>
        </form>

        <div id="status"></div>
        <div id="result"></div>
    </div>

    <script>
        function updateClientFields() {
            const client = document.getElementById('client').value;

            // Hide all client-specific fields
            document.querySelectorAll('.client-specific').forEach(el => {
                el.classList.remove('active');
            });

            // Show relevant fields
            if (client === 'vertexai') {
                document.getElementById('vertexai-fields').classList.add('active');
            } else if (client === 'azure') {
                document.getElementById('azure-fields').classList.add('active');
            } else if (client === 'aws') {
                document.getElementById('aws-fields').classList.add('active');
            }
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = type;
            status.style.display = 'block';
        }

        function hideStatus() {
            document.getElementById('status').style.display = 'none';
        }

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const resultDiv = document.getElementById('result');

            // Disable button and show loading
            submitBtn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> Processing...';
            resultDiv.style.display = 'none';
            hideStatus();

            const formData = new FormData(e.target);

            try {
                showStatus('Uploading and processing presentation...', 'info');

                const response = await fetch('/convert', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Conversion failed');
                }

                const result = await response.json();

                // Display results
                showStatus('Conversion completed successfully!', 'success');
                displayResults(result);

            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
            } finally {
                submitBtn.disabled = false;
                btnText.textContent = 'Convert Presentation';
            }
        });

        function displayResults(data) {
            const resultDiv = document.getElementById('result');

            let html = `<h3>Results for: ${data.deck}</h3>`;
            html += `<p><strong>Model:</strong> ${data.model}</p>`;
            html += `<p><strong>Total Slides:</strong> ${data.slides.length}</p>`;

            data.slides.forEach(slide => {
                html += `
                    <div class="slide-content">
                        <div class="slide-number">Slide ${slide.number}</div>
                        <div class="slide-text">${slide.content}</div>
                    </div>
                `;
            });

            resultDiv.innerHTML = html;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web interface"""
    return HTML_TEMPLATE


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/convert")
async def convert_presentation(
    file: UploadFile = File(...),
    client: str = Form(...),
    model: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    instructions: Optional[str] = Form(None),
    libreoffice_url: Optional[str] = Form(None),
    rate_limit: int = Form(60),
    save_pdf: bool = Form(False),
    save_images: bool = Form(False),
    # Vertex AI fields
    gcp_project_id: Optional[str] = Form(None),
    gcp_region: Optional[str] = Form(None),
    gcp_application_credentials: Optional[str] = Form(None),
    # Azure fields
    azure_openai_api_key: Optional[str] = Form(None),
    azure_openai_endpoint: Optional[str] = Form(None),
    azure_deployment_name: Optional[str] = Form(None),
    azure_api_version: Optional[str] = Form("2023-12-01-preview"),
    # AWS fields
    aws_access_key_id: Optional[str] = Form(None),
    aws_secret_access_key: Optional[str] = Form(None),
    aws_region: Optional[str] = Form("us-east-1"),
):
    """
    Convert a PowerPoint presentation to semantic descriptions
    """
    if not file.filename or not file.filename.lower().endswith(('.pptx', '.ppt')):
        raise HTTPException(status_code=400, detail="File must be a .pptx or .ppt")

    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    try:
        temp_path = Path(temp_dir)
        output_path = Path(output_dir)
        input_file = temp_path / file.filename

        # Save uploaded file
        with input_file.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # Build prompt
        prompt = BASE_PROMPT
        if instructions and instructions.strip():
            prompt = f"{BASE_PROMPT}\n\nAdditional instructions:\n{instructions}"

        # Set default model based on client if not provided
        if not model or model.strip() == "":
            model_defaults = {
                "gemini": "gemini-2.5-flash",
                "vertexai": "gemini-2.5-flash",
                "openai": "gpt-4o",
                "anthropic": "claude-3-5-sonnet-20241022",
                "azure": "gpt-4o",
                "aws": "us.amazon.nova-lite-v1:0"
            }
            model = model_defaults.get(client, "gemini-2.5-flash")

        # Initialize model instance
        try:
            if client == "gemini":
                model_instance = GoogleUnifiedClient(
                    api_key=api_key,
                    model=model,
                    use_vertex=False
                )
            elif client == "vertexai":
                if not gcp_project_id or not gcp_application_credentials:
                    raise HTTPException(
                        status_code=400,
                        detail="GCP project_id and application_credentials are required for Vertex AI"
                    )
                model_instance = GoogleUnifiedClient(
                    credentials_path=gcp_application_credentials,
                    project_id=gcp_project_id,
                    region=gcp_region,
                    model=model,
                    use_vertex=True
                )
            elif client == "openai":
                model_instance = OpenAIClient(api_key=api_key, model=model)
            elif client == "anthropic":
                model_instance = AnthropicClient(api_key=api_key, model=model)
            elif client == "azure":
                if not azure_openai_api_key or not azure_openai_endpoint or not azure_deployment_name:
                    raise HTTPException(
                        status_code=400,
                        detail="Azure API key, endpoint, and deployment name are required"
                    )
                model_instance = AzureClient(
                    api_key=azure_openai_api_key,
                    endpoint=azure_openai_endpoint,
                    deployment=azure_deployment_name,
                    api_version=azure_api_version
                )
            elif client == "aws":
                if not aws_access_key_id or not aws_secret_access_key:
                    raise HTTPException(
                        status_code=400,
                        detail="AWS access key ID and secret access key are required"
                    )
                model_instance = AWSClient(
                    access_key_id=aws_access_key_id,
                    secret_access_key=aws_secret_access_key,
                    region=aws_region,
                    model=model
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported client: {client}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

        # Determine LibreOffice configuration
        if libreoffice_url and libreoffice_url.strip():
            libreoffice_endpoint = libreoffice_url
            libreoffice_path = None
        else:
            # Try to find local LibreOffice
            libreoffice_binary = shutil.which("soffice") or shutil.which("libreoffice")
            if libreoffice_binary:
                libreoffice_path = Path(libreoffice_binary)
                libreoffice_endpoint = None
            else:
                raise HTTPException(
                    status_code=500,
                    detail="LibreOffice not found. Please provide --libreoffice_url or install LibreOffice locally"
                )

        # Process the presentation
        logger.info(f"Processing {file.filename} with {client} model {model}")
        results = process_input_path(
            input_path=input_file,
            output_dir=output_path,
            libreoffice_path=libreoffice_path,
            libreoffice_endpoint=libreoffice_endpoint,
            model_instance=model_instance,
            rate_limit=rate_limit,
            prompt=prompt,
            save_pdf=save_pdf,
            save_images=save_images,
            max_workers=None
        )

        if not results or len(results) == 0:
            raise HTTPException(status_code=500, detail="Processing failed - no results returned")

        # Get the first (and should be only) result
        ppt_file, slides = results[0]

        if len(slides) == 0:
            raise HTTPException(status_code=500, detail="Processing failed - no slides extracted")

        # Format response
        response_data = {
            "deck": file.filename,
            "model": model,
            "slides": [
                {
                    "number": i + 1,
                    "content": slide
                }
                for i, slide in enumerate(slides)
            ]
        }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        # Cleanup temporary directories
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
