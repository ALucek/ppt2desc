<img src="ppt2desc_icon.png" width=250>

# ppt2desc

Convert PowerPoint presentations into semantically rich text using Vision Language Models.

## Overview

ppt2desc is a command-line tool that converts PowerPoint presentations into detailed textual descriptions. PowerPoint presentations are an inherently visual medium that often convey complex ideas through a combination of text, graphics, charts, and other visual layouts. This tool uses vision language models to not only transcribe the text content but also interpret and describe the visual elements and their relationships, capturing the full semantic meaning of each slide in a machine-readable format.

## Features

- Convert PPT/PPTX files to semantic descriptions
- Process individual files or entire directories
- Support for visual elements interpretation (charts, graphs, figures)
- Rate limiting for API calls
- Customizable prompts and instructions
- JSON output format for easy integration

**Current Model Provider Support**
- Gemini models via Google Gemini API
- GPT Models via OpenAI API
- Claude Models via Anthropic API
- Gemini Models via Google Cloud Platform Vertex AI
- GPT Models via Microsoft Azure AI Foundry Deployments
- Nova & Claude Models via Amazon Web Services's Amazon Bedrock

## Prerequisites

- Python 3.13 or higher
- [UV](https://github.com/astral-sh/uv) package manager (install from [uv.pm](https://github.com/astral-sh/uv#installation))
- LibreOffice (for PPT/PPTX to PDF conversion)
  - Option 1: Install LibreOffice locally.
  - Option 2: Use the provided Docker container for LibreOffice.
- vLLM API credentials

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ALucek/ppt2desc.git
cd ppt2desc
```

2. Installing LibreOffice

LibreOffice is a critical dependency for this tool as it handles the headless conversion of PowerPoint files to PDF format

**Option 1: Local Installation**

**Linux Systems:**
```bash
sudo apt install libreoffice
```

macOS:
```bash
brew install libreoffice
```  

Windows:  
Build from the installer at [LibreOffice's Official Website](https://www.libreoffice.org/download/download/)

**Option 2: Docker-based Installation**

a. Ensure you have [Docker](https://www.docker.com/) installed on your system  
b. Run the following command
```bash
docker compose up -d
```

This command will build the Docker image based on the provided [Dockerfile](./src/libreoffice_docker/) and start the container in detached mode. The LibreOffice conversion service will be accessible at`http://localhost:2002`. Take down with `docker compose down`.

3. Install dependencies using UV:
```bash
uv sync
```

This will create a virtual environment and install all dependencies from `pyproject.toml`.

## Usage

Basic usage with Gemini API:
```bash
uv run src/main.py \
    --input_dir /path/to/presentations \
    --output_dir /path/to/output \
    --libreoffice_path /path/to/soffice \
    --client gemini \
    --api_key YOUR_GEMINI_API_KEY
```

### Command Line Arguments

General Arguments:
- `--input_dir`: Path to input directory or PPT file (required)
- `--output_dir`: Output directory path (required)
- `--client`: LLM client to use: 'gemini', 'vertexai', 'anthropic', 'azure', 'aws' or 'openai' (required)
- `--model`: Model to use (default: "gemini-1.5-flash")
- `--instructions`: Additional instructions for the model
- `--libreoffice_path`: Path to LibreOffice installation
- `--libreoffice_url`: Url for docker-based libreoffice installation (configured: http://localhost:2002)
- `--rate_limit`: API calls per minute (default: 60)
- `--prompt_path`: Custom prompt file path
- `--api_key`: Model Provider API key (if not set via environment variable)
- `--save_pdf`: Include to save the converted PDF in your output folder
- `--save_images`: Include to save the individual slide images in your output folder

Vertex AI Specific Arguments:
- `--gcp_project_id`: GCP project ID for Vertex AI service account
- `--gcp_region`: GCP region for Vertex AI service (e.g., us-central1)
- `--gcp_application_credentials`: Path to GCP service account JSON credentials file

Azure AI Foundry Specific Arguments:
- `--azure_openai_api_key`: Azure AI Foundry Resource Key 1 or Key 2
- `--azure_openai_endpoint`: Azure AI Foundry deployment service endpoint link
- `--azure_deployment_name`: The name of your model deployment
- `--azure_api_version`: Azure API Version (Default: "2023-12-01-preview")

AWS Amazon Bedrock Specific Arguments:
- `--aws_access_key_id`: Bedrock Account Access Key
- `--aws_secret_access_key`: Bedrock Account Account Secret Access Key
- `--aws_region`: AWS Bedrock Region

### Example Commands

Using Gemini API:
```bash
uv run src/main.py \
    --input_dir ./presentations \
    --output_dir ./output \
    --libreoffice_path ./soffice \
    --client gemini \
    --model gemini-1.5-flash \
    --rate_limit 30 \
    --instructions "Focus on extracting numerical data from charts and graphs"
```

Using Vertex AI:
```bash
uv run src/main.py \
    --input_dir ./presentations \
    --output_dir ./output \
    --client vertexai \
    --libreoffice_path ./soffice \
    --gcp_project_id my-project-123 \
    --gcp_region us-central1 \
    --gcp_application_credentials ./service-account.json \
    --model gemini-1.5-pro \
    --instructions "Extract detailed information from technical diagrams"
```
Using Azure AI Foundry:
```bash
uv run src/main.py \
    --input_dir ./presentations \
    --output_dir ./output \
    --libreoffice_path ./soffice \
    --client azure \
    --azure_openai_api_key 123456790ABCDEFG \
    --azure_openai_endpoint 'https://example-endpoint-001.openai.azure.com/' \
    --azure_deployment_name gpt-4o \
    --azure_api_version 2023-12-01-preview \
    --rate_limit 60
```

Using AWS Amazon Bedrock:
```bash
uv run src/main.py \
    --input_dir ./presentations \
    --output_dir ./output \
    --libreoffice_path ./soffice \
    --client aws \
    --model us.amazon.nova-lite-v1:0 \
    --aws_access_key_id 123456790ABCDEFG \
    --aws_secret_access_key 123456790ABCDEFG \
    --aws_region us-east-1 \
    --rate_limit 60
```

## Output Format

The tool generates JSON files with the following structure:

```json
{
  "deck": "presentation.pptx",
  "model": "model-name",
  "slides": [
    {
      "number": 1,
      "content": "Detailed description of slide content..."
    },
    // ... more slides
  ]
}
```

## Advanced Usage

### Using Docker-based LibreOffice Conversion

When using the Docker container for LibreOffice, you can use the `--libreoffice_url` argument to direct the conversion process to the container's API endpoint, rather than a local installation.

```bash
uv run src/main.py \
    --input_dir ./presentations \
    --output_dir ./output \
    --libreoffice_url http://localhost:2002 \
    --client vertexai \
    --model gemini-1.5-pro \
    --gcp_project_id my-project-123 \
    --gcp_region us-central1 \
    --gcp_application_credentials ./service-account.json \
    --rate_limit 30 \
    --instructions "Extract detailed information from technical diagrams" \
    --save_pdf \
    --save_images
```

You should use either `--libreoffice_url` or `--libreoffice_path` but not both.

### Custom Prompts

You can modify the base prompt by editing `src/prompt.py` (specifically the `BASE_PROMPT` constant) or providing additional instructions via the command line:

```bash
uv run src/main.py \
    --input_dir ./presentations \
    --output_dir ./output \
    --libreoffice_path ./soffice \
    --instructions "Include mathematical equations and formulas in LaTeX format"
```

### Authentication

For Consumer APIs:
- Set your API key via the `--api_key` argument or through your respective provider's environment variables

For Vertex AI:
1. Create a service account in your GCP project IAM
2. Grant necessary permissions (typically, "Vertex AI User" role)
3. Download the service account JSON key file
4. Provide the credentials file path via `--gcp_application_credentials`

For Azure OpenAI Foundry:
1. Create an Azure OpenAI Resource
2. Navigate to Azure AI Foundry and choose the subscription and Azure OpenAI Resource to work with
3. Under management select deployments
4. Select create new deployment and configure with your vision LLM
5. Provide deployment name, API key, endpoint, and api version via `--azure_deployment_name`, `--azure_openai_api_key`, `--azure_openai_endpoint`, `--azure_api_version`,

For AWS Bedrock:
1. Request access to serverless model deployments in Amazon Bedrock's model catalog
2. Create a user in your AWS IAM
3. Enable Amazon Bedrock access policies for your user
4. Save User Credentials access key and secret access key
5. Provide user's credentials via `--aws_access_key_id`, and `--aws_secret_access_key`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Todo**
- Handling google's new genai SDK for a unified gemini/vertex experience
- Better Docker Setup
- AWS Llama Vision Support Confirmation
- Combination of JSON files across multiple ppts
- Dynamic font understanding (i.e. struggles when font that ppt is using is not installed on machine)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LibreOffice](https://www.libreoffice.org/) for PPT/PPTX conversion
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for PDF processing