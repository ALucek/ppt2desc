# Running PPT2Desc on Localhost

This guide explains how to run the PPT2Desc web application on your local machine.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

This is the easiest way to get started. Both LibreOffice converter and the web application will run in containers.

1. **Start the services:**
   ```bash
   docker compose up -d
   ```

2. **Access the web interface:**
   - Open your browser and navigate to: **http://localhost:5001**

3. **Stop the services:**
   ```bash
   docker compose down
   ```

That's it! The web interface will be available at http://localhost:5001, and you can upload PowerPoint files directly through your browser.

### Option 2: Running Locally with UV

If you prefer to run the application directly on your machine without Docker:

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Start the LibreOffice converter (optional, if you want to use Docker-based conversion):**
   ```bash
   docker compose up -d libreoffice-converter
   ```

3. **Run the web application:**
   ```bash
   uv run uvicorn src.webapp:app --host 0.0.0.0 --port 5001
   ```

4. **Access the web interface:**
   - Open your browser and navigate to: **http://localhost:5001**

## Using the Web Interface

Once the application is running, you can:

1. **Upload a PowerPoint file** (.ppt or .pptx)
2. **Select an AI provider** (Gemini, OpenAI, Anthropic, etc.)
3. **Configure model settings** (API keys, model name, etc.)
4. **Add optional instructions** to customize the output
5. **Click "Convert Presentation"** to process your file

The results will be displayed directly in the browser, showing detailed descriptions for each slide.

The web service runs on **port 5001** by default.

## Configuration Options

### AI Provider Settings

The web interface supports multiple AI providers:

- **Google Gemini API**: Requires API key
- **Google Vertex AI**: Requires GCP project ID, region, and service account credentials
- **OpenAI**: Requires API key
- **Anthropic Claude**: Requires API key
- **Azure OpenAI**: Requires API key, endpoint, and deployment name
- **AWS Bedrock**: Requires access key ID, secret access key, and region

### LibreOffice Configuration

By default, the web application uses the Docker-based LibreOffice converter at `http://libreoffice-converter:2002` (when using Docker Compose) or `http://localhost:2002` (when running locally).

If you have LibreOffice installed locally, you can leave the LibreOffice URL field blank, and the application will attempt to find it in your system PATH.

## API Endpoints

If you want to integrate the service programmatically:

### Health Check
```bash
curl http://localhost:5001/health
```

### Convert Presentation
```bash
curl -X POST http://localhost:5001/convert \
  -F "file=@presentation.pptx" \
  -F "client=gemini" \
  -F "api_key=YOUR_API_KEY" \
  -F "model=gemini-2.5-flash"
```

## Troubleshooting

### Port Already in Use

If port 5001 is already in use, you can change it:

**Docker Compose:**
Edit `docker-compose.yml` and change the port mapping:
```yaml
ports:
  - "5002:8000"  # Change 5002 to any available port
```

**Local Running:**
```bash
uv run uvicorn src.webapp:app --host 0.0.0.0 --port 5002
```

### LibreOffice Connection Issues

If you get errors about LibreOffice conversion:

1. Make sure the LibreOffice converter is running:
   ```bash
   docker compose ps
   ```

2. Check the health of the converter:
   ```bash
   curl http://localhost:2002/health
   ```

3. If using local LibreOffice, ensure it's installed:
   ```bash
   which soffice
   # or
   which libreoffice
   ```

### Memory Issues

For large presentations or high rate limits, you may need to increase Docker memory limits. Edit your Docker settings or add resource limits to `docker-compose.yml`.

## Development

To run in development mode with auto-reload:

```bash
uv run uvicorn src.webapp:app --host 0.0.0.0 --port 5001 --reload
```

## Environment Variables

You can set default values using environment variables:

```bash
export GEMINI_API_KEY=your_api_key
export OPENAI_API_KEY=your_api_key
export ANTHROPIC_API_KEY=your_api_key
```

Then you won't need to enter API keys in the web interface each time.

## Next Steps

- Check the main [README.md](README.md) for detailed information about the project
- Learn about customizing prompts and instructions
- Explore the CLI version for batch processing
