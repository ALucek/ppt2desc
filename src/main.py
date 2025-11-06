import logging
import argparse
import sys
import shutil
from pathlib import Path

from llm.google_unified import GoogleUnifiedClient
from llm.openai import OpenAIClient
from llm.anthropic import AnthropicClient
from llm.azure import AzureClient
from llm.aws import AWSClient

from processor import process_input_path
from prompt import BASE_PROMPT

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Process PPT/PPTX files via vLLM.")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input directory or PPT file"
    )
    parser.add_argument(
        "--client",
        type=str,
        required=True,
        choices=["gemini", "vertexai", "openai", "anthropic", "azure", "aws"],
        help="LLM client to use: 'gemini', 'vertexai', 'openai', 'azure', 'aws', or 'anthropic'"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Suggested models: gemini-1.5-flash, gemini-1.5-pro, gpt-4o, claude-3-5-sonnet-latest"
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="None Provided",
        help="Additional instructions appended to the base prompt"
    )
    parser.add_argument(
        "--libreoffice_path",
        type=str,
        default=None,
        help="Path to the local installation of LibreOffice."
    )
    parser.add_argument(
        "--rate_limit",
        type=int,
        default=60,
        help="Number of API calls allowed per minute (default: 60)",
        metavar="RATE"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of concurrent workers for processing slides. Defaults to rate_limit if set, otherwise 10."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Optional path to a custom prompt file. If not provided, uses the default prompt from src/prompt.py"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the LLM. If not provided, the environment variable may be used."
    )
    parser.add_argument(
        "--gcp_region",
        type=str,
        default=None,
        help="GCP Region for connecting to vertex AI service account."
    )
    parser.add_argument(
        "--gcp_project_id",
        type=str,
        default=None,
        help="GCP project id for connecting to vertex AI service account."
    )
    parser.add_argument(
        "--gcp_application_credentials",
        type=str,
        default=None,
        help="Path to JSON credentials for GCP service account"
    )
    parser.add_argument(
        "--azure_openai_api_key",
        type=str,
        default=None,
        help="Value for AZURE_OPENAI_KEY if not set in env"
    )
    parser.add_argument(
        "--azure_openai_endpoint",
        type=str,
        default=None,
        help="Value for AZURE_OPENAI_ENDPOINT if not set in env"
    )
    parser.add_argument(
        "--azure_deployment_name",
        type=str,
        default=None,
        help="Name of your Azure deployment"
    )
    parser.add_argument(
        "--azure_api_version",
        type=str,
        default="2023-12-01-preview",
        help="Azure API version"
    )
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        help="AWS User Access Key"
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        help="AWS User Secret Access Key"
    )
    parser.add_argument(
        "--aws_region",
        type=str,
        default="us-east-1",
        help="Region for AWS Bedrock Instance"
    )
    parser.add_argument(
        "--save_pdf",
        action='store_true',
        default=False,
        help="Save converted PDF files in the output directory"
    )
    parser.add_argument(
        "--save_images",
        action='store_true',
        default=False,
        help="Save extracted images in a subfolder within the output directory named after the presentation"
    )
    parser.add_argument(
        "--libreoffice_url",
        type=str,
        default=None,
        help="If provided, uses the Docker container's endpoint (e.g., http://localhost:2002) for PPT->PDF conversion."
    )

    args = parser.parse_args(input_args)
    return args

def validate_args(args, logger):
    """
    Validate arguments based on client type and other constraints.
    """
    # Validate rate_limit
    if args.rate_limit <= 0:
        logger.error(f"rate_limit must be positive, got: {args.rate_limit}")
        sys.exit(1)
    
    # Validate max_workers if provided
    if args.max_workers is not None and args.max_workers <= 0:
        logger.error(f"max_workers must be positive, got: {args.max_workers}")
        sys.exit(1)
    
    # Validate client-specific required arguments
    if args.client == "vertexai":
        if not args.gcp_project_id:
            logger.error("--gcp_project_id is required when using 'vertexai' client")
            sys.exit(1)
        if not args.gcp_application_credentials:
            logger.error("--gcp_application_credentials is required when using 'vertexai' client")
            sys.exit(1)
    
    elif args.client == "azure":
        if not args.azure_openai_api_key:
            logger.error("--azure_openai_api_key is required when using 'azure' client")
            sys.exit(1)
        if not args.azure_openai_endpoint:
            logger.error("--azure_openai_endpoint is required when using 'azure' client")
            sys.exit(1)
        if not args.azure_deployment_name:
            logger.error("--azure_deployment_name is required when using 'azure' client")
            sys.exit(1)
    
    elif args.client == "aws":
        if not args.aws_access_key_id:
            logger.error("--aws_access_key_id is required when using 'aws' client")
            sys.exit(1)
        if not args.aws_secret_access_key:
            logger.error("--aws_secret_access_key is required when using 'aws' client")
            sys.exit(1)
    
    # Validate input path exists
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Validate output directory can be created (will be created if needed)
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        sys.exit(1)


def main():
    # ---- 1) Parse arguments ----
    args = parse_args()

    # ---- 2) Configure logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # ---- 2.5) Validate arguments ----
    validate_args(args, logger)

    # ---- 3) Read prompt once ----
    if args.prompt_path:
        # Use custom prompt file if provided
        base_prompt_file = Path(args.prompt_path)
        if not base_prompt_file.is_file():
            logger.error(f"Prompt file not found at {base_prompt_file}")
            sys.exit(1)
        base_prompt = base_prompt_file.read_text(encoding="utf-8").strip()
    else:
        # Use default prompt from Python module
        base_prompt = BASE_PROMPT
    if args.instructions and args.instructions.lower() != "none provided":
        prompt = f"{base_prompt}\n\nAdditional instructions:\n{args.instructions}"
    else:
        prompt = base_prompt

    # ---- 4) Initialize model instance ----
    try:
        if args.client == "gemini":
            # Using the new unified client for Gemini
            model_instance = GoogleUnifiedClient(
                api_key=args.api_key, 
                model=args.model,
                use_vertex=False
            )
            logger.info(f"Initialized Google GenAI Client (Gemini API) with model: {args.model}")
        elif args.client == "vertexai":
            # Using the new unified client for Vertex AI
            model_instance = GoogleUnifiedClient(
                credentials_path=args.gcp_application_credentials,
                project_id=args.gcp_project_id,
                region=args.gcp_region,
                model=args.model,
                use_vertex=True
            )
            logger.info(f"Initialized Google GenAI Client (Vertex AI) for project: {args.gcp_project_id}")
        elif args.client == "openai":
            model_instance = OpenAIClient(api_key=args.api_key, model=args.model)
            logger.info(f"Initialized OpenAIClient with model: {args.model}")
        elif args.client == "anthropic":
            model_instance = AnthropicClient(api_key=args.api_key, model=args.model)
            logger.info(f"Initialized AnthropicClient with model: {args.model}")
        elif args.client == "azure":
            model_instance = AzureClient(
                api_key=args.azure_openai_api_key,
                endpoint=args.azure_openai_endpoint,
                deployment=args.azure_deployment_name,
                api_version=args.azure_api_version
            )
            logger.info(f"Initialized AzureClient for deployment: {args.azure_deployment_name}")
        elif args.client == "aws":
            model_instance = AWSClient(
                access_key_id=args.aws_access_key_id,
                secret_access_key=args.aws_secret_access_key,
                region=args.aws_region,
                model=args.model
            )
            logger.info(f"Initialized AWSClient in region: {args.aws_region} with model {args.model}")
        else:
            logger.error(f"Unsupported client specified: {args.client}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    # Output directory already created in validate_args

    # ---- 5) Identify local vs. container-based conversion ----
    if args.libreoffice_url:
        logger.info(f"Using Docker-based LibreOffice at: {args.libreoffice_url}")
        libreoffice_endpoint = args.libreoffice_url
        # We'll pass this URL into the processor so it knows to do remote conversion
        libreoffice_path = None
    else:
        # If no URL is provided, use local LibreOffice
        if args.libreoffice_path:
            libreoffice_path = Path(args.libreoffice_path)
            if not libreoffice_path.exists():
                logger.error(f"LibreOffice not found at specified path: {libreoffice_path}")
                sys.exit(1)
            if not libreoffice_path.is_file():
                logger.error(f"LibreOffice path is not a file: {libreoffice_path}")
                sys.exit(1)
        else:
            # Try to find LibreOffice in PATH
            # Check for 'soffice' first (the actual executable), then 'libreoffice' (common alias)
            libreoffice_binary = shutil.which("soffice") or shutil.which("libreoffice")
            if libreoffice_binary:
                libreoffice_path = Path(libreoffice_binary)
                if not libreoffice_path.exists():
                    logger.error(f"LibreOffice binary resolved but does not exist: {libreoffice_path}")
                    sys.exit(1)
                logger.info(f"Found LibreOffice in PATH: {libreoffice_path}")
            else:
                logger.error(
                    "LibreOffice not found. Please either:\n"
                    "  1) Install LibreOffice and ensure 'soffice' or 'libreoffice' is in your PATH, or\n"
                    "  2) Provide the path via --libreoffice_path, or\n"
                    "  3) Use Docker-based conversion via --libreoffice_url"
                )
                sys.exit(1)
        libreoffice_endpoint = None

    # ---- 6) Process input path ----
    results = process_input_path(
        input_path=input_path,
        output_dir=output_dir,
        libreoffice_path=libreoffice_path,
        libreoffice_endpoint=libreoffice_endpoint,
        model_instance=model_instance,
        rate_limit=args.rate_limit,
        prompt=prompt,
        save_pdf=args.save_pdf,
        save_images=args.save_images,
        max_workers=args.max_workers
    )

    # ---- 6) Log Summary ----
    successes = [res for res in results if len(res[1]) > 0]
    failures = [res for res in results if len(res[1]) == 0]

    if successes:
        logger.info(f"Successfully processed {len(successes)} PPT file(s).")
    if failures:
        logger.warning(f"Failed to process {len(failures)} PPT file(s).")


if __name__ == "__main__":
    main()