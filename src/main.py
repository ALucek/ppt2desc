import logging
import argparse
import sys
from pathlib import Path

from llm.gemini import GeminiClient
from llm.vertex import VertexAIClient
from llm.openai import OpenAIClient
from llm.anthropic import AnthropicClient
from llm.azure import AzureClient

from processor import process_input_path

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Process PPT/PPTX files via vLLM.")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        required=True,
        help="Path to input directory or PPT file"
    )
    parser.add_argument(
        "--client",
        type=str,
        default="gemini",
        choices=["gemini", "vertexai", "openai", "anthropic", "azure"],
        help="LLM client to use: 'gemini', 'vertexai', 'openai', 'azure', or 'anthropic'"
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
        help="Number of API calls allowed per minute (default: 60)"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prompt.txt",
        help="Path to the base prompt file (default: src/prompt.txt)"
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

    args = parser.parse_args(input_args) if input_args else parser.parse_args()
    return args

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

    # ---- 3) Read prompt once ----
    base_prompt_file = Path(args.prompt_path)
    if not base_prompt_file.is_file():
        logger.error(f"Prompt file not found at {base_prompt_file}")
        sys.exit(1)

    base_prompt = base_prompt_file.read_text(encoding="utf-8").strip()
    if args.instructions and args.instructions.lower() != "none provided":
        prompt = f"{base_prompt}\n\nAdditional instructions:\n{args.instructions}"
    else:
        prompt = base_prompt

    # ---- 4) Initialize model instance ----
    try:
        if args.client == "gemini":
            model_instance = GeminiClient(api_key=args.api_key, model=args.model)
            logger.info(f"Initialized GeminiClient with model: {args.model}")
        elif args.client == "vertexai":
            model_instance = VertexAIClient(
                credentials_path=args.gcp_application_credentials,
                project_id=args.gcp_project_id,
                region=args.gcp_region,
                model=args.model
            )
            logger.info(f"Initialized VertexAIClient for project: {args.gcp_project_id}")
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
        else:
            logger.error(f"Unsupported client specified: {args.client}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.libreoffice_path:
        libreoffice_path = Path(args.libreoffice_path)
    else:
        # If no path is provided, assume 'libreoffice' is in PATH
        libreoffice_path = Path("libreoffice")

    # ---- 5) Process input path ----
    results = process_input_path(
        input_path=input_path,
        output_dir=output_dir,
        libreoffice_path=libreoffice_path,
        model_instance=model_instance,
        rate_limit=args.rate_limit,
        prompt=prompt
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
