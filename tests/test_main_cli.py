import argparse
import logging
from pathlib import Path

import pytest

import main


def test_parse_args_defaults(tmp_path):
    args = main.parse_args(
        [
            "--output_dir",
            str(tmp_path / "out"),
            "--input_dir",
            str(tmp_path / "slides"),
            "--client",
            "openai",
        ]
    )

    assert args.model == "gemini-2.5-flash"
    assert args.instructions == "None Provided"
    assert args.rate_limit == 60
    assert args.prompt_path is None


def _base_args(tmp_path):
    input_file = tmp_path / "deck.pptx"
    input_file.write_text("dummy", encoding="utf-8")
    return argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        input_dir=str(input_file),
        client="openai",
        model="test-model",
        instructions="None Provided",
        libreoffice_path=None,
        libreoffice_url=None,
        rate_limit=10,
        max_workers=5,
        prompt_path=None,
        api_key="abc",
        gcp_region=None,
        gcp_project_id=None,
        gcp_application_credentials=None,
        azure_openai_api_key=None,
        azure_openai_endpoint=None,
        azure_deployment_name=None,
        azure_api_version="2023-12-01-preview",
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_region="us-east-1",
        save_pdf=False,
        save_images=False,
    )


def test_validate_args_success_creates_output_dir(tmp_path):
    args = _base_args(tmp_path)
    logger = logging.getLogger("test")

    main.validate_args(args, logger)

    assert Path(args.output_dir).exists()


def test_validate_args_rejects_non_positive_rate_limit(tmp_path):
    args = _base_args(tmp_path)
    args.rate_limit = 0
    logger = logging.getLogger("test")

    with pytest.raises(SystemExit):
        main.validate_args(args, logger)


def test_validate_args_requires_vertex_credentials(tmp_path):
    args = _base_args(tmp_path)
    args.client = "vertexai"
    args.gcp_project_id = None
    args.gcp_application_credentials = None
    logger = logging.getLogger("test")

    with pytest.raises(SystemExit):
        main.validate_args(args, logger)


def test_validate_args_requires_azure_settings(tmp_path):
    args = _base_args(tmp_path)
    args.client = "azure"
    logger = logging.getLogger("test")

    with pytest.raises(SystemExit):
        main.validate_args(args, logger)


def test_validate_args_requires_aws_keys(tmp_path):
    args = _base_args(tmp_path)
    args.client = "aws"
    logger = logging.getLogger("test")

    with pytest.raises(SystemExit):
        main.validate_args(args, logger)


def test_main_uses_custom_prompt_file_and_instructions(tmp_path, monkeypatch):
    deck = tmp_path / "deck.pptx"
    deck.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Base Prompt", encoding="utf-8")

    args = argparse.Namespace(
        output_dir=str(output_dir),
        input_dir=str(deck),
        client="openai",
        model="test-model",
        instructions="Focus on charts",
        libreoffice_path=None,
        libreoffice_url="http://localhost:2002",
        rate_limit=5,
        max_workers=None,
        prompt_path=str(prompt_file),
        api_key="abc",
        gcp_region=None,
        gcp_project_id=None,
        gcp_application_credentials=None,
        azure_openai_api_key=None,
        azure_openai_endpoint=None,
        azure_deployment_name=None,
        azure_api_version="2023-12-01-preview",
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_region="us-east-1",
        save_pdf=False,
        save_images=False,
    )

    captured = {}

    class DummyClient:
        def __init__(self, api_key, model):
            self.model_name = model

    def fake_process_input_path(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        assert kwargs["input_path"] == Path(deck)
        return []

    monkeypatch.setattr(main, "parse_args", lambda input_args=None: args)
    monkeypatch.setattr(main, "OpenAIClient", lambda api_key, model: DummyClient(api_key, model))
    monkeypatch.setattr(main, "process_input_path", fake_process_input_path)

    main.main()

    assert captured["prompt"] == "Base Prompt\n\nAdditional instructions:\nFocus on charts"

