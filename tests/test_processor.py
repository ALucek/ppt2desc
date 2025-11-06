import json
from pathlib import Path

import pytest  # type: ignore[import]

import processor
from conftest import StubLLMClient
from processor import ThreadSafeRateLimiter, _process_single_slide, process_input_path, process_single_file
from schemas.deck import SlideData


def test_rate_limiter_enforces_min_interval(monkeypatch):
    limiter = ThreadSafeRateLimiter(rate_limit=60)  # 1 request per second
    call_times = iter([10.0, 10.4, 11.4, 12.5])
    sleeps = []

    monkeypatch.setattr(processor.time, "time", lambda: next(call_times))

    def fake_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr(processor.time, "sleep", fake_sleep)

    limiter.acquire()
    limiter.acquire()

    assert sleeps == pytest.approx([0.6])


class NoopRateLimiter:
    def acquire(self):
        return None


def test_process_single_slide_success(tmp_path):
    image_path = tmp_path / "slide_1.png"
    image_path.write_bytes(b"fake")
    client = StubLLMClient(responses=["slide content"])

    slide_number, slide_data = _process_single_slide(1, image_path, client, NoopRateLimiter(), "prompt")

    assert slide_number == 1
    assert isinstance(slide_data, SlideData)
    assert slide_data.content == "slide content"


def test_process_single_slide_empty_response(tmp_path):
    image_path = tmp_path / "slide_1.png"
    image_path.write_bytes(b"fake")
    client = StubLLMClient(responses=[""])

    _, slide_data = _process_single_slide(1, image_path, client, NoopRateLimiter(), "prompt")

    assert "WARNING" in slide_data.content


def test_process_single_slide_missing_image(tmp_path):
    client = StubLLMClient()

    _, slide_data = _process_single_slide(1, tmp_path / "missing.png", client, NoopRateLimiter(), "prompt")

    assert slide_data.content.startswith("ERROR")


def test_process_single_slide_handles_exception(tmp_path):
    image_path = tmp_path / "slide_1.png"
    image_path.write_bytes(b"fake")
    client = StubLLMClient(error=RuntimeError("boom"))

    _, slide_data = _process_single_slide(1, image_path, client, NoopRateLimiter(), "prompt")

    assert slide_data.content.startswith("ERROR")


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, value):
        return None


def test_process_single_file_happy_path(tmp_path, monkeypatch):
    ppt_file = tmp_path / "deck.pptx"
    ppt_file.write_text("ppt", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    client = StubLLMClient(responses=["first", "second"])

    def fake_convert_pptx_to_pdf(input_file, libreoffice_path, temp_dir):
        pdf_path = temp_dir / f"{input_file.stem}.pdf"
        pdf_path.write_text("pdf", encoding="utf-8")
        return pdf_path

    def fake_convert_pdf_to_images(pdf_path, temp_dir):
        paths = []
        for idx in range(1, 3):
            img = temp_dir / f"slide_{idx}.png"
            img.write_text(f"img{idx}", encoding="utf-8")
            paths.append(img)
        return paths

    monkeypatch.setattr(processor, "convert_pptx_to_pdf", fake_convert_pptx_to_pdf)
    monkeypatch.setattr(processor, "convert_pdf_to_images", fake_convert_pdf_to_images)
    monkeypatch.setattr(processor, "convert_pptx_via_docker", fake_convert_pptx_to_pdf)
    monkeypatch.setattr(processor, "tqdm", DummyTqdm)

    processed = process_single_file(
        ppt_file=ppt_file,
        output_dir=output_dir,
        libreoffice_path=None,
        libreoffice_endpoint=None,
        model_instance=client,
        rate_limit=10,
        prompt="prompt",
        save_pdf=True,
        save_images=True,
        max_workers=1,
    )

    output_json = output_dir / "deck.json"
    assert output_json.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["deck"] == "deck.pptx"
    assert [slide["content"] for slide in data["slides"]] == ["first", "second"]

    assert (output_dir / "deck.pdf").exists()
    images_dir = output_dir / "deck"
    assert {p.name for p in images_dir.iterdir()} == {"slide_1.png", "slide_2.png"}
    assert processed[0] == ppt_file

    intermediate_dir = output_dir / "deck" / "intermediate"
    assert not intermediate_dir.exists()


def test_process_single_file_invalid_extension(tmp_path):
    ppt_file = tmp_path / "deck.txt"
    ppt_file.write_text("bad", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = process_single_file(
        ppt_file=ppt_file,
        output_dir=output_dir,
        libreoffice_path=None,
        libreoffice_endpoint=None,
        model_instance=StubLLMClient(),
        rate_limit=10,
        prompt="prompt",
    )

    assert result[1] == []


def test_process_single_file_resume_from_partial(tmp_path, monkeypatch):
    ppt_file = tmp_path / "deck.pptx"
    ppt_file.write_text("ppt", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    intermediate_dir = output_dir / "deck" / "intermediate"
    images_dir = intermediate_dir / "images"
    images_dir.mkdir(parents=True)

    pdf_path = intermediate_dir / "deck.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")

    # Existing first slide, second pending
    partial = {
        "deck": "deck.pptx",
        "model": "stub-llm",
        "slides": [
            {"number": 1, "content": "existing"},
            {"number": 2, "content": "...processing..."},
        ],
    }
    (intermediate_dir / "slides_partial.json").write_text(json.dumps(partial), encoding="utf-8")

    slide1 = images_dir / "slide_1.png"
    slide2 = images_dir / "slide_2.png"
    slide1.write_text("img1", encoding="utf-8")
    slide2.write_text("img2", encoding="utf-8")

    client = StubLLMClient(responses=["new slide 2"])

    monkeypatch.setattr(processor, "tqdm", DummyTqdm)

    result = process_single_file(
        ppt_file=ppt_file,
        output_dir=output_dir,
        libreoffice_path=None,
        libreoffice_endpoint=None,
        model_instance=client,
        rate_limit=10,
        prompt="prompt",
        max_workers=1,
    )

    output_json = output_dir / "deck.json"
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert [slide["content"] for slide in data["slides"]] == ["existing", "new slide 2"]
    assert result[0] == ppt_file


def test_process_input_path_handles_file_and_directory(tmp_path, monkeypatch):
    ppt_file = tmp_path / "deck.pptx"
    ppt_file.write_text("ppt", encoding="utf-8")
    directory = tmp_path / "dir"
    directory.mkdir()
    ppt2 = directory / "other.pptx"
    ppt2.write_text("ppt", encoding="utf-8")
    (directory / "skip.txt").write_text("text", encoding="utf-8")

    calls = []

    def fake_process_single_file(**kwargs):
        calls.append(kwargs["ppt_file"])
        return (kwargs["ppt_file"], [])

    monkeypatch.setattr(processor, "process_single_file", fake_process_single_file)

    args = dict(
        output_dir=tmp_path / "output",
        libreoffice_path=None,
        libreoffice_endpoint=None,
        model_instance=StubLLMClient(),
        rate_limit=10,
        prompt="prompt",
        save_pdf=False,
        save_images=False,
        max_workers=None,
    )
    args["output_dir"].mkdir()

    process_input_path(ppt_file, **args)
    process_input_path(directory, **args)

    assert calls == [ppt_file, ppt2]

