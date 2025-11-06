from pathlib import Path
from types import SimpleNamespace

import pytest

from converters.docker_converter import convert_pptx_via_docker
from converters.exceptions import ConversionError, LibreOfficeNotFoundError
from converters.pdf_converter import convert_pdf_to_images
from converters.ppt_converter import convert_pptx_to_pdf


def test_convert_pptx_to_pdf_invokes_libreoffice(tmp_path, monkeypatch):
    input_file = tmp_path / "deck.pptx"
    input_file.write_text("ppt", encoding="utf-8")
    libreoffice = tmp_path / "soffice"
    libreoffice.write_text("bin", encoding="utf-8")
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()

    def fake_run(cmd, check, capture_output, text):
        assert cmd[0] == str(libreoffice)
        assert cmd[-1] == str(input_file)
        (temp_dir / "deck.pdf").write_text("pdf", encoding="utf-8")
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr("converters.ppt_converter.subprocess.run", fake_run)

    result = convert_pptx_to_pdf(input_file, libreoffice, temp_dir)

    assert result == temp_dir / "deck.pdf"


def test_convert_pptx_to_pdf_missing_binary(tmp_path):
    input_file = tmp_path / "deck.pptx"
    input_file.write_text("ppt", encoding="utf-8")
    libreoffice = tmp_path / "missing"

    with pytest.raises(LibreOfficeNotFoundError):
        convert_pptx_to_pdf(input_file, libreoffice, tmp_path)


def test_convert_pptx_to_pdf_subprocess_error(tmp_path, monkeypatch):
    import subprocess

    input_file = tmp_path / "deck.pptx"
    input_file.write_text("ppt", encoding="utf-8")
    libreoffice = tmp_path / "soffice"
    libreoffice.write_text("bin", encoding="utf-8")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="soffice", stderr="boom")

    monkeypatch.setattr("converters.ppt_converter.subprocess.run", fake_run)

    with pytest.raises(ConversionError):
        convert_pptx_to_pdf(input_file, libreoffice, tmp_path)


def test_convert_pdf_to_images_success(tmp_path, monkeypatch):
    pdf_path = tmp_path / "deck.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")

    class FakePixmap:
        width = 100
        height = 100
        samples = b"\xff" * (width * height * 3)

    class FakePage:
        rect = SimpleNamespace(width=200, height=100)

        def get_pixmap(self, matrix, alpha=False):
            return FakePixmap()

    class FakeDoc:
        def __init__(self, path):
            self.pages = [FakePage(), FakePage()]

        def __len__(self):
            return len(self.pages)

        def load_page(self, idx):
            return self.pages[idx]

        def close(self):
            pass

    monkeypatch.setattr("converters.pdf_converter.fitz.open", lambda path: FakeDoc(path))

    images = convert_pdf_to_images(pdf_path, tmp_path)

    assert len(images) == 2
    assert images[0].name == "slide_1.png"
    assert images[0].exists()


def test_convert_pdf_to_images_skips_failed_pages(tmp_path, monkeypatch):
    pdf_path = tmp_path / "deck.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")

    class FakePixmap:
        width = 100
        height = 100
        samples = b"\xff" * (width * height * 3)

    class GoodPage:
        rect = SimpleNamespace(width=200, height=100)

        def get_pixmap(self, matrix, alpha=False):
            return FakePixmap()

    class BadPage:
        rect = SimpleNamespace(width=200, height=100)

        def get_pixmap(self, matrix, alpha=False):
            raise RuntimeError("fail")

    class FakeDoc:
        def __init__(self, path):
            self.pages = [GoodPage(), BadPage()]

        def __len__(self):
            return len(self.pages)

        def load_page(self, idx):
            return self.pages[idx]

        def close(self):
            pass

    monkeypatch.setattr("converters.pdf_converter.fitz.open", lambda path: FakeDoc(path))

    images = convert_pdf_to_images(pdf_path, tmp_path)

    assert len(images) == 1
    assert images[0].name == "slide_1.png"


def test_convert_pdf_to_images_raises_on_open_failure(monkeypatch):
    monkeypatch.setattr("converters.pdf_converter.fitz.open", lambda path: (_ for _ in ()).throw(RuntimeError("fail")))

    with pytest.raises(ConversionError):
        convert_pdf_to_images(Path("dummy.pdf"), Path("/tmp"))


def test_convert_pptx_via_docker_success(tmp_path, monkeypatch):
    ppt_file = tmp_path / "deck.pptx"
    ppt_file.write_text("ppt", encoding="utf-8")
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()

    class FakeResponse:
        def __init__(self):
            self._iter_content = [b"chunk1", b"chunk2"]

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            return iter(self._iter_content)

    def fake_post(url, files, timeout):
        assert url.endswith("/convert/ppt-to-pdf")
        assert files["file"][0] == "deck.pptx"
        return FakeResponse()

    monkeypatch.setattr("converters.docker_converter.requests.post", fake_post)

    pdf_path = convert_pptx_via_docker(ppt_file, "http://localhost:2002", temp_dir)

    assert pdf_path.exists()
    assert pdf_path.read_bytes() == b"chunk1chunk2"


def test_convert_pptx_via_docker_failure(tmp_path, monkeypatch):
    import requests

    ppt_file = tmp_path / "deck.pptx"
    ppt_file.write_text("ppt", encoding="utf-8")

    def fake_post(*args, **kwargs):
        raise requests.RequestException("fail")

    monkeypatch.setattr("converters.docker_converter.requests.post", fake_post)

    with pytest.raises(ConversionError):
        convert_pptx_via_docker(ppt_file, "http://localhost:2002", tmp_path)

