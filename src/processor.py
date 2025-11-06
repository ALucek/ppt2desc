import time
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Union
from tqdm import tqdm
import shutil

from llm import LLMClient
from converters.ppt_converter import convert_pptx_to_pdf
from converters.pdf_converter import convert_pdf_to_images
from converters.docker_converter import convert_pptx_via_docker
from schemas.deck import DeckData, SlideData

# Create a type alias for all possible clients
logger = logging.getLogger(__name__)


def process_single_file(
    ppt_file: Path,
    output_dir: Path,
    libreoffice_path: Union[Path, None],
    libreoffice_endpoint: Union[str, None],
    model_instance: LLMClient,
    rate_limit: int,
    prompt: str,
    save_pdf: bool = False,
    save_images: bool = False
) -> Tuple[Path, List[Path]]:
    """
    Process a single PowerPoint file:
      1) Convert to PDF (either via local LibreOffice or Docker container)
      2) Convert PDF to images
      3) Send images to LLM
      4) Save JSON output
      5) Optionally save PDF and images
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            # 1) PPT -> PDF
            if libreoffice_endpoint:
                # Docker-based conversion
                pdf_path = convert_pptx_via_docker(
                    ppt_file,
                    libreoffice_endpoint,
                    temp_dir
                )
            else:
                # Local-based conversion
                pdf_path = convert_pptx_to_pdf(
                    input_file=ppt_file,
                    libreoffice_path=libreoffice_path,
                    temp_dir=temp_dir
                )

            logger.info(f"Successfully converted {ppt_file.name} to {pdf_path.name}")

            # 2) PDF -> Images (local PyMuPDF)
            image_paths = convert_pdf_to_images(pdf_path, temp_dir)
            if not image_paths:
                logger.error(f"No images were generated from {pdf_path.name}")
                return (ppt_file, [])

            # 3) Generate LLM content
            slides_data = []
            min_interval = 60.0 / rate_limit if rate_limit > 0 else 0
            last_call_time = 0.0

            # Sort images by slide number (assuming "slide_1.png", "slide_2.png", etc.)
            image_paths.sort(key=lambda p: int(p.stem.split('_')[1]))

            for idx, image_path in enumerate(
                tqdm(image_paths, desc=f"Processing slides for {ppt_file.name}", unit="slide"), start=1
            ):
                if min_interval > 0:
                    current_time = time.time()
                    time_since_last = current_time - last_call_time
                    if time_since_last < min_interval:
                        time.sleep(min_interval - time_since_last)

                try:
                    response = model_instance.generate(prompt, image_path)
                    slides_data.append(SlideData(number=idx, content=response))
                except Exception as e:
                    logger.error(f"Error generating content for slide {idx}: {str(e)}")
                    slides_data.append(SlideData(number=idx, content="ERROR: Failed to process slide"))
                finally:
                    # Always update last_call_time after the API call (whether success or failure)
                    if min_interval > 0:
                        last_call_time = time.time()

            logger.info(f"Successfully converted {ppt_file.name} to {len(slides_data)} slides.")

            # 4) Build pydantic model and save JSON
            deck_data = DeckData(
                deck=ppt_file.name,
                model=model_instance.model_name,
                slides=slides_data
            )
            output_file = output_dir / f"{ppt_file.stem}.json"
            output_file.write_text(deck_data.model_dump_json(indent=2), encoding='utf-8')
            logger.info(f"Output written to {output_file}")

            # 5) Optionally save PDF
            if save_pdf:
                destination_pdf = output_dir / pdf_path.name
                shutil.copy2(pdf_path, destination_pdf)
                logger.info(f"Saved PDF to {destination_pdf}")

            # 6) Optionally save images
            if save_images:
                images_subdir = output_dir / ppt_file.stem
                images_subdir.mkdir(parents=True, exist_ok=True)
                for img_path in image_paths:
                    shutil.copy2(img_path, images_subdir / img_path.name)
                logger.info(f"Saved images to {images_subdir}")

            return (ppt_file, image_paths)

        except Exception as ex:
            logger.error(f"Unexpected error while processing {ppt_file.name}: {str(ex)}")
            return (ppt_file, [])

def process_input_path(
    input_path: Path,
    output_dir: Path,
    libreoffice_path: Union[Path, None],
    libreoffice_endpoint: Union[str, None],
    model_instance: LLMClient,
    rate_limit: int,
    prompt: str,
    save_pdf: bool = False,
    save_images: bool = False
) -> List[Tuple[Path, List[Path]]]:
    """
    Process one or more PPT files from the specified path.
    Optionally save PDFs and images to the output directory.
    """
    results = []

    # Single file mode
    if input_path.is_file():
        if input_path.suffix.lower() in ('.ppt', '.pptx'):
            res = process_single_file(
                ppt_file=input_path,
                output_dir=output_dir,
                libreoffice_path=libreoffice_path,
                libreoffice_endpoint=libreoffice_endpoint,
                model_instance=model_instance,
                rate_limit=rate_limit,
                prompt=prompt,
                save_pdf=save_pdf,
                save_images=save_images
            )
            results.append(res)

    # Directory mode
    else:
        for ppt_file in input_path.glob('*.ppt*'):
            res = process_single_file(
                ppt_file=ppt_file,
                output_dir=output_dir,
                libreoffice_path=libreoffice_path,
                libreoffice_endpoint=libreoffice_endpoint,
                model_instance=model_instance,
                rate_limit=rate_limit,
                prompt=prompt,
                save_pdf=save_pdf,
                save_images=save_images
            )
            results.append(res)

    return results