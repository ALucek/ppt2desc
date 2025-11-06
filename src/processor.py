import time
import logging
import tempfile
import threading
from pathlib import Path
from typing import List, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

from llm import LLMClient
from converters.ppt_converter import convert_pptx_to_pdf
from converters.pdf_converter import convert_pdf_to_images
from converters.docker_converter import convert_pptx_via_docker
from schemas.deck import DeckData, SlideData

# Create a type alias for all possible clients
logger = logging.getLogger(__name__)


class ThreadSafeRateLimiter:
    """
    Thread-safe rate limiter that enforces API rate limits across multiple threads.
    Uses a token bucket algorithm to ensure requests are evenly distributed.
    """
    
    def __init__(self, rate_limit: int):
        """
        Initialize the rate limiter.
        
        :param rate_limit: Maximum number of API calls allowed per minute
        """
        self.rate_limit = rate_limit
        self.min_interval = 60.0 / rate_limit if rate_limit > 0 else 0
        self.last_call_time = 0.0
        self.lock = threading.Lock()
    
    def acquire(self):
        """
        Block until the rate limit allows another API call.
        This method is thread-safe and can be called concurrently.
        """
        if self.rate_limit <= 0:
            return
        
        while True:
            with self.lock:
                current_time = time.time()
                time_since_last = current_time - self.last_call_time
                
                if time_since_last >= self.min_interval:
                    # We can proceed, update last call time and exit
                    self.last_call_time = current_time
                    return
                
                # Calculate sleep time while holding the lock
                sleep_time = self.min_interval - time_since_last
            
            # Sleep outside the lock to allow other threads to acquire
            time.sleep(sleep_time)


def _process_single_slide(
    slide_number: int,
    image_path: Path,
    model_instance: LLMClient,
    rate_limiter: ThreadSafeRateLimiter,
    prompt: str
) -> Tuple[int, SlideData]:
    """
    Process a single slide image through the LLM.
    This function is designed to be called concurrently.
    
    :param slide_number: The slide number (1-indexed)
    :param image_path: Path to the slide image
    :param model_instance: The LLM client instance
    :param rate_limiter: Thread-safe rate limiter
    :param prompt: The prompt to use for generation
    :return: Tuple of (slide_number, SlideData)
    """
    # Acquire rate limit permission (blocks if needed)
    rate_limiter.acquire()
    
    try:
        response = model_instance.generate(prompt, image_path)
        return (slide_number, SlideData(number=slide_number, content=response))
    except Exception as e:
        logger.error(f"Error generating content for slide {slide_number}: {str(e)}")
        return (slide_number, SlideData(number=slide_number, content="ERROR: Failed to process slide"))


def process_single_file(
    ppt_file: Path,
    output_dir: Path,
    libreoffice_path: Union[Path, None],
    libreoffice_endpoint: Union[str, None],
    model_instance: LLMClient,
    rate_limit: int,
    prompt: str,
    save_pdf: bool = False,
    save_images: bool = False,
    max_workers: Optional[int] = None
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

            # 3) Generate LLM content (concurrently with rate limiting)
            # Sort images by slide number (assuming "slide_1.png", "slide_2.png", etc.)
            image_paths.sort(key=lambda p: int(p.stem.split('_')[1]))
            
            # Create a thread-safe rate limiter shared across all workers
            rate_limiter = ThreadSafeRateLimiter(rate_limit)
            
            # Determine optimal number of workers
            # Use min of max_workers, rate_limit, and number of slides
            num_slides = len(image_paths)
            if max_workers is None:
                # Default to rate_limit if available, otherwise use a reasonable default
                workers = min(rate_limit, num_slides) if rate_limit > 0 else min(10, num_slides)
            else:
                workers = min(max_workers, rate_limit, num_slides) if rate_limit > 0 else min(max_workers, num_slides)
            
            # Ensure we have at least 1 worker
            workers = max(1, workers)
            
            logger.info(f"Processing {num_slides} slides with {workers} worker(s) (rate limit: {rate_limit}/min)")
            
            # Process slides concurrently
            slides_data_dict = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_slide = {
                    executor.submit(
                        _process_single_slide,
                        idx,
                        image_path,
                        model_instance,
                        rate_limiter,
                        prompt
                    ): (idx, image_path)
                    for idx, image_path in enumerate(image_paths, start=1)
                }
                
                # Collect results as they complete, with progress bar
                with tqdm(total=num_slides, desc=f"Processing slides for {ppt_file.name}", unit="slide") as pbar:
                    for future in as_completed(future_to_slide):
                        slide_num, slide_data = future.result()
                        slides_data_dict[slide_num] = slide_data
                        pbar.update(1)
            
            # Reconstruct slides_data in order
            slides_data = [slides_data_dict[i] for i in range(1, num_slides + 1)]

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
    save_images: bool = False,
    max_workers: Optional[int] = None
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
                save_images=save_images,
                max_workers=max_workers
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
                save_images=save_images,
                max_workers=max_workers
            )
            results.append(res)

    return results