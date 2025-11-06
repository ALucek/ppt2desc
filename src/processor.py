import time
import logging
import tempfile
import threading
import re
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    # Validate image path exists
    if not image_path.exists():
        error_msg = f"Image path does not exist for slide {slide_number}: {image_path}"
        logger.error(error_msg)
        return (slide_number, SlideData(number=slide_number, content=f"ERROR: {error_msg}"))
    
    # Acquire rate limit permission (blocks if needed)
    rate_limiter.acquire()
    
    try:
        response = model_instance.generate(prompt, image_path)
        if not response or not response.strip():
            logger.warning(f"Empty response from LLM for slide {slide_number}")
            return (slide_number, SlideData(number=slide_number, content="WARNING: Empty response from LLM"))
        return (slide_number, SlideData(number=slide_number, content=response))
    except Exception as e:
        error_msg = f"Error generating content for slide {slide_number}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (slide_number, SlideData(number=slide_number, content=f"ERROR: Failed to process slide - {str(e)}"))


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
      5) Optionally save PDF and images to main output directory
    
    Note: Intermediate results (PDF, images, partial JSON) are saved to 
    {output_dir}/{ppt_file.stem}/intermediate/ for resumability. If processing 
    completes successfully, the intermediate directory is automatically cleaned up. 
    The save_pdf and save_images flags control whether PDF/images are ALSO copied 
    to the main output directory.
    
    :param ppt_file: Path to the PowerPoint file to process
    :param output_dir: Directory where output JSON will be saved
    :param libreoffice_path: Path to local LibreOffice executable (if using local conversion)
    :param libreoffice_endpoint: URL to Docker LibreOffice service (if using Docker conversion)
    :param model_instance: The LLM client instance
    :param rate_limit: Maximum API calls per minute
    :param prompt: The prompt to use for LLM generation
    :param save_pdf: Whether to ALSO save the converted PDF to main output directory
    :param save_images: Whether to ALSO save extracted images to main output directory
    :param max_workers: Maximum number of concurrent workers (None for auto)
    :return: Tuple of (ppt_file_path, list_of_processed_image_paths)
    """
    # Validate inputs
    if not ppt_file.exists():
        logger.error(f"PowerPoint file does not exist: {ppt_file}")
        return (ppt_file, [])
    
    if not ppt_file.suffix.lower() in ('.ppt', '.pptx'):
        logger.error(f"Invalid file extension: {ppt_file.suffix}. Expected .ppt or .pptx")
        return (ppt_file, [])
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return (ppt_file, [])
    
    if rate_limit <= 0:
        logger.error(f"Rate limit must be positive, got: {rate_limit}")
        return (ppt_file, [])
    
    if max_workers is not None and max_workers <= 0:
        logger.error(f"max_workers must be positive, got: {max_workers}")
        return (ppt_file, [])
    
    # Create intermediate directory structure for saving intermediate results
    intermediate_dir = output_dir / ppt_file.stem / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    intermediate_images_dir = intermediate_dir / "images"
    intermediate_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing intermediate results to resume from
    intermediate_pdf = intermediate_dir / f"{ppt_file.stem}.pdf"
    partial_json_path = intermediate_dir / "slides_partial.json"
    output_file = output_dir / f"{ppt_file.stem}.json"
    error_json_path = intermediate_dir / "slides_partial_error.json"
    
    # Try to load existing partial results
    existing_slides_data: Dict[int, SlideData] = {}
    resume_from_intermediate = False
    
    # Check if we have a final JSON in the output directory (complete processing)
    if output_file.exists():
        try:
            final_data = DeckData.model_validate_json(output_file.read_text(encoding='utf-8'))
            logger.info(f"Found complete results in {output_file}. Skipping reprocessing.")
            # Return the paths - we'll use the intermediate images if they exist
            existing_intermediate_images = list(intermediate_images_dir.glob("slide_*.png"))
            if not existing_intermediate_images:
                existing_intermediate_images = list(intermediate_images_dir.glob("*.png"))
            return (ppt_file, existing_intermediate_images if existing_intermediate_images else [])
        except Exception as e:
            logger.warning(f"Could not load final JSON, will reprocess: {e}")
    
    # Check for partial results (try error JSON first, then partial JSON)
    partial_json_to_load = error_json_path if error_json_path.exists() else partial_json_path
    if partial_json_to_load.exists():
        try:
            partial_data = DeckData.model_validate_json(partial_json_to_load.read_text(encoding='utf-8'))
            # Load existing slide data, skipping placeholders and errors
            for slide in partial_data.slides:
                # Only keep slides that have valid content (not placeholders or errors)
                if (slide.content and 
                    not slide.content.startswith("...processing...") and 
                    not slide.content.startswith("ERROR:")):
                    existing_slides_data[slide.number] = slide
            
            if existing_slides_data:
                logger.info(f"Found partial results: {len(existing_slides_data)} slides already processed. Will resume from slide {max(existing_slides_data.keys()) + 1}")
                resume_from_intermediate = True
        except Exception as e:
            logger.warning(f"Could not load partial JSON, will start fresh: {e}")
    
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            # 1) PPT -> PDF (check if intermediate PDF exists)
            if intermediate_pdf.exists() and resume_from_intermediate:
                logger.info(f"Using existing intermediate PDF: {intermediate_pdf}")
                pdf_path = intermediate_pdf
                # Copy to temp dir for processing
                temp_pdf = temp_dir / pdf_path.name
                shutil.copy2(pdf_path, temp_pdf)
                pdf_path = temp_pdf
            else:
                # Need to convert (either no resume, or PDF is missing)
                if resume_from_intermediate and not intermediate_pdf.exists():
                    logger.info(f"Resuming with partial slide data, but PDF missing. Regenerating PDF...")
                elif not resume_from_intermediate:
                    logger.info(f"Starting fresh conversion")
                
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
                
                # Save intermediate PDF
                try:
                    shutil.copy2(pdf_path, intermediate_pdf)
                    logger.info(f"Saved intermediate PDF to {intermediate_pdf}")
                except Exception as e:
                    logger.warning(f"Failed to save intermediate PDF: {e}")

            # 2) PDF -> Images (check if intermediate images exist)
            intermediate_image_paths = []
            # Check if we have images in the intermediate directory
            intermediate_image_files = []
            if intermediate_images_dir.exists():
                intermediate_image_files = sorted(intermediate_images_dir.glob("slide_*.png"))
                if not intermediate_image_files:
                    intermediate_image_files = sorted(intermediate_images_dir.glob("*.png"))
            
            if resume_from_intermediate and intermediate_image_files:
                # Have partial results and images exist - use them
                logger.info(f"Using existing intermediate images: {len(intermediate_image_files)} images found")
                # Copy to temp dir for processing
                image_paths = []
                intermediate_image_paths = list(intermediate_image_files)  # Track original paths
                for img_file in intermediate_image_files:
                    temp_img = temp_dir / img_file.name
                    shutil.copy2(img_file, temp_img)
                    image_paths.append(temp_img)
            else:
                # Need to extract images (either no resume, or images are missing)
                if resume_from_intermediate and not intermediate_image_files:
                    logger.info(f"Resuming with partial slide data, but images missing. Regenerating images...")
                elif not resume_from_intermediate:
                    logger.info(f"Extracting images from PDF")
                
                image_paths = convert_pdf_to_images(pdf_path, temp_dir)
                if not image_paths:
                    logger.error(f"No images were generated from {pdf_path.name}")
                    return (ppt_file, [])
                
                # Save intermediate images
                try:
                    intermediate_image_paths = []
                    for img_path in image_paths:
                        if img_path.exists():
                            intermediate_img = intermediate_images_dir / img_path.name
                            shutil.copy2(img_path, intermediate_img)
                            intermediate_image_paths.append(intermediate_img)
                    logger.info(f"Saved {len(intermediate_image_paths)} intermediate images to {intermediate_images_dir}")
                except Exception as e:
                    logger.warning(f"Failed to save some intermediate images: {e}")

            # 3) Generate LLM content (concurrently with rate limiting)
            # Sort images by slide number (assuming "slide_1.png", "slide_2.png", etc.)
            def extract_slide_number(path: Path) -> int:
                """Extract slide number from filename like 'slide_1.png' or 'slide_10.png'."""
                try:
                    # Expected format: slide_N.png
                    parts = path.stem.split('_')
                    if len(parts) >= 2:
                        return int(parts[1])
                    else:
                        # Fallback: try to extract any number from filename
                        numbers = re.findall(r'\d+', path.stem)
                        if numbers:
                            return int(numbers[0])
                        raise ValueError(f"Cannot extract slide number from: {path.name}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not extract slide number from {path.name}, using 0: {e}")
                    return 0
            
            try:
                image_paths.sort(key=extract_slide_number)
            except Exception as e:
                logger.warning(f"Failed to sort images by slide number: {e}. Using natural sort.")
                image_paths.sort()
            
            # Create a thread-safe rate limiter shared across all workers
            rate_limiter = ThreadSafeRateLimiter(rate_limit)
            
            # Determine optimal number of workers
            num_slides = len(image_paths)
            
            if max_workers is None:
                # Default: use rate_limit if reasonable, otherwise cap at 10 or number of slides
                workers = min(rate_limit, num_slides, 10) if rate_limit > 0 else min(10, num_slides)
            else:
                # User specified max_workers: respect it but don't exceed rate_limit or num_slides
                workers = min(max_workers, num_slides)
                if rate_limit > 0:
                    workers = min(workers, rate_limit)
            
            # Ensure we have at least 1 worker
            workers = max(1, workers)
            
            logger.info(f"Processing {num_slides} slides with {workers} worker(s) (rate limit: {rate_limit}/min)")
            
            # Process slides concurrently (skip already processed slides)
            slides_data_dict = existing_slides_data.copy()  # Start with existing results
            partial_json_path = intermediate_dir / "slides_partial.json"
            
            # Determine which slides still need processing
            slides_to_process = []
            for idx, image_path in enumerate(image_paths, start=1):
                if idx not in slides_data_dict:
                    slides_to_process.append((idx, image_path))
            
            if slides_to_process:
                logger.info(f"Resuming: {len(slides_to_process)} slides remaining, {len(slides_data_dict)} already processed")
            else:
                logger.info(f"All {len(image_paths)} slides already processed. Loading from intermediate results.")
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit only tasks for slides that need processing
                future_to_slide = {
                    executor.submit(
                        _process_single_slide,
                        idx,
                        image_path,
                        model_instance,
                        rate_limiter,
                        prompt
                    ): (idx, image_path)
                    for idx, image_path in slides_to_process
                }
                
                # Collect results as they complete, with progress bar
                total_to_process = len(slides_to_process) if slides_to_process else 0
                initial_count = len(slides_data_dict)
                
                if total_to_process > 0:
                    with tqdm(
                        total=total_to_process, 
                        initial=0,
                        desc=f"Processing slides for {ppt_file.name} ({initial_count} already done)", 
                        unit="slide"
                    ) as pbar:
                        for future in as_completed(future_to_slide):
                            slide_num, slide_data = future.result()
                            slides_data_dict[slide_num] = slide_data
                            pbar.update(1)
                            
                            # Save partial JSON after each slide completion
                            try:
                                # Reconstruct partial slides_data in order based on what we have so far
                                if slides_data_dict:
                                    partial_slides = []
                                    # Include all slides up to num_slides, showing progress
                                    for i in range(1, num_slides + 1):
                                        if i in slides_data_dict:
                                            partial_slides.append(slides_data_dict[i])
                                        else:
                                            # Placeholder for slides not yet processed
                                            partial_slides.append(SlideData(number=i, content="...processing..."))
                            
                                    partial_deck_data = DeckData(
                                        deck=ppt_file.name,
                                        model=model_instance.model_name,
                                        slides=partial_slides
                                    )
                                    partial_json_path.write_text(
                                        partial_deck_data.model_dump_json(indent=2), 
                                        encoding='utf-8'
                                    )
                            except Exception as e:
                                # Don't fail the entire process if partial save fails
                                logger.debug(f"Failed to save partial JSON (non-critical): {e}")
                else:
                    # All slides already processed
                    logger.info(f"All {num_slides} slides were already processed. Using existing results.")
            
            # Reconstruct slides_data in order
            slides_data = [slides_data_dict[i] for i in range(1, num_slides + 1)]

            logger.info(f"Successfully converted {ppt_file.name} to {len(slides_data)} slides.")

            # 4) Build pydantic model and save final JSON
            deck_data = DeckData(
                deck=ppt_file.name,
                model=model_instance.model_name,
                slides=slides_data
            )
            output_file = output_dir / f"{ppt_file.stem}.json"
            output_file.write_text(deck_data.model_dump_json(indent=2), encoding='utf-8')
            logger.info(f"Final output written to {output_file}")

            # 5) Optionally save PDF
            if save_pdf:
                try:
                    destination_pdf = output_dir / pdf_path.name
                    shutil.copy2(pdf_path, destination_pdf)
                    logger.info(f"Saved PDF to {destination_pdf}")
                except Exception as e:
                    logger.error(f"Failed to save PDF to {output_dir}: {e}")

            # 6) Optionally save images
            if save_images:
                try:
                    images_subdir = output_dir / ppt_file.stem
                    images_subdir.mkdir(parents=True, exist_ok=True)
                    for img_path in image_paths:
                        if img_path.exists():
                            shutil.copy2(img_path, images_subdir / img_path.name)
                        else:
                            logger.warning(f"Image file no longer exists, skipping: {img_path}")
                    logger.info(f"Saved {len(image_paths)} images to {images_subdir}")
                except Exception as e:
                    logger.error(f"Failed to save images to {images_subdir}: {e}")

            # 7) Clean up intermediate files after successful processing
            try:
                if intermediate_dir.exists():
                    shutil.rmtree(intermediate_dir)
                    logger.info(f"Cleaned up intermediate directory: {intermediate_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up intermediate directory (non-critical): {e}")

            # Return success - image paths are no longer needed since processing completed successfully
            return (ppt_file, image_paths)

        except Exception as ex:
            logger.error(f"Unexpected error while processing {ppt_file.name}: {str(ex)}", exc_info=True)
            logger.info(f"Intermediate results saved to: {intermediate_dir}")
            # Try to save any partial results we might have
            try:
                # Check if we have any partial slide data
                if 'slides_data_dict' in locals() and slides_data_dict and len(slides_data_dict) > 0:
                    partial_slides = [slides_data_dict[i] for i in sorted(slides_data_dict.keys())]
                    partial_deck_data = DeckData(
                        deck=ppt_file.name,
                        model=model_instance.model_name,
                        slides=partial_slides
                    )
                    error_json_path = intermediate_dir / "slides_partial_error.json"
                    error_json_path.write_text(
                        partial_deck_data.model_dump_json(indent=2), 
                        encoding='utf-8'
                    )
                    logger.info(f"Saved partial results ({len(partial_slides)} slides) to {error_json_path}")
                else:
                    logger.info(f"No partial slide data to save (processing failed before slide processing)")
            except Exception as save_error:
                logger.warning(f"Could not save partial results: {save_error}")
            
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
    
    :param input_path: Path to a single PPT file or directory containing PPT files
    :param output_dir: Directory where output JSON files will be saved
    :param libreoffice_path: Path to local LibreOffice executable (if using local conversion)
    :param libreoffice_endpoint: URL to Docker LibreOffice service (if using Docker conversion)
    :param model_instance: The LLM client instance
    :param rate_limit: Maximum API calls per minute
    :param prompt: The prompt to use for LLM generation
    :param save_pdf: Whether to save converted PDFs
    :param save_images: Whether to save extracted images
    :param max_workers: Maximum number of concurrent workers (None for auto)
    :return: List of tuples (ppt_file_path, list_of_processed_image_paths)
    """
    results = []

    # Single file mode
    if input_path.is_file():
        if input_path.suffix.lower() in ('.ppt', '.pptx'):
            logger.info(f"Processing single file: {input_path.name}")
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
        else:
            logger.warning(f"Skipping non-PowerPoint file: {input_path.name} (extension: {input_path.suffix})")

    # Directory mode
    elif input_path.is_dir():
        # Find all PPT files (case-insensitive matching)
        ppt_files = list(input_path.glob('*.ppt')) + list(input_path.glob('*.pptx'))
        ppt_files = [f for f in ppt_files if f.is_file()]  # Filter out directories
        
        if not ppt_files:
            logger.warning(f"No PowerPoint files found in directory: {input_path}")
            return results
        
        logger.info(f"Found {len(ppt_files)} PowerPoint file(s) in {input_path}")
        
        for ppt_file in ppt_files:
            logger.info(f"Processing file: {ppt_file.name}")
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
    else:
        logger.error(f"Input path is neither a file nor a directory: {input_path}")

    return results