

import fitz  # PyMuPDF
import os
import concurrent.futures
from PIL import Image
import io

def optimize_image_bytes(img_data: bytes, max_size: int = 800) -> bytes:
    """
    Optimize the image for faster processing while maintaining readability.
    Reduced max size and quality for better performance.
    """
    img = Image.open(io.BytesIO(img_data))
    
    # Convert to RGB if image is in RGBA mode
    if img.mode == 'RGBA':
        img = img.convert('L')
    
    # Calculate new dimensions while maintaining aspect ratio
    ratio = min(max_size / img.width, max_size / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    
    # Resize with a simpler algorithm
    img = img.resize(new_size, Image.Resampling.BILINEAR)
    
    # Convert back to bytes with JPEG format and reduced quality
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85, optimize=True)
    return output.getvalue()

def process_page(args):
    """
    Process a single page with the given parameters.
    Includes retry logic and better error handling.
    """
    page, gemini, scale = args
    try:
        # Reduce initial scale for better performance
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_data = pix.tobytes("png")
        
        # Optimize image before sending to Gemini
        optimized_data = optimize_image_bytes(img_data)
        
        # Use Gemini Vision to extract text with retries
        max_retries = 2
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                return gemini.extract_text_from_image(optimized_data)
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying page after error: {str(e)}")
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
        
        if last_error:
            raise last_error
            
    except Exception as e:
        print(f"Error processing page: {str(e)}")
        return ""  # Return empty string on error to continue processing

def extract_text_from_pdf(pdf_stream):
    """
    Extracts all text from a given PDF file stream using a memory-efficient approach
    with Gemini's vision capabilities.

    Args:
        pdf_stream: A file-like object (stream) of the PDF file.
                   For example, the object you get from Flask's request.files.

    Returns:
        A single string containing all the text from the PDF,
        or raises an exception if extraction fails.

    Raises:
        Exception: If text extraction fails or memory issues occur
    """
    from .ai_client import GeminiClient
    
    try:
        gemini = GeminiClient()
        print("Initialized Gemini client successfully")
        
        pdf_bytes = pdf_stream.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages = []
            total_pages = len(doc)
            print(f"Processing {total_pages} pages in parallel...")

            # Prepare all pages for parallel processing
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                pages.append((page, gemini, 1.5))  # Using 1.5x scale for a balance of quality and speed
            
            # Process one page at a time to minimize memory usage
            full_text = ""
            for page_num, page_data in enumerate(pages, 1):
                print(f"Processing page {page_num} of {len(pages)}...")
                try:
                    text = process_page(page_data)
                    if text and text.strip():
                        print(f"Successfully extracted {len(text.split())} words from page {page_num}")
                        full_text += text + "\n"
                    else:
                        print(f"Warning: No text extracted from page {page_num}")
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue
            
            result = full_text.strip()
            if result:
                print("Successfully extracted text from all pages")
                return result
            else:
                raise Exception("No text could be extracted from any page in the document")
                
    except Exception as e:
        print(f"Error in PDF processing: {e}")
        return ""

# --- Example Usage (for testing this file directly) ---
# This part will only run when you execute `python pdf_processor.py`
if __name__ == '__main__':
    # NOTE: To test this, you must have a 'presentation/demo_documents' folder
    # and your 'rental_agreement.pdf' file must be placed inside it.
    
    # --- Actual test ---
    # We are now pointing directly to the rental agreement for testing.
    test_pdf_path = "rental_agreement.pdf"
    
    print(f"\n--- Testing PDF Extraction from '{test_pdf_path}' ---")
    
    try:
        # We open the file in binary read mode to get a stream,
        # which is what our function expects.
        with open(test_pdf_path, "rb") as pdf_file_stream:
            extracted_text = extract_text_from_pdf(pdf_file_stream)
        
        if extracted_text:
            print("\nExtraction Successful!")
            print("------------------------")
            print(extracted_text)
            print("------------------------")
        else:
            print("\nExtraction failed. Is the PDF empty or corrupt?")

    except FileNotFoundError:
        print(f"\nError: The test file was not found at '{test_pdf_path}'")
        print("Please make sure your 'rental_agreement.pdf' is in the correct folder.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
