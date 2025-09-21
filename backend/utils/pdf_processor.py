

import fitz  # PyMuPDF
import os
import concurrent.futures
from PIL import Image
import io

def optimize_image_bytes(img_data: bytes) -> bytes:
    """
    Optimize the image for faster processing:
    - Convert to grayscale
    - Resize to max 1024x1024
    - Optimize quality
    """
    img = Image.open(io.BytesIO(img_data))
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Set maximum dimensions
    max_size = (1024, 1024)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert back to bytes with JPEG format and optimized settings
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85, optimize=True)
    return output.getvalue()

def process_page(page, gemini):
    """
    Process a single page sequentially with optimization and error handling.
    """
    pix = None
    result = ""
    try:
        # Use a lower scale factor for the initial render
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
        img_data = pix.tobytes("png")
        
        # Optimize image before sending to Gemini
        optimized_data = optimize_image_bytes(img_data)
        
        # Use Gemini Vision to extract text with retries
        max_retries = 2
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                result = gemini.extract_text_from_image(optimized_data)
                if result:
                    break
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying page after error: {str(e)}")
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
        
        if last_error and not result:
            raise last_error
        
        return result
            
    except Exception as e:
        print(f"Error processing page: {str(e)}")
        return ""  # Return empty string on error to continue processing
    finally:
        # Clean up
        if pix:
            pix = None

def extract_text_from_pdf(pdf_stream):
    """
    Extracts text from a PDF file stream using sequential processing and
    optimized image handling to minimize memory usage.

    Args:
        pdf_stream: A file-like object (stream) of the PDF file.
                   For example, the object you get from Flask's request.files.

    Returns:
        A single string containing all the text from the PDF.

    Raises:
        Exception: If text extraction completely fails
    """
    from .ai_client import GeminiClient
    
    full_text = ""
    try:
        gemini = GeminiClient()
        print("Initialized Gemini client successfully")
        
        pdf_bytes = pdf_stream.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        try:
            total_pages = len(doc)
            print(f"Processing {total_pages} pages sequentially...")
            
            # Process each page sequentially
            for page_num in range(total_pages):
                print(f"Processing page {page_num + 1} of {total_pages}...")
                try:
                    page = doc.load_page(page_num)
                    text = process_page(page, gemini)
                    
                    if text and text.strip():
                        print(f"Successfully extracted text from page {page_num + 1}")
                        full_text += text + "\n"
                    else:
                        print(f"No text extracted from page {page_num + 1}")
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
        finally:
            # Always close the document
            doc.close()
        
        # Check if we got any text at all
        if not full_text.strip():
            raise Exception("No text could be extracted from the PDF")
            
        return full_text.strip()
                
    except Exception as e:
        print(f"Error in PDF processing: {str(e)}")
        if not full_text.strip():
            raise Exception("Failed to extract text from PDF") from e
        return full_text.strip()  # Return any text we managed to extract
            


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
