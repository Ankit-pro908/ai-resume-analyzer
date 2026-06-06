import PyPDF2
import io

def extract_text_from_pdf(uploaded_file):
    """
    Takes uploaded PDF file
    Returns all text from it as a string
    """
    text = ""
    
    try:
        # Read the PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        
        # Loop through every page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
            
        return text.strip()
    
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def clean_text(text):
    """
    Cleans extra spaces, newlines from text
    """
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().strip()