import os

def process_text_document(file_path):
    """
    Reads text files and chunks them into readable segments for the Vector Brain.
    (Phase 2 of your startup: Upgrade this to read PDFs using PyPDF2)
    """
    print(f"📄 Reading document: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split text into rough chunks of ~500 characters
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]
    
    segments = []
    for chunk in chunks:
        segments.append({
            "type": "document",
            "content": chunk,
            # Text doesn't have a video timestamp, so we default to 0
            "start": 0,  
            "end": 0
        })
        
    return segments