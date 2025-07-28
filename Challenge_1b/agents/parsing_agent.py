# agents/parsing_agent.py
import fitz  # PyMuPDF

class PDFParsingAgent:
    """
    Parses the PDF to extract raw text blocks with rich metadata.
    """
    def extract_text_blocks(self, pdf_path: str) -> list:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening {pdf_path}: {e}")
            return []
            
        blocks = []
        for page_num, page in enumerate(doc):
            page_blocks = page.get_text("dict")["blocks"]
            for block in page_blocks:
                if block['type'] == 0:  # Text blocks
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                blocks.append({
                                    "text": span["text"].strip(),
                                    "font_size": span["size"],
                                    "font_name": span["font"],
                                    "bbox": span["bbox"],
                                    "page": page_num + 1
                                })
        doc.close()
        return blocks