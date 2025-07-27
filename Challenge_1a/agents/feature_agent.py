# agents/feature_agent.py
import statistics
import re

class FeatureEngineeringAgent:
    """
    Engineers intelligent features from raw text blocks.
    """
    def create_features(self, text_blocks: list) -> list:
        if not text_blocks:
            return []

        font_sizes = [b['font_size'] for b in text_blocks]
        median_font_size = statistics.median(font_sizes) if font_sizes else 12.0

        for block in text_blocks:
            block['is_bold'] = 'bold' in block['font_name'].lower()
            block['relative_font_size'] = block['font_size'] / median_font_size if median_font_size > 0 else 1
            
            # A page width of 612 points is standard for letter/A4 size
            page_width = 612 
            block['is_centered'] = abs((block['bbox'][0] + block['bbox'][2]) / 2 - page_width / 2) < 20
            
            block['word_count'] = len(block['text'].split())
            block['starts_with_numbering'] = bool(re.match(r'^\s*(\d+(\.\d+)*|[A-Za-z])\.\s+', block['text']))
            block['text_case'] = 'UPPER' if block['text'].isupper() else 'Title' if block['text'].istitle() else 'Other'
            
        return text_blocks