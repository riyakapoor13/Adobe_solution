# agents/output_agent.py

class JSONOutputAgent:
    """
    Formats the classified headings into the required JSON structure.
    """
    def format_output(self, classified_blocks: list) -> dict:
        title = "Default Title - No Title Found"
        title_blocks = [b for b in classified_blocks if b['level'] == 'Title']
        if title_blocks:
            # Assume the first title found is the main one
            title = title_blocks[0]['text']

        outline = []
        heading_blocks = [b for b in classified_blocks if b['level'] in ["H1", "H2", "H3"]]

        # Create a dictionary to easily check for duplicates
        seen = set()
        for block in heading_blocks:
            # Use text and page to identify a unique heading
            identifier = (block['text'], block['page'])
            if identifier not in seen:
                outline.append({
                    "level": block['level'],
                    "text": block['text'],
                    "page": block['page']
                })
                seen.add(identifier)
        
        return {"title": title, "outline": outline}