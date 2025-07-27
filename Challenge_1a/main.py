# main.py
import json
from pathlib import Path
import time

# Import agent classes from the agents package
from agents.parsing_agent import PDFParsingAgent
from agents.feature_agent import FeatureEngineeringAgent
from agents.classification_agent import HeadingClassificationAgent
from agents.output_agent import JSONOutputAgent

def process_pipeline(pdf_path: Path) -> dict:
    """
    Runs the full agentic pipeline for a single PDF file.
    """
    # 1. Instantiate Agents
    parser = PDFParsingAgent()
    feature_engineer = FeatureEngineeringAgent()
    classifier = HeadingClassificationAgent(model_path="model.txt")
    formatter = JSONOutputAgent()

    # 2. Run Pipeline
    raw_blocks = parser.extract_text_blocks(str(pdf_path))
    featured_blocks = feature_engineer.create_features(raw_blocks)
    classified_blocks = classifier.predict(featured_blocks)
    json_output = formatter.format_output(classified_blocks)
    
    return json_output

def main():
    """
    Main function to process all PDFs from input to output directory.
    """
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in /app/input.")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        start_time = time.time()
        
        result_json = process_pipeline(pdf_file)
        
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)
        
        duration = time.time() - start_time
        print(f"Finished {pdf_file.name} in {duration:.2f} seconds. Output saved to {output_file.name}.")

if __name__ == '__main__':
    main()