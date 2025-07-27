# create_training_data.py

import pandas as pd
from pathlib import Path
import json

# Import the agents you already created
from agents.parsing_agent import PDFParsingAgent
from agents.feature_agent import FeatureEngineeringAgent

def create_training_data():
    """
    Automatically creates a training CSV file from the sample dataset.
    """
    # Define paths
    pdf_dir = Path("sample_dataset/pdfs")
    output_dir = Path("sample_dataset/outputs")
    
    # Check if directories exist
    if not pdf_dir.exists() or not output_dir.exists():
        print("Error: 'sample_dataset/pdfs' or 'sample_dataset/outputs' directory not found.")
        print("Please make sure you are running this from the 'challenge_1a' root directory.")
        return

    # Instantiate the agents we need
    parser = PDFParsingAgent()
    feature_engineer = FeatureEngineeringAgent()

    all_labeled_blocks = []

    # Process each PDF and its corresponding JSON output
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name} for training data...")
        json_file = output_dir / f"{pdf_file.stem}.json"

        if not json_file.exists():
            print(f"Warning: No corresponding JSON output found for {pdf_file.name}. Skipping.")
            continue

        # Load the ground truth data
        with open(json_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Create a quick lookup for true headings { (text, page): level }
        true_headings = { (item['text'], item['page']): item['level'] for item in ground_truth.get('outline', []) }
        
        # Use our agents to get featured blocks from the PDF
        raw_blocks = parser.extract_text_blocks(str(pdf_file))
        featured_blocks = feature_engineer.create_features(raw_blocks)

        # Label each block
        for block in featured_blocks:
            label = "Other"
            block_key = (block['text'], block['page'])
            
            # Check if this block is a known heading
            if block_key in true_headings:
                label = true_headings[block_key]
            # Check if this block is the title
            elif block['text'] == ground_truth.get('title'):
                label = "Title"

            block['label'] = label
            all_labeled_blocks.append(block)

    # Create a DataFrame and save to CSV
    if not all_labeled_blocks:
        print("No data was processed. Could not create training file.")
        return

    df = pd.DataFrame(all_labeled_blocks)
    
    # Select only the columns needed for training
    columns_to_keep = [
        'text', 'font_size', 'is_bold', 'is_centered', 'word_count', 
        'starts_with_numbering', 'text_case', 'label'
    ]
    # Filter out any rows where these columns might be missing
    df_final = df[columns_to_keep].dropna()

    df_final.to_csv("train.csv", index=False)
    print("\nSuccessfully created train.csv with all sample data!")

if __name__ == '__main__':
    create_training_data()