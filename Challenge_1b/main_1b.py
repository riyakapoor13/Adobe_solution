# main_1b.py

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import agents
from agents.parsing_agent import PDFParsingAgent
from agents_1b.embedding_agent import EmbeddingAgent
from agents_1b.ranking_agent import RankingAgent

# --- Configuration Values (previously in config.py) ---
MODEL_NAME = 'all-MiniLM-L6-v2'
RELEVANCE_THRESHOLD = 0.3
TOP_N_RESULTS = 15
MAX_WORKERS = 8
# ---------------------------------------------------------

def parse_pdf_worker(pdf_path: Path, parser: PDFParsingAgent) -> list:
    """Worker function to parse a single PDF."""
    print(f"Parsing: {pdf_path.name}")
    chunks = parser.extract_text_blocks(str(pdf_path))
    for chunk in chunks:
        chunk['document'] = pdf_path.name
        chunk['section_title'] = chunk['text'].split('\n')[0]
    return chunks

def run_pipeline(doc_paths: list, persona: str, job_to_be_done: str):
    """Main pipeline for Challenge 1B."""
    start_time = time.time()

    # --- Stage 1: Parallel Document Parsing ---
    all_chunks = []
    pdf_parser = PDFParsingAgent()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {executor.submit(parse_pdf_worker, pdf_path, pdf_parser): pdf_path for pdf_path in doc_paths}
        for future in as_completed(future_to_pdf):
            all_chunks.extend(future.result())

    # --- Stage 2: Embedding Generation ---
    embedding_agent = EmbeddingAgent(MODEL_NAME)
    ranking_agent = RankingAgent()

    query = f"Persona: {persona}. Task: {job_to_be_done}"
    chunk_texts = [chunk['text'] for chunk in all_chunks]

    query_embedding = embedding_agent.create_query_embedding(query)
    chunk_embeddings = embedding_agent.create_embeddings(chunk_texts)

    # --- Stage 3: Relevance Ranking ---
    relevance_scores = ranking_agent.calculate_relevance_scores(query_embedding, chunk_embeddings)
    for i, chunk in enumerate(all_chunks):
        chunk['relevance_score'] = relevance_scores[i]

    # Filter and sort chunks
    relevant_chunks = [c for c in all_chunks if c['relevance_score'] >= RELEVANCE_THRESHOLD]
    ranked_chunks = sorted(relevant_chunks, key=lambda x: x['relevance_score'], reverse=True)

    # --- Stage 4: JSON Formatting ---
    output = {
        "metadata": {
            "input_documents": [path.name for path in doc_paths],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": time.ctime()
        },
        "Extracted Section": []
    }

    for rank, chunk in enumerate(ranked_chunks[:TOP_N_RESULTS]):
        output["Extracted Section"].append({
            "Document": chunk['document'],
            "Page number": chunk['page'],
            "Section title": chunk['section_title'],
            "Importance_rank": rank + 1,
            "Relevance Score": float(chunk['relevance_score']),
            "Text Snippet": chunk['text'][:200] + "..."
        })
        
    print(f"\n✅ Pipeline finished in {time.time() - start_time:.2f}s")
    return output

if __name__ == '__main__':
    input_pdf_dir = Path("sample_dataset/pdfs_1b")
    document_collection = list(input_pdf_dir.glob("*.pdf"))

    if not document_collection:
        print(f"Error: No PDF files found in '{input_pdf_dir}'.")
    else:
        persona_1 = "PhD Researcher in Computational Biology"
        job_1 = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for Graph Neural Networks in drug discovery."

        final_output = run_pipeline(document_collection, persona_1, job_1)

        with open("challenge1b_output.json", "w") as f:
            json.dump(final_output, f, indent=4)
        print("\nSuccessfully created challenge1b_output.json")