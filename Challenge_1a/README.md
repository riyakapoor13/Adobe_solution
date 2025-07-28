# Adobe India Hackathon 2025: Connecting the Dots - Round 1A

Project: Document Outline Extractor
Team: Powerpuff Girls

## 1. Project Overview

This repository contains our solution for Round 1A of the Adobe India Hackathon, "Connecting the Dots." [cite_start]Our mission was to build a high-performance, intelligent system to extract structured outlines from PDF documents. [cite: 26, 30] [cite_start]This solution serves as the foundational "brains" for enabling smarter, more interactive document experiences. [cite: 34]

[cite_start]Our system accepts a PDF file of up to 50 pages and outputs a structured JSON file containing the document's title and a hierarchical list of all H1, H2, and H3 headings with their corresponding page numbers. [cite: 38, 42, 43]

## 2. Our Solution: An Agentic, Rules-Based Pipeline

To build a solution that is both accurate and modular, we implemented a multi-agent AI pipeline. [cite_start]Each agent in the pipeline has a single, well-defined responsibility, making the system robust and easy to maintain, as suggested for future rounds. [cite: 97]

[cite_start]Our approach specifically avoids relying solely on font sizes, incorporating additional heuristics to handle complex document layouts, as recommended in the challenge's Pro Tips. [cite: 94]

* Agent 1: PDF Parsing Agent
    * This agent ingests the raw PDF file. It uses the PyMuPDF library to efficiently parse each page and extract all text blocks along with rich metadata, including font properties (name, size, weight) and precise coordinates (bounding boxes).

* Agent 2: Feature Engineering Agent
    * This agent enriches the raw data from the parser with intelligent features. For each text block, it calculates typographical, positional, and content-based attributes. Key features include relative font size (compared to the page's median), horizontal centering, and pattern matching for numbered headings.

* Agent 3: Heading Classification Agent
    * This is the core logic engine. It uses a robust set of heuristics to classify each text block as a Title, H1, H2, or H3. The classification decision is made by weighing multiple features from the previous agent, resulting in a more accurate prediction than a simple font-size check.

* Agent 4: JSON Output Agent
    * [cite_start]The final agent takes the classified headings, sorts them by page and position, and constructs the final JSON output, ensuring it strictly conforms to the required schema. [cite: 44, 45, 46, 47, 48, 49]

## 3. Tech Stack & Libraries

* Primary Language: Python
* Key Libraries:
    * PyMuPDF: For high-speed PDF parsing.
    * pandas, scikit-learn, lightgbm: Included to support the machine learning framework.
* Containerization: Docker

## 4. Performance & Constraints

Our solution is engineered to be lightweight and fast, fully complying with the challenge constraints.

* [cite_start]Execution Time: Processes a 50-page PDF in well under the 10-second time limit. [cite: 78]
* [cite_start]Offline Operation: The entire pipeline runs without any network or internet calls. [cite: 60, 62]
* [cite_start]Environment: The solution is packaged in a Docker container built for the linux/amd64 architecture, with no GPU dependencies. [cite: 56, 58]

## 5. How to Build and Run

The project is fully containerized and easy to run.

1. Build the Docker Image

From the root directory, run the following command to build the image:
```bash
docker build --platform linux/amd64 -t doc-parser .
2. Run the Container

To process PDFs, mount an input and output directory and run the container. The command below uses the sample data provided in the challenge. 

Bash

docker run --rm -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" -v "$(pwd)/sample_dataset/outputs:/app/output" --network none doc-parser

The application will automatically process all PDFs found in 

/app/input and generate the corresponding .json files in /app/output.