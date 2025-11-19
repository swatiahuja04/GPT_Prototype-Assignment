# GPT - Assignment

This repository contains a simple RAG (Retrieval-Augmented Generation) Q&A system over a short Ambedkar speech using:

- LangChain 1.0.5 (modular)
- ChromaDB (local vector store)
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Ollama (phi3:mini recommended for low-RAM Systems)

## Files
- `main.py` - runnable CLI script
- `speech.txt` - provided speech text (place in same folder)

## Setup
1. Install Ollama (https://ollama.com) and pull a small model:

ollama pull phi3:mini

2. Create & activate your conda environment (if using conda):

conda create -n ambedkar-env python=3.12 -y
conda activate ambedkar-env


3. Or using pip/venv:

python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate


4. Run the script:

python main.py


## Notes
- Ensure `speech.txt` is in the same folder as `main.py`.
- If your machine has very low RAM, use `phi3:mini` or `mistral:tiny` rather than full `mistral`.

