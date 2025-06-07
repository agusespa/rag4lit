import os

PATH = "./knowledge_base/raw/"


def load_documents():
    documents = []
    if not os.path.exists(PATH):
        print(f"Error: Directory '{PATH}' does not exist.")
        return []
    for filename in os.listdir(PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(PATH, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return documents


def chunk_text_by_tokens(
    text,
    tokenizer,
    max_tokens=500,
    overlap=50,
):
    tokens = tokenizer.tokenize(
        text, truncation=True, max_length=tokenizer.model_max_length
    )
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)

        if end == len(tokens):
            break

        move_by = max(max_tokens - overlap, 1)
        start += move_by

    return chunks


def embed_chunks(texts, model):
    if not texts:
        return None

    embeddings = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings
