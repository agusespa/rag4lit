import os
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


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


def split_into_sentence_chunks(text, max_chars):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_text_by_tokens(text, tokenizer, max_tokens=500, overlap=50):
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += max_tokens - overlap

    return chunks


def get_tokenized_chunks(text, tokenizer, token_limit=500, token_overlap=50):
    sentence_chunks = split_into_sentence_chunks(text, 1000)
    final_chunks = []

    for chunk in sentence_chunks:
        token_chunks = chunk_text_by_tokens(
            chunk,
            tokenizer,
            max_tokens=token_limit,
            overlap=token_overlap,
        )
        final_chunks.extend(token_chunks)

    return final_chunks


def embed_chunks(texts, model):
    if not texts:
        return None

    embeddings = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings
