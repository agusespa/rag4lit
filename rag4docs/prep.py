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


def get_chunks(text):
    sentence_chunks = split_into_sentence_chunks(text, 1000)

    return sentence_chunks


def embed_chunks(texts, model):
    if not texts:
        return None

    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )

    return embeddings
