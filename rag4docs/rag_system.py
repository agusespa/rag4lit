import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import prep
import gen_llm
from tqdm import tqdm


class RAGSystem:
    def __init__(
        self,
    ):
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.vector_store = None
        self.texts = []
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        print("Initializing RAG knowledge base...")

        documents = prep.load_documents()
        if not documents:
            return

        all_chunks = []
        for doc in documents:
            chunk = prep.get_tokenized_chunks(doc, self.embedding_model.tokenizer)
            all_chunks.extend(chunk)

        print(f"Generated {len(all_chunks)} chunks from documents.")

        print("Generating embeddings for chunks...")
        embeddings = []

        batch_size = 32

        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding Chunks"):
            batch_chunks = all_chunks[i : i + batch_size]
            batch_embeddings = prep.embed_chunks(batch_chunks, self.embedding_model)
            if batch_embeddings is not None:
                embeddings.extend(batch_embeddings)
                self.texts.extend(batch_chunks)

        if not embeddings:
            print("No embeddings were generated. Knowledge base not initialized.")
            return

        embeddings_np = np.array(embeddings).astype("float32")
        emb_dim = embeddings_np.shape[1]

        self.vector_store = faiss.IndexFlatL2(emb_dim)
        self.vector_store.add(embeddings_np)

        print(f"\nKnowledge base initialized with {len(self.texts)} chunks.")

    def retrieve_with_scores(self, q_embedding, top_k=5):
        if self.vector_store is None:
            print("Vector store not initialized. Cannot retrieve.")
            return [], []

        if q_embedding.ndim == 1:
            q_embedding = q_embedding.reshape(1, -1)
        q_embedding = q_embedding.astype("float32")

        scores, I = self.vector_store.search(q_embedding, top_k)
        chunks = [self.texts[i] for i in I[0]]
        return chunks, scores[0].tolist()

    def build_prompt(self, query, docs):
        context = "\n\n".join(docs)
        return f"""
            <role>You are the Lead Developer for the software development project and your main purpose is to help other developers understand the system.</role>
            <task>Answer the question based on the following context. If the answer is not available in the context, state that you don't have enough information.</task>
            <context>{context}</context>
            <question>{query}</question>
        """

    def query(
        self,
        user_question: str,
        top_k_retrievals: int = 3,
        similarity_threshold: float = 1.0,
    ) -> str | None:
        print(f"\nUser question: {user_question}")
        try:
            query_for_embedding = (
                "Represent this sentence for searching relevant passages: "
                + user_question
            )
            query_embedding = self.embedding_model.encode(
                [query_for_embedding],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return

        retrieved_chunks, scores = self.retrieve_with_scores(
            query_embedding, top_k=top_k_retrievals
        )

        filtered_chunks = []
        for chunk, score in zip(retrieved_chunks, scores):
            if score <= similarity_threshold:
                filtered_chunks.append((chunk, score))

        if not filtered_chunks:
            print("No relevant chunks found in the knowledge base.")
            return

        print("\n--- Retrieved Chunks ---")
        for i, (chunk, score) in enumerate(filtered_chunks):
            print(f"Chunk {i+1} (similarity: {score:.3f}):\n{chunk}\n")
        print("------------------------\n")

        print("Generating response with LLM...")
        augmented_prompt = self.build_prompt(user_question, retrieved_chunks)
        llm_response = gen_llm.generate_response(augmented_prompt)
        return llm_response
