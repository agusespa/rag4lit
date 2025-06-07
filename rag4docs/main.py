from rag_system import RAGSystem

if __name__ == "__main__":
    rag_system = RAGSystem()

    query = "Where can I find historical meteor shower data?"

    answer = rag_system.query(query)
    print(answer)
