from sentence_transformers import SentenceTransformer

def get_embeddings(text):
    embedding_path = r"C:\Users\USER\Desktop\ML\LLM\RAG Test\all-MiniLM-L6-v2"
    model = SentenceTransformer(embedding_path)
    return model.encode(text)