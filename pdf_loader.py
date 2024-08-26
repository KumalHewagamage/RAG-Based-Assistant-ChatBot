from langchain_community.document_loaders import PyPDFLoader
from embeddings import get_embeddings

def load_pdf(file_path):

    loader = PyPDFLoader(file_path)
    chunk = loader.load_and_split()
    return chunk


#print(load_pdf(r"C:\Users\USER\Desktop\ML\LLM\RAG Test\P.pdf"))

chunks = load_pdf(r"C:\Users\USER\Desktop\ML\LLM\RAG Test\P.pdf")

#print(chunks[1].page_content)
#print(get_embeddings(chunks[1].page_content))
vec_store=[]

for splits in chunks:
    vec_store.append(get_embeddings(splits.page_content))

print(vec_store)