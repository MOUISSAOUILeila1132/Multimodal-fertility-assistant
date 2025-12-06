import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings #pour creer un index vectoriel pour la recherche semantique 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core.node_parser import SentenceSplitter

PDF_DIR = "/content/tanit-multimodal-fertility-assistant/data"
#ou sauvegarder l'index vectoriel 
INDEX_DIR = "/content/tanit-multimodal-fertility-assistant/rag/graphrag_index"

def build_knowledge_base():
    print(" Initialisation embedding...")
    # modele d'embeddings utilisé par llama_index
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
    Settings.llm = None
    # Verification de l'existance de pdfs
    if not os.path.exists(PDF_DIR) or not os.listdir(PDF_DIR):
        print(f" Aucun PDF trouvé : {PDF_DIR}")
        return
    # chargement des pdfs 
    documents = SimpleDirectoryReader(PDF_DIR).load_data()
    print(f" {len(documents)} pages chargées.")
    #configuration de decoupages des chunks 
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    #creation de l'index vectoriel 
    index = VectorStoreIndex.from_documents(documents, transformations=[parser])
    #sauvegarde de l'index vectoriel 
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f" Index créé dans {INDEX_DIR}")

if __name__ == "__main__":
    build_knowledge_base()
