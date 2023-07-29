# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    text_loader_kwargs={'autodetect_encoding': True}

    loader = DirectoryLoader(cfg.DATA_PATH,
                             glob='*.txt',
                             loader_cls=TextLoader, 
                             loader_kwargs=text_loader_kwargs)
    documents = loader.load()

    lencorpus = len(documents)
    print(lencorpus)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

if __name__ == "__main__":
    run_db_build()
