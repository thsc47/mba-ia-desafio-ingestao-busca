import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Configurações do ambiente
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")


def ingest_pdf():
    """
    Lê um arquivo PDF, divide em chunks e salva no banco de dados PostgreSQL com pgVector.

    Processo:
    1. Carrega o PDF usando PyPDFLoader
    2. Divide o texto em chunks de 1000 caracteres com overlap de 150
    3. Cria embeddings usando Google Generative AI
    4. Armazena os vetores no PostgreSQL com pgVector
    """
    try:
        print(f"Iniciando ingestão do PDF: {PDF_PATH}")

        # 1. Carregar o PDF
        print("Carregando PDF...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"PDF carregado com sucesso! Total de páginas: {len(documents)}")

        # 2. Dividir em chunks
        print("Dividindo documento em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Documento dividido em {len(chunks)} chunks")

        # 3. Criar embeddings e armazenar no banco
        print("Criando embeddings e armazenando no banco de dados...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        # Criar ou substituir a coleção no PGVector
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )

        # Adicionar documentos
        vectorstore.add_documents(chunks)

        print(f"✓ Ingestão concluída com sucesso!")
        print(f"✓ {len(chunks)} chunks foram armazenados no banco de dados")
        print(f"✓ Coleção: {COLLECTION_NAME}")

    except FileNotFoundError:
        print(f"Erro: Arquivo PDF não encontrado: {PDF_PATH}")
        print("Verifique se o caminho está correto no arquivo .env")
    except Exception as e:
        print(f"Erro durante a ingestão: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ingest_pdf()