import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Configurações do ambiente
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(question=None):
    """
    Cria uma chain do LangChain para buscar e responder perguntas baseadas no PDF.

    Se uma pergunta for fornecida, executa a busca e retorna a resposta.
    Se não for fornecida pergunta, retorna a chain configurada.

    Args:
        question: A pergunta do usuário (opcional)

    Returns:
        Se question for None: retorna a chain configurada
        Se question for fornecida: retorna a resposta (string)
    """
    try:
        # Configurar embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        # Conectar ao vectorstore
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )

        # Configurar LLM (Gemini 2.5 Flash Lite)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,  # Respostas mais determinísticas
        )

        # Criar o prompt template
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # Função para buscar e formatar os documentos recuperados
        def get_context(input_dict):
            """Busca documentos relevantes e retorna o contexto formatado."""
            question = input_dict["pergunta"]
            # similarity_search_with_score retorna lista de tuplas (Document, score)
            results = vectorstore.similarity_search_with_score(question, k=10)
            # Extrair apenas os documentos (ignorar scores)
            docs = [doc for doc, score in results]
            # Concatenar o conteúdo dos documentos
            return "\n\n".join(doc.page_content for doc in docs)

        # Criar a chain
        chain = (
            {
                "contexto": get_context,
                "pergunta": lambda x: x["pergunta"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Se uma pergunta foi fornecida, executar a chain
        if question:
            return chain.invoke({"pergunta": question})

        # Caso contrário, retornar a chain para uso posterior
        return chain

    except Exception as e:
        print(f"Erro ao configurar a busca: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def search_documents(question: str, k: int = 10):
    """
    Busca documentos relevantes no vectorstore sem gerar resposta.

    Args:
        question: A pergunta/query de busca
        k: Número de documentos a retornar (padrão: 10)

    Returns:
        Lista de tuplas (documento, score)
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )

        results = vectorstore.similarity_search_with_score(question, k=k)
        return results

    except Exception as e:
        print(f"Erro ao buscar documentos: {str(e)}")
        import traceback
        traceback.print_exc()
        return []