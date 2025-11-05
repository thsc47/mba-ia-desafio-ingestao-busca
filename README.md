# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de busca em PDF usando RAG (Retrieval-Augmented Generation) com LangChain, PostgreSQL + pgVector e Google Gemini.

## Descrição

Este projeto implementa um sistema completo de ingestão e busca de documentos PDF utilizando:
- **LangChain 0.3**: Framework para construção de aplicações com LLMs
- **PostgreSQL + pgVector**: Banco de dados vetorial para armazenar embeddings
- **Google Gemini**: Modelo de embeddings e LLM para geração de respostas
- **RAG (Retrieval-Augmented Generation)**: Técnica para responder perguntas baseadas no conteúdo do PDF

## Estrutura do Projeto

```
├── docker-compose.yml         # Configuração do PostgreSQL + pgVector
├── requirements.txt           # Dependências Python
├── .env.example              # Template de variáveis de ambiente
├── .env                      # Variáveis de ambiente (não versionar)
├── src/
│   ├── ingest.py            # Script de ingestão do PDF
│   ├── search.py            # Funções de busca e resposta
│   ├── chat.py              # Interface CLI interativa
├── document.pdf             # PDF para ingestão
└── README.md                # Este arquivo
```

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Conta Google Cloud com API Key do Gemini

## Instalação

### 1. Clonar o repositório

```bash
git clone <seu-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar ambiente virtual Python

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Copie o arquivo `.env.example` para `.env` e configure sua API Key do Google:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione sua GOOGLE_API_KEY:

```env
GOOGLE_API_KEY=sua_api_key_aqui
GOOGLE_EMBEDDING_MODEL=models/embedding-001
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=pdf_documents
PDF_PATH=document.pdf
```

**Como obter a API Key do Google Gemini:**
1. Acesse [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crie uma nova API Key
3. Cole a chave no arquivo `.env`

### 5. Adicionar seu PDF

Coloque o arquivo PDF que deseja processar na raiz do projeto com o nome `document.pdf`, ou ajuste o caminho no arquivo `.env`.

## Execução

### 1. Subir o banco de dados PostgreSQL

```bash
docker compose up -d
```

Aguarde o banco de dados estar pronto (healthcheck). Você pode verificar com:

```bash
docker compose ps
```

### 2. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Este script irá:
- Carregar o PDF
- Dividir em chunks de 1000 caracteres com overlap de 150
- Gerar embeddings usando Google Gemini
- Armazenar no PostgreSQL com pgVector

**Saída esperada:**
```
Iniciando ingestão do PDF: document.pdf
Carregando PDF...
PDF carregado com sucesso! Total de páginas: X
Dividindo documento em chunks...
Documento dividido em Y chunks
Criando embeddings e armazenando no banco de dados...
✓ Ingestão concluída com sucesso!
✓ Y chunks foram armazenados no banco de dados
✓ Coleção: pdf_documents
```

### 3. Executar o chat interativo

```bash
python src/chat.py
```

**Exemplo de uso:**

```
======================================================================
  Sistema de Busca em PDF com RAG (Retrieval-Augmented Generation)
======================================================================

Inicializando sistema...
✓ Sistema inicializado com sucesso!

Digite suas perguntas ou 'sair' para encerrar.

======================================================================

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?

Processando...

RESPOSTA: O faturamento foi de 10 milhões de reais.

======================================================================

PERGUNTA: Quantos clientes temos em 2024?

Processando...

RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

======================================================================

PERGUNTA: sair

Encerrando o chat. Até logo!
```

## Como funciona

### Ingestão (ingest.py)

1. **Carregamento**: Usa `PyPDFLoader` para ler o PDF
2. **Chunking**: Divide o texto em pedaços usando `RecursiveCharacterTextSplitter`
   - Tamanho do chunk: 1000 caracteres
   - Overlap: 150 caracteres
3. **Embeddings**: Gera vetores usando `GoogleGenerativeAIEmbeddings` (modelo: embedding-001)
4. **Armazenamento**: Salva no PostgreSQL usando `PGVector`

### Busca (search.py)

1. **Vetorização**: Converte a pergunta do usuário em embedding
2. **Busca por similaridade**: Encontra os 10 chunks mais relevantes (k=10)
3. **Prompt Engineering**: Monta um prompt com contexto e regras rígidas
4. **Geração**: Usa Gemini 2.0 Flash para gerar resposta baseada apenas no contexto
5. **Validação**: Retorna "Não tenho informações necessárias" se não encontrar resposta no contexto

### Chat (chat.py)

Interface CLI interativa que permite ao usuário fazer perguntas continuamente até digitar "sair".

## Tecnologias Utilizadas

- **Python 3.11+**
- **LangChain 0.3**: Framework para aplicações com LLMs
- **PostgreSQL 17**: Banco de dados relacional
- **pgVector**: Extensão para busca vetorial
- **Google Gemini**:
  - `models/embedding-001`: Para embeddings
  - `gemini-2.0-flash-exp`: Para geração de respostas
- **PyPDF**: Para leitura de arquivos PDF
- **python-dotenv**: Para gerenciamento de variáveis de ambiente

## Troubleshooting

### Erro ao conectar no banco de dados

Verifique se o PostgreSQL está rodando:
```bash
docker compose ps
```

Se não estiver, inicie com:
```bash
docker compose up -d
```

### Erro de API Key

Certifique-se de que:
1. O arquivo `.env` existe e está configurado
2. A `GOOGLE_API_KEY` é válida
3. A API do Gemini está habilitada na sua conta Google Cloud

### PDF não encontrado

Verifique se:
1. O arquivo PDF existe no caminho especificado
2. O `PDF_PATH` no `.env` está correto
3. Use caminho absoluto se necessário

### Erro "No documents found"

Execute a ingestão primeiro:
```bash
python src/ingest.py
```

## Parar o ambiente

Para parar o banco de dados:

```bash
docker compose down
```

Para remover também os volumes (dados):

```bash
docker compose down -v
```

## Licença

MIT

## Autor

Desenvolvido como parte do MBA em Engenharia de Software com IA - Full Cycle