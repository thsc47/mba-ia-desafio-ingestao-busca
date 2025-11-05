from search import search_prompt
import sys


def print_separator():
    """Imprime uma linha separadora."""
    print("\n" + "=" * 70 + "\n")


def main():
    """
    Interface CLI para interação com o sistema de busca baseado em PDF.

    O usuário pode fazer perguntas e receber respostas baseadas no conteúdo
    do PDF que foi previamente processado.
    """
    print("=" * 70)
    print("  Sistema de Busca em PDF com RAG (Retrieval-Augmented Generation)")
    print("=" * 70)
    print("\nInicializando sistema...")

    # Inicializar a chain de busca
    chain = search_prompt()

    if not chain:
        print("\nErro: Não foi possível iniciar o chat.")
        print("Verifique se:")
        print("  1. O arquivo .env está configurado corretamente")
        print("  2. O banco de dados PostgreSQL está rodando")
        print("  3. O PDF foi ingerido (execute: python src/ingest.py)")
        print("  4. A GOOGLE_API_KEY está configurada")
        return

    print("✓ Sistema inicializado com sucesso!")
    print("\nDigite suas perguntas ou 'sair' para encerrar.")
    print_separator()

    # Loop principal do chat
    while True:
        try:
            # Solicitar pergunta do usuário
            pergunta = input("PERGUNTA: ").strip()

            # Verificar comando de saída
            if pergunta.lower() in ["sair", "exit", "quit", "q"]:
                print("\nEncerrando o chat. Até logo!")
                break

            # Ignorar entradas vazias
            if not pergunta:
                print("Por favor, digite uma pergunta válida.")
                continue

            # Buscar e gerar resposta
            print("\nProcessando...")
            resposta = chain.invoke({"pergunta": pergunta})

            # Exibir resposta
            print(f"\nRESPOSTA: {resposta}")
            print_separator()

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário. Encerrando...")
            break
        except Exception as e:
            print(f"\nErro ao processar pergunta: {str(e)}")
            print("Tente novamente ou digite 'sair' para encerrar.")
            print_separator()


if __name__ == "__main__":
    main()