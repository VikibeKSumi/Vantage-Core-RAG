from src.engine_load import RAGEngine


if __name__ == "__main__":
    
    engine = RAGEngine()
    exit_cmd = ['exit', 'quit']
    while True:
        q = input("Enter query (type 'exit' or 'quit' end): ").strip()
        if q.lower() in exit_cmd:
            print("👋 Goodbye!")
            break
        engine.ask(q)
