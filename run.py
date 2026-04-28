from src.engine import Engine
from loguru import logger



if __name__ == "__main__":

    engine = Engine()
    while True:
        
        query = input("type 'exit' to quit: ")
        if query.lower() == 'exit':
            break
        result = engine.run(query)
        
        for keys, value in result.items():
          print(f"{keys}: {value}")

    logger.info(".....Engine Close.......")
