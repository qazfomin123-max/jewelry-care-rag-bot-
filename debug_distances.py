"""
Диагностика расстояний ChromaDB для калибровки RAG_MAX_DISTANCE.

Запуск:
    py debug_distances.py

Смотри на разброс:
- профильные вопросы должны давать явно МЕНЬШИЕ distance, чем контрольный
  вопрос не по теме;
- вопросы про серебро НЕ должны стабильно давать бОльшие distance до
  serebro.txt, чем вопросы про золото до zoloto.txt — если это так,
  проблема с префиксами/индексом ещё не решена.
"""

import rag_core

# Подставь сюда реальные имена своих .txt файлов в knowledge_base/,
# чтобы удобно было сверять "ожидаемый источник" глазами.
TEST_QUESTIONS = [
    # (вопрос, ожидаемый источник — для твоего удобства при чтении вывода)
    ("Как почистить золотое кольцо в домашних условиях?", "золото"),
    ("Можно ли носить золото каждый день?", "золото"),
    ("Как ухаживать за серебряным украшением?", "серебро"),
    ("Почему серебро темнеет и как это остановить?", "серебро"),
    ("Как проверить подлинность бриллианта?", "бриллианты"),
    ("Что означает проба 585 на украшении?", "клеймо/проба"),
    ("Как правильно определить размер кольца?", "размер кольца"),
    # контрольный вопрос НЕ по теме базы — должен давать самые большие distance
    ("Как починить сломанные наручные часы?", "— вне базы, контроль"),
]


def debug_distances():
    print("Загружаю коллекцию...")
    collection = rag_core.load_collection()
    print(f"В базе {collection.count()} кусков.\n")

    for question, expected in TEST_QUESTIONS:
        # Важно: тот же префикс, что и в retrieve_relevant_chunks
        results = collection.query(
            query_texts=[f"query: {question}"],
            n_results=5,
        )

        print(f"Вопрос: {question}")
        print(f"  (ожидаемая тема: {expected})")

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            source = metadata.get("source", "?")
            preview = text[:50].replace("\n", " ")
            print(f"    dist={distance:.4f} | {source:20s} | {preview}...")

        print()


if __name__ == "__main__":
    debug_distances()
