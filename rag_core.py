import os
import re

import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("TOOKEN_API_KEY")

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "jewelry_care_kb"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Настраиваются через .env, чтобы можно было подбирать без правки кода.
TOP_K = int(os.environ.get("RAG_TOP_K", 9))
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", 0.8))


def load_collection():
    if not os.path.isdir(CHROMA_DB_DIR):
        raise RuntimeError(
            f"Не найдена база {CHROMA_DB_DIR}/. "
            "Сначала запусти: python build_index.py"
        )

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )
    except Exception:
        raise RuntimeError(
            f"Коллекция '{COLLECTION_NAME}' не найдена. "
            "Сначала запусти: python build_index.py"
        )

    return collection


def retrieve_relevant_chunks(collection, question: str, top_k: int = TOP_K,
                              max_distance: float = MAX_DISTANCE) -> list[dict]:
    results = collection.query(
        query_texts=[f"query: {question}"],
        n_results=top_k,)

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        if distance > max_distance:
            continue
        chunks.append({
            "text": text,
            "source": metadata.get("source", "неизвестный источник"),
            "distance": distance,
        })

    return chunks


def build_system_prompt(chunks: list[dict]) -> str:
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Источник: {chunk['source']}]\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_parts)

    return f"""Ты — консультант по уходу за ювелирными изделиями. Отвечаешь
живым, дружелюбным языком, как знающий продавец в хорошем магазине,
а не как справочник. Пиши простым текстом, без markdown — не используй
звёздочки, решётки и другое форматирование, только обычные предложения.

Вот фрагменты базы знаний по теме вопроса — используй их как основу
ответа. Приводи конкретику оттуда (способы, пропорции, сроки), а не
общие фразы вроде "ухаживайте бережно". Если фрагменты закрывают
вопрос только частично — ответь тем, что есть, и по-простому скажи,
чего не хватает. Если по теме вопроса вообще ничего релевантного нет —
так и скажи, не выдумывай.

Фрагменты базы знаний:

{context_text}
"""


def strip_markdown(text: str) -> str:
    """
    Убирает базовую markdown-разметку из ответа модели (жирный, курсив,
    заголовки), чтобы текст нормально смотрелся в Telegram без parse_mode.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **жирный**
    text = re.sub(r"__(.+?)__", r"\1", text)        # __жирный__
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"\1", text)  # *курсив*
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # # заголовки
    return text


def ask_claude(system_prompt: str, question: str) -> str:
    try:
        response = requests.post(
            "https://tooken.club/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 700,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": question}
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print("Ошибка сети/API:", e)
        return "Не получилось связаться с сервером, попробуйте ещё раз."
    except ValueError:
        print("Ответ сервера не является JSON:", response.text[:200])
        return "Сервер вернул некорректный ответ."

    if "content" not in result:
        print("Ошибка API:", result)
        return "Произошла ошибка при обращении к API."

    answer = result["content"][0]["text"]
    return strip_markdown(answer)


def ask_rag(collection, question: str) -> str:
    """
    Полный RAG-цикл на один вопрос: поиск -> сборка промпта -> запрос к Claude.
    """
    chunks = retrieve_relevant_chunks(collection, question)

    if not chunks:
        return "У меня нет информации по этому вопросу в базе знаний."

    system_prompt = build_system_prompt(chunks)
    return ask_claude(system_prompt, question)
