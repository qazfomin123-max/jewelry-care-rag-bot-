import logging
import os
import re
import time

import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("TOOKEN_API_KEY")
# URL и модель вынесены в .env — если прокси сменит адрес или маппинг
# моделей, не придётся трогать код.
API_URL = os.environ.get("TOOKEN_API_URL", "https://tooken.club/v1/messages")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "jewelry_care_kb"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Настраиваются через .env, чтобы можно было подбирать без правки кода.
TOP_K = int(os.environ.get("RAG_TOP_K", 6))
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", 0.163))

# Ретраи на транзиентные сбои сети/прокси (не путать с ошибками самого
# ответа модели — тех мы не ретраим).
CLAUDE_MAX_RETRIES = int(os.environ.get("CLAUDE_MAX_RETRIES", 2))
CLAUDE_RETRY_BACKOFF_SECONDS = float(os.environ.get("CLAUDE_RETRY_BACKOFF_SECONDS", 1))

# Circuit breaker: если подряд накопилось CIRCUIT_BREAKER_THRESHOLD
# сбоев, следующие CIRCUIT_BREAKER_COOLDOWN_SECONDS секунд не бьёмся в
# прокси заново на каждое сообщение (там timeout=60с — при падении
# прокси это подвешивает ответ каждому пользователю на минуту), а сразу
# отвечаем "сервис недоступен".
CIRCUIT_BREAKER_THRESHOLD = int(os.environ.get("CIRCUIT_BREAKER_THRESHOLD", 3))
CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(
    os.environ.get("CIRCUIT_BREAKER_COOLDOWN_SECONDS", 60)
)

_consecutive_failures = 0
_circuit_open_until = 0.0


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
    # Модели семейства E5 обучены с разными префиксами для запроса и для
    # текста базы ("query: " / "passage: ") — без них поиск хуже
    # различает близкие темы. Симметричный префикс — в build_index.py.
    results = collection.query(
        query_texts=[f"query: {question}"],
        n_results=top_k,
    )

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
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **жирный**
    text = re.sub(r"__(.+?)__", r"\1", text)        # __жирный__
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"\1", text)  # *курсив*
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # # заголовки
    return text


def _circuit_is_open() -> bool:
    return time.monotonic() < _circuit_open_until


def _record_failure() -> None:
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures += 1
    if _consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        _circuit_open_until = time.monotonic() + CIRCUIT_BREAKER_COOLDOWN_SECONDS
        logger.warning(
            "Circuit breaker открыт на %.0fс после %s сбоев подряд",
            CIRCUIT_BREAKER_COOLDOWN_SECONDS, _consecutive_failures,
        )


def _record_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


def _call_claude_api(system_prompt: str, question: str):
    return requests.post(
        API_URL,
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 700,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": question}
            ],
        },
        timeout=60,
    )


def ask_claude(system_prompt: str, question: str) -> str:
    if _circuit_is_open():
        return "Сервис временно недоступен, попробуйте через минуту."

    last_error_message = "Не получилось связаться с сервером, попробуйте ещё раз."

    for attempt in range(CLAUDE_MAX_RETRIES + 1):
        try:
            response = _call_claude_api(system_prompt, question)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            logger.warning("Ошибка сети/API (попытка %s): %s", attempt + 1, e)
            last_error_message = "Не получилось связаться с сервером, попробуйте ещё раз."
        except ValueError:
            logger.error("Ответ сервера не является JSON: %s", response.text[:200])
            _record_failure()
            return "Сервер вернул некорректный ответ."
        else:
            if "content" not in result:
                logger.error("Ошибка API: %s", result)
                _record_failure()
                return "Произошла ошибка при обращении к API."

            _record_success()
            answer = result["content"][0]["text"]
            return strip_markdown(answer)

        if attempt < CLAUDE_MAX_RETRIES:
            time.sleep(CLAUDE_RETRY_BACKOFF_SECONDS * (2 ** attempt))

    _record_failure()
    return last_error_message


def ask_rag(collection, question: str, chunks: list[dict] | None = None) -> str:
    if chunks is None:
        chunks = retrieve_relevant_chunks(collection, question)
    if not chunks:
        return "У меня нет информации по этому вопросу в базе знаний."
    system_prompt = build_system_prompt(chunks)
    return ask_claude(system_prompt, question)
