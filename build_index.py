import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
import chromadb
from chromadb.utils import embedding_functions


KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "jewelry_care_kb"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

MIN_CHUNK_CHARS = 30
TARGET_CHUNK_CHARS = 1000   # целевой размер блока — группируем абзацы, пока не наберём этот объём
MAX_CHUNK_CHARS = 1400      # жёсткий потолок на блок (страховка от аномально длинных абзацев)

# Заголовки статей (короткие абзацы) не несут пользы как самостоятельные
# чанки — их нужно приклеивать к следующему абзацу с реальным контентом.
HEADER_MAX_CHARS = 80


def split_into_chunks(text: str) -> list[str]:
    """
    Группирует абзацы статьи в блоки по ~TARGET_CHUNK_CHARS символов.
    Каждый блок заканчивается последним абзацем, который в него поместился,
    и этот же абзац становится первым в следующем блоке (overlap) —
    так соседние блоки не режут мысль на границе и поиск не теряет контекст,
    который был на стыке двух абзацев.
    """
    raw_paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in raw_paragraphs if len(p.strip()) >= MIN_CHUNK_CHARS]

    if not paragraphs:
        return []

    # Заголовок (короткий первый абзац) приклеиваем к следующему абзацу,
    # чтобы он не терялся и не становился самостоятельным пустым блоком.
    if len(paragraphs[0]) <= HEADER_MAX_CHARS and len(paragraphs) > 1:
        paragraphs[1] = paragraphs[0] + "\n\n" + paragraphs[1]
        paragraphs = paragraphs[1:]

    blocks = []
    current_block = []
    current_len = 0

    for paragraph in paragraphs:
        # Аномально длинный одиночный абзац — режем отдельно, чтобы не
        # получить один гигантский блок далеко за MAX_CHUNK_CHARS.
        if len(paragraph) > MAX_CHUNK_CHARS:
            if current_block:
                blocks.append("\n\n".join(current_block))
                current_block = []
                current_len = 0
            blocks.extend(_split_long_text(paragraph))
            continue

        if current_len + len(paragraph) > TARGET_CHUNK_CHARS and current_block:
            blocks.append("\n\n".join(current_block))
            # overlap: последний абзац предыдущего блока переносим в начало нового
            current_block = [current_block[-1], paragraph]
            current_len = len(current_block[0]) + len(paragraph)
        else:
            current_block.append(paragraph)
            current_len += len(paragraph)

    if current_block:
        blocks.append("\n\n".join(current_block))

    return blocks


def _split_long_text(text: str) -> list[str]:
    overlap = 100
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def load_documents() -> list[dict]:
    documents = []
    file_paths = sorted(glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt")))

    if not file_paths:
        raise RuntimeError(
            f"В папке {KNOWLEDGE_BASE_DIR}/ не найдено ни одного .txt файла."
        )

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        source_name = os.path.basename(file_path)
        chunks = split_into_chunks(text)

        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{source_name}::chunk_{i}",
                "text": chunk,
                "source": source_name,
            })

    return documents


def build_index() -> None:
    print("Загружаю документы из knowledge_base/...")
    documents = load_documents()
    print(f"Найдено кусков текста: {len(documents)}")

    print(f"Загружаю модель эмбеддингов ({EMBEDDING_MODEL_NAME})...")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("Открываю/создаю базу ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print("Старая версия базы удалена, строю заново.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    print("Считаю эмбеддинги и сохраняю в базу...")
    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[{"source": doc["source"]} for doc in documents],
    )

    print(f"Готово! Проиндексировано кусков: {collection.count()}")
    del embedding_function
    del client
    del collection
    import gc
    gc.collect()

if __name__ == "__main__":
    build_index()
