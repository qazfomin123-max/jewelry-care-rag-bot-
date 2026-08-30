"""
Golden-set из debug_distances.py, оформленный как настоящие pytest-тесты
с ассертами вместо print(). Проверяет, что топ-совпадение приходит из
ожидаемого источника, и что вопрос не по теме базы отсекается порогом
RAG_MAX_DISTANCE.

Требует собранный индекс (chroma_db/, см. python build_index.py) и
установленные rag-зависимости — если их нет, модуль целиком
пропускается, а не падает.
"""

import os

import pytest

pytest.importorskip("chromadb")

import rag_core  # noqa: E402  (после importorskip)

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

if not os.path.isdir(CHROMA_DB_DIR):
    pytest.skip(
        "chroma_db/ не найдена — сначала запусти python build_index.py",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def collection():
    return rag_core.load_collection()


TOPIC_CASES = [
    ("Как почистить золотое кольцо в домашних условиях?", "zolot"),
    ("Можно ли носить золото каждый день?", "zolot"),
    ("Как ухаживать за серебряным украшением?", "serebr"),
    ("Почему серебро темнеет и как это остановить?", "serebr"),
    ("Как проверить подлинность бриллианта?", "brillian"),
    ("Что означает проба 585 на украшении?", "prob"),
    ("Как правильно определить размер кольца?", "razmer"),
]


@pytest.mark.parametrize("question, expected_source_substring", TOPIC_CASES)
def test_top_match_comes_from_expected_source(collection, question, expected_source_substring):
    chunks = rag_core.retrieve_relevant_chunks(collection, question)

    assert chunks, f"По вопросу «{question}» не нашлось ни одного куска в пределах порога"
    top_source = chunks[0]["source"].lower()
    assert expected_source_substring in top_source, (
        f"Ожидали источник с «{expected_source_substring}» в имени, "
        f"получили «{chunks[0]['source']}»"
    )


def test_out_of_domain_question_is_filtered_out(collection):
    # Контрольный вопрос НЕ по теме базы — должен давать distance выше
    # порога RAG_MAX_DISTANCE и не возвращать ни одного куска.
    chunks = rag_core.retrieve_relevant_chunks(
        collection, "Как починить сломанные наручные часы?"
    )
    assert chunks == []
