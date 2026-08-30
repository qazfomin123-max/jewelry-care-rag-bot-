import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import asyncio
import logging

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import rag_core
from rate_limiter import RateLimiter


load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Троттлинг: не чаще одного запроса в RATE_LIMIT_MIN_INTERVAL секунд и не
# больше RATE_LIMIT_MAX_PER_WINDOW запросов за RATE_LIMIT_WINDOW_SECONDS
# на один chat_id. Каждое сообщение — платный вызов LLM через прокси.
RATE_LIMIT_MIN_INTERVAL = float(os.environ.get("RATE_LIMIT_MIN_INTERVAL", 3))
RATE_LIMIT_MAX_PER_WINDOW = int(os.environ.get("RATE_LIMIT_MAX_PER_WINDOW", 10))
RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", 60))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

collection = None  # загружается один раз при старте

rate_limiter = RateLimiter(
    min_interval=RATE_LIMIT_MIN_INTERVAL,
    max_per_window=RATE_LIMIT_MAX_PER_WINDOW,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)

# Замок на chat_id: не даём второму сообщению того же пользователя
# обрабатываться, пока не завершилось первое — иначе можно наспамить
# параллельных платных запросов к LLM, пока первый ещё висит на прокси.
_chat_locks: dict[int, asyncio.Lock] = {}


def _get_chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _chat_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[chat_id] = lock
    return lock


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я консультант по уходу за ювелирными изделиями.\n"
        "Спроси меня, например:\n"
        "— Как чистить золотое кольцо?\n"
        "— Можно ли мыть жемчуг с мылом?\n"
        "— Что означает проба 585?"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text
    chat_id = update.effective_chat.id

    allowed, reason = rate_limiter.check(chat_id)
    if not allowed:
        await update.message.reply_text(reason)
        return

    lock = _get_chat_lock(chat_id)
    if lock.locked():
        await update.message.reply_text(
            "Ещё обрабатываю предыдущий вопрос, подождите ответа."
        )
        return

    async with lock:
        await update.message.chat.send_action(action="typing")

        try:
            # collection.query() и requests.post() внутри синхронные и
            # CPU/IO-блокирующие — уводим их в отдельный поток, чтобы не
            # подвешивать event loop (и обработку апдейтов от Telegram)
            # на всё время инференса эмбеддинга и ожидания ответа прокси.
            chunks = await asyncio.to_thread(
                rag_core.retrieve_relevant_chunks, collection, question
            )
            logger.info("Вопрос: %s | Найдено кусков: %s", question, len(chunks))
            for chunk in chunks:
                preview = chunk["text"][:60].replace("\n", " ")
                logger.info(
                    "  dist=%.3f | %s | %s...",
                    chunk["distance"], chunk["source"], preview,
                )
            answer = await asyncio.to_thread(
                rag_core.ask_rag, collection, question, chunks
            )
        except Exception:
            logger.exception("Ошибка при обработке вопроса: %s", question)
            answer = "Произошла внутренняя ошибка, попробуйте позже."

        try:
            await update.message.reply_text(answer)
        except Exception:
            # Если это упадёт — пользователь не должен остаться совсем
            # без ответа молча: как минимум это попадёт в логи.
            logger.exception("Не удалось отправить ответ пользователю (chat_id=%s)", chat_id)


def main() -> None:
    global collection

    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "Не найден TELEGRAM_BOT_TOKEN. Проверь файл .env рядом с этим скриптом."
        )
    if not os.environ.get("TOOKEN_API_KEY"):
        raise RuntimeError(
            "Не найден TOOKEN_API_KEY. Проверь файл .env рядом с этим скриптом."
        )

    logger.info("Загружаю базу знаний...")
    collection = rag_core.load_collection()
    logger.info("База загружена, кусков в базе: %s", collection.count())

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен, жду сообщений (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
