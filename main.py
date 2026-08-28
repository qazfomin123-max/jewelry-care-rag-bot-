import logging
import os
import build_index

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


load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

collection = None  # загружается один раз при старте


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
    await update.message.chat.send_action(action="typing")

    try:
        chunks = rag_core.retrieve_relevant_chunks(collection, question)
        logger.info("Вопрос: %s | Найдено кусков: %s", question, len(chunks))
        for chunk in chunks:
            preview = chunk["text"][:60].replace("\n", " ")
            logger.info("  dist=%.3f | %s | %s...", chunk["distance"], chunk["source"], preview)

        answer = rag_core.ask_rag(collection, question)
    except Exception:
        logger.exception("Ошибка при обработке вопроса: %s", question)
        answer = "Произошла внутренняя ошибка, попробуйте позже."

    await update.message.reply_text(answer)

def main() -> None:
    global collection

    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "Не найден TELEGRAM_BOT_TOKEN. Проверь файл .env рядом с этим скриптом."
        )

    logger.info("Строю индекс базы знаний...")
    build_index.build_index()

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
