"""
Простой троттлинг обращений по chat_id.

Каждое сообщение пользователя = платный вызов LLM через прокси, поэтому
без лимита кто угодно может зациклить сообщения и нагенерировать счёт.
Ограничиваем два раза:
  - не чаще одного запроса раз в `min_interval` секунд;
  - не больше `max_per_window` запросов за скользящее окно `window_seconds`.

Состояние хранится в памяти процесса — рестарт бота сбрасывает счётчики.
Для одного процесса (Railway free/hobby-план) этого достаточно; если бот
когда-нибудь будет жить в нескольких процессах одновременно, лимиты
нужно будет вынести в общее хранилище (Redis и т.п.).
"""

import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, min_interval: float, max_per_window: int, window_seconds: float):
        self.min_interval = min_interval
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._last_request: dict[int, float] = {}
        self._history: dict[int, deque] = defaultdict(deque)

    def check(self, chat_id: int) -> tuple[bool, str]:
        """
        Возвращает (разрешено, сообщение_для_пользователя_если_нет).
        Если запрос разрешён — сразу засчитывает его (идемпотентности нет,
        вызывать один раз на входящее сообщение).
        """
        now = time.monotonic()

        last = self._last_request.get(chat_id)
        if last is not None and now - last < self.min_interval:
            return False, "Слишком часто. Подождите пару секунд и повторите вопрос."

        history = self._history[chat_id]
        while history and now - history[0] > self.window_seconds:
            history.popleft()

        if len(history) >= self.max_per_window:
            return False, (
                "Превышен лимит вопросов в минуту. "
                "Подождите немного и попробуйте снова."
            )

        self._last_request[chat_id] = now
        history.append(now)
        return True, ""
