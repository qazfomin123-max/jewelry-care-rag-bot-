# Пустой conftest.py в корне репозитория гарантирует, что pytest
# добавляет корень проекта в sys.path — иначе `import rag_core` и
# `from rate_limiter import RateLimiter` из tests/ не найдутся.
