import time

from rate_limiter import RateLimiter


def test_first_request_always_allowed():
    limiter = RateLimiter(min_interval=1, max_per_window=5, window_seconds=60)
    allowed, reason = limiter.check(chat_id=1)
    assert allowed
    assert reason == ""


def test_second_request_too_soon_is_blocked():
    limiter = RateLimiter(min_interval=10, max_per_window=5, window_seconds=60)
    allowed_first, _ = limiter.check(chat_id=1)
    allowed_second, reason = limiter.check(chat_id=1)

    assert allowed_first
    assert not allowed_second
    assert reason != ""


def test_different_chats_do_not_affect_each_other():
    limiter = RateLimiter(min_interval=10, max_per_window=5, window_seconds=60)
    allowed_chat_1, _ = limiter.check(chat_id=1)
    allowed_chat_2, _ = limiter.check(chat_id=2)

    assert allowed_chat_1
    assert allowed_chat_2


def test_window_limit_blocks_after_max_requests():
    limiter = RateLimiter(min_interval=0, max_per_window=3, window_seconds=60)

    results = [limiter.check(chat_id=1)[0] for _ in range(4)]

    assert results == [True, True, True, False]


def test_request_allowed_again_after_min_interval_passes():
    limiter = RateLimiter(min_interval=0.05, max_per_window=5, window_seconds=60)

    allowed_first, _ = limiter.check(chat_id=1)
    time.sleep(0.06)
    allowed_second, _ = limiter.check(chat_id=1)

    assert allowed_first
    assert allowed_second
