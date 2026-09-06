"""Shared rate limiter, applied only to endpoints that call OpenAI (/chat).

In-memory, per-process, keyed by client IP, this is deliberately simple:
Render's free tier runs a single uvicorn worker, so there is no
multi-process state to coordinate. Limit is generous for a real evaluator
(expected well under 20 questions in one session) while blocking a script
that would otherwise run up the OpenAI bill on a publicly reachable demo.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

CHAT_RATE_LIMIT = "40/hour"
