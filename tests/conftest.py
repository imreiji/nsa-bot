import os

# Set once, before any test imports nsabot.bot (which reads its config at import time).
os.environ.update(
    DEEPSEEK_API_KEY="x", NSA_PROVIDER="deepseek", NSA_DB_PATH=":memory:", NSA_ADMIN_IDS="1", NSA_GUILD_IDS="10",
    NSA_QUEUE_TRIGGER="3", NSA_USER_RATE="5", NSA_QUIPS="off",
)
