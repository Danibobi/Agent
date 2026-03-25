import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

IB_HOST: str = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT: int = int(os.getenv("IB_PORT", "7497"))
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() in ("true", "1", "yes")
IB_CLIENT_ID: int = int(os.getenv("IB_CLIENT_ID", "1"))

WATCH_LIST: list[str] = [s.strip() for s in os.getenv("WATCH_LIST", "AAPL,MSFT,GOOGL,AMZN,NVDA").split(",")]
AGENT_INTERVAL_SECONDS: int = int(os.getenv("AGENT_INTERVAL_SECONDS", "300"))
MAX_POSITION_SIZE_PCT: float = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.1"))

TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")
