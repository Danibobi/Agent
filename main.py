"""Entry point: runs the trading agent on a configurable interval."""

import logging
import logging.handlers
import time
import sys

import config
from broker import IBBroker
from agent import TradingAgent
from performance import PerformanceTracker
from telegram_notifier import TelegramNotifier

_rot = logging.handlers.RotatingFileHandler(
    "trading_agent.log", maxBytes=10 * 1024 * 1024, backupCount=7
)
_rot.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        _rot,
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initialising trading agent")
    logger.info("Log files: trading_agent.log | daily_log.json | trade_journal.json")
    logger.info(f"Watchlist: {config.WATCH_LIST}")
    logger.info(f"IB connection: {config.IB_HOST}:{config.IB_PORT}")
    logger.info(f"Cycle interval: {config.AGENT_INTERVAL_SECONDS}s")

    broker = IBBroker()
    broker.connect()

    notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    notifier.notify_startup(
        watchlist=config.WATCH_LIST,
        mode_label="PAPER TRADING" if config.PAPER_TRADING else "LIVE",
        interval_sec=config.AGENT_INTERVAL_SECONDS,
    )

    tracker = PerformanceTracker()
    agent = TradingAgent(broker, tracker, notifier)

    try:
        while True:
            try:
                agent.run_cycle()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Agent cycle failed: {e}", exc_info=True)
                notifier.notify_error(str(e))

            logger.info(f"Sleeping {config.AGENT_INTERVAL_SECONDS}s until next cycle...")
            time.sleep(config.AGENT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        broker.disconnect()
        logger.info("Disconnected from IB")


if __name__ == "__main__":
    main()
