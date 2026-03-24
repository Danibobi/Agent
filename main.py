"""Entry point: runs the trading agent on a configurable interval."""

import logging
import time
import sys

import config
from broker import IBBroker
from agent import TradingAgent
from performance import PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_agent.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initialising trading agent")
    logger.info(f"Watchlist: {config.WATCH_LIST}")
    logger.info(f"IB connection: {config.IB_HOST}:{config.IB_PORT}")
    logger.info(f"Cycle interval: {config.AGENT_INTERVAL_SECONDS}s")

    broker = IBBroker()
    broker.connect()

    tracker = PerformanceTracker()
    agent = TradingAgent(broker, tracker)

    try:
        while True:
            try:
                agent.run_cycle()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Agent cycle failed: {e}", exc_info=True)

            logger.info(f"Sleeping {config.AGENT_INTERVAL_SECONDS}s until next cycle...")
            time.sleep(config.AGENT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        broker.disconnect()
        logger.info("Disconnected from IB")


if __name__ == "__main__":
    main()
