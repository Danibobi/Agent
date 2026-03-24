"""Claude-powered trading agent using tool use."""

import json
import logging
from datetime import datetime

import anthropic

import config
import data as market_data
from broker import IBBroker
from performance import PerformanceTracker

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "name": "get_market_data",
        "description": (
            "Get historical price data and technical indicators (SMA, RSI, MACD) "
            "for a stock symbol. Use this to analyse a stock before making a decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                "period": {
                    "type": "string",
                    "description": "Lookback period: 1d, 5d, 1mo, 3mo, 6mo, 1y",
                    "default": "1mo",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_news",
        "description": (
            "Get recent news headlines for a stock symbol with pre-scored sentiment. "
            "Each headline includes 'sentiment' (bullish/bearish/neutral) and 'sentiment_score' "
            "(-1.0 = strongly bearish, +1.0 = strongly bullish). The response also includes a "
            "'sentiment_summary' with headline counts and an 'overall_score' for the symbol. "
            "Combine these signals with technical indicators before making a trading decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio",
        "description": "Get current portfolio positions, cash balance, and buying power.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_open_orders",
        "description": "Get all currently open or pending orders.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "place_order",
        "description": (
            "Place a buy or sell order for a stock. "
            "Always check get_portfolio first to ensure sufficient buying power before buying. "
            "Do not allocate more than MAX_POSITION_SIZE_PCT of portfolio value to a single position."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "action": {"type": "string", "enum": ["BUY", "SELL"], "description": "BUY or SELL"},
                "quantity": {"type": "integer", "description": "Number of shares", "minimum": 1},
                "order_type": {
                    "type": "string",
                    "enum": ["MKT", "LMT"],
                    "description": "MKT for market order, LMT for limit order",
                },
                "limit_price": {
                    "type": "number",
                    "description": "Required only for LMT orders. Set slightly below ask for buys, above bid for sells.",
                },
            },
            "required": ["symbol", "action", "quantity", "order_type"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an open order by its order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer", "description": "Order ID to cancel"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_performance_report",
        "description": (
            "Get a full performance report showing realized and unrealized P&L, "
            "win rate, total return %, best/worst trade, and P&L broken down by symbol. "
            "Call this at the start of a cycle to assess your own track record before "
            "making new decisions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


SYSTEM_PROMPT = """You are an autonomous stock trading agent managing a {mode_label} brokerage account via Interactive Brokers.

{mode_instructions}

Your responsibilities:
1. Analyse the watchlist stocks using market data (price, volume, RSI, MACD, moving averages) and recent news
2. Use the sentiment scores in the news data to gauge market mood for each stock:
   - 'overall_score' near +1.0 = strongly bullish news environment
   - 'overall_score' near -1.0 = strongly bearish news environment
   - Combine sentiment with technicals: high RSI + bullish news = caution (overbought hype)
   - Low RSI + bearish news = wait for stabilisation before buying
3. Review the current portfolio and open orders
4. Make rational, risk-managed trading decisions
5. Execute trades when you have sufficient conviction across both technical and sentiment signals

Risk management rules you MUST follow:
- Never allocate more than {max_position_pct}% of total portfolio value to a single position
- Check buying power before placing any BUY order
- Prefer limit orders over market orders for better price control
- Avoid trading in illiquid or thinly traded stocks
- If RSI > 70, a stock is overbought — be cautious buying
- If RSI < 30, a stock is oversold — potential buy opportunity
- Always consider the overall market context

Today's date: {date}
Watchlist: {watchlist}

Work through the watchlist systematically. For each stock: get market data, check news (note the sentiment scores), then decide.
After reviewing all stocks, check the portfolio and manage existing positions if needed.
When you are done making all decisions and have executed all intended orders, stop."""


class TradingAgent:
    def __init__(self, broker: IBBroker, tracker: PerformanceTracker):
        self.broker = broker
        self.tracker = tracker
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def run_cycle(self):
        """Run one full agent decision cycle."""
        logger.info("--- Starting agent cycle ---")
        self.tracker.log_startup_summary()

        if config.PAPER_TRADING:
            mode_label = "PAPER TRADING (simulated)"
            mode_instructions = (
                "IMPORTANT: This is a PAPER TRADING session — no real money is at risk. "
                "Use this environment to test strategies freely, explore different position sizes, "
                "and experiment with entry/exit timing. Apply the same logic you would in live "
                "trading, but act on moderate conviction rather than waiting only for the highest-"
                "confidence setups."
            )
        else:
            mode_label = "LIVE (real money)"
            mode_instructions = (
                "IMPORTANT: This is a LIVE trading session with real capital. "
                "Apply strict risk management and only act on high-conviction signals."
            )

        system = SYSTEM_PROMPT.format(
            mode_label=mode_label,
            mode_instructions=mode_instructions,
            max_position_pct=int(config.MAX_POSITION_SIZE_PCT * 100),
            date=datetime.now().strftime("%A, %B %d, %Y %H:%M"),
            watchlist=", ".join(config.WATCH_LIST),
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Please analyse the watchlist ({', '.join(config.WATCH_LIST)}) "
                    "and manage the portfolio. Review each stock, check the current "
                    "portfolio, and execute any trades you deem appropriate."
                ),
            }
        ]

        # Accumulators for the daily activity log
        cycle_decisions: list[str] = []
        cycle_orders: list[dict] = []

        # Agentic tool-use loop
        while True:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            logger.info(f"Claude stop_reason: {response.stop_reason}")

            # Collect any text Claude outputs
            for block in response.content:
                if hasattr(block, "text"):
                    logger.info(f"Claude: {block.text}")
                    cycle_decisions.append(block.text)

            # If no tool calls, we're done
            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
                break

            # Append Claude's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Process tool calls and gather results
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                logger.info(f"Tool call: {tool_name}({json.dumps(tool_input)})")

                result = self._dispatch_tool(tool_name, tool_input)
                logger.info(f"Tool result: {json.dumps(result)}")

                # Track placed orders for the activity log
                if tool_name == "place_order" and "error" not in result:
                    cycle_orders.append({
                        "symbol": tool_input.get("symbol"),
                        "action": tool_input.get("action"),
                        "quantity": tool_input.get("quantity"),
                        "order_type": tool_input.get("order_type"),
                        "limit_price": tool_input.get("limit_price"),
                        "status": result.get("status"),
                        "actual_fill_price": result.get("actual_fill_price"),
                        "order_id": result.get("order_id"),
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

        # Persist cycle activity to daily_log.json
        self.tracker.record_cycle_log(cycle_decisions, cycle_orders)
        logger.info("--- Agent cycle complete ---")

    def _dispatch_tool(self, name: str, inputs: dict) -> dict:
        try:
            if name == "get_market_data":
                return market_data.get_market_data(
                    inputs["symbol"], inputs.get("period", "1mo")
                )
            elif name == "get_news":
                return market_data.get_news(inputs["symbol"])
            elif name == "get_portfolio":
                portfolio = self.broker.get_portfolio()
                self.tracker.record_snapshot(portfolio)
                return portfolio
            elif name == "get_open_orders":
                return self.broker.get_open_orders()
            elif name == "place_order":
                result = self.broker.place_order(
                    symbol=inputs["symbol"],
                    action=inputs["action"],
                    quantity=int(inputs["quantity"]),
                    order_type=inputs["order_type"],
                    limit_price=inputs.get("limit_price"),
                )
                if "error" not in result and result.get("status") != "Cancelled":
                    self.tracker.record_trade(
                        symbol=inputs["symbol"],
                        action=inputs["action"],
                        quantity=int(inputs["quantity"]),
                        order_type=inputs["order_type"],
                        limit_price=inputs.get("limit_price"),
                        order_id=result["order_id"],
                        actual_fill_price=result.get("actual_fill_price"),
                    )
                return result
            elif name == "cancel_order":
                return self.broker.cancel_order(int(inputs["order_id"]))
            elif name == "get_performance_report":
                return self.tracker.get_performance_report(portfolio=None)
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"Tool {name} error: {e}")
            return {"error": str(e)}
