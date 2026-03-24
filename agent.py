"""Claude-powered trading agent using tool use."""

import json
import logging
from datetime import datetime

import anthropic

import config
import data as market_data
from broker import IBBroker

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
        "description": "Get recent news headlines for a stock symbol to assess sentiment.",
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
]


SYSTEM_PROMPT = """You are an autonomous stock trading agent managing a live brokerage account via Interactive Brokers.

Your responsibilities:
1. Analyse the watchlist stocks using market data (price, volume, RSI, MACD, moving averages) and recent news
2. Review the current portfolio and open orders
3. Make rational, risk-managed trading decisions
4. Execute trades when you have sufficient conviction

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

Work through the watchlist systematically. For each stock: get market data, check news, then decide.
After reviewing all stocks, check the portfolio and manage existing positions if needed.
When you are done making all decisions and have executed all intended orders, stop."""


class TradingAgent:
    def __init__(self, broker: IBBroker):
        self.broker = broker
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def run_cycle(self):
        """Run one full agent decision cycle."""
        logger.info("--- Starting agent cycle ---")
        system = SYSTEM_PROMPT.format(
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

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

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
                return self.broker.get_portfolio()
            elif name == "get_open_orders":
                return self.broker.get_open_orders()
            elif name == "place_order":
                return self.broker.place_order(
                    symbol=inputs["symbol"],
                    action=inputs["action"],
                    quantity=int(inputs["quantity"]),
                    order_type=inputs["order_type"],
                    limit_price=inputs.get("limit_price"),
                )
            elif name == "cancel_order":
                return self.broker.cancel_order(int(inputs["order_id"]))
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"Tool {name} error: {e}")
            return {"error": str(e)}
