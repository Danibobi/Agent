"""Interactive Brokers integration via ib_insync."""

import logging
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import config

logger = logging.getLogger(__name__)

LIVE_PORT = 7496
PAPER_PORT = 7497


class IBBroker:
    def __init__(self):
        self.ib = IB()

    def connect(self):
        if config.PAPER_TRADING and config.IB_PORT == LIVE_PORT:
            logger.warning("=" * 70)
            logger.warning("WARNING: PAPER_TRADING=true but IB_PORT=%s (live trading port).", LIVE_PORT)
            logger.warning("You may be connecting to a LIVE account with paper-trading mode set.")
            logger.warning("Set IB_PORT=7497 in .env, or PAPER_TRADING=false to suppress this warning.")
            logger.warning("=" * 70)
        elif not config.PAPER_TRADING and config.IB_PORT == PAPER_PORT:
            logger.warning("=" * 70)
            logger.warning("WARNING: PAPER_TRADING=false but IB_PORT=%s (paper trading port).", PAPER_PORT)
            logger.warning("Config claims live mode but you are connecting to a paper account.")
            logger.warning("=" * 70)

        self.ib.connect(config.IB_HOST, config.IB_PORT, clientId=config.IB_CLIENT_ID)
        mode_label = "PAPER TRADING" if config.PAPER_TRADING else "LIVE TRADING"
        logger.info("Connected to IB on %s:%s [%s]", config.IB_HOST, config.IB_PORT, mode_label)

    def disconnect(self):
        self.ib.disconnect()

    def get_portfolio(self) -> dict:
        """Return current positions and account cash balance."""
        account_values = self.ib.accountSummary()
        cash = 0.0
        net_liquidation = 0.0
        buying_power = 0.0

        for av in account_values:
            if av.tag == "CashBalance" and av.currency == "BASE":
                cash = float(av.value)
            elif av.tag == "NetLiquidation" and av.currency == "BASE":
                net_liquidation = float(av.value)
            elif av.tag == "BuyingPower" and av.currency == "BASE":
                buying_power = float(av.value)

        positions = []
        for pos in self.ib.positions():
            contract = pos.contract
            if hasattr(contract, "symbol"):
                positions.append({
                    "symbol": contract.symbol,
                    "shares": float(pos.position),
                    "avg_cost": round(float(pos.avgCost), 2),
                    "market_value": round(float(pos.position) * float(pos.avgCost), 2),
                })

        return {
            "cash": round(cash, 2),
            "net_liquidation": round(net_liquidation, 2),
            "buying_power": round(buying_power, 2),
            "positions": positions,
        }

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: float | None = None,
    ) -> dict:
        """Place a buy or sell order. action must be 'BUY' or 'SELL'."""
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        if order_type == "MKT":
            order = MarketOrder(action, quantity)
        elif order_type == "LMT":
            if limit_price is None:
                return {"error": "limit_price is required for LMT orders"}
            order = LimitOrder(action, quantity, limit_price)
        else:
            return {"error": f"Unknown order_type: {order_type}"}

        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)  # allow IB to acknowledge

        return {
            "order_id": trade.order.orderId,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": trade.orderStatus.status,
        }

    def cancel_order(self, order_id: int) -> dict:
        """Cancel an open order by order ID."""
        open_trades = self.ib.openTrades()
        for trade in open_trades:
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                self.ib.sleep(1)
                return {"order_id": order_id, "status": "cancelled"}
        return {"error": f"Order {order_id} not found in open orders"}

    def get_open_orders(self) -> dict:
        """Return all currently open/pending orders."""
        self.ib.reqAllOpenOrders()
        self.ib.sleep(1)
        orders = []
        for trade in self.ib.openTrades():
            orders.append({
                "order_id": trade.order.orderId,
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "limit_price": getattr(trade.order, "lmtPrice", None),
                "status": trade.orderStatus.status,
            })
        return {"open_orders": orders}
