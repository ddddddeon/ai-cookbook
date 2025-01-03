from smolagents import Tool


class StockInfoTool(Tool):
    name = "get_stock_info"
    description = "Get stock info for a stock ticker"
    inputs = {
        "ticker": {
            "type": "string",
            "description": "The stock ticker to look up. This is the ticker symbol, NOT the full company name. Examples: AAPL, MSFT, TSLA",
        }
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(self)

    def forward(self, ticker: str) -> str:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        stock_price = stock.history(period="1d")["Close"].iloc[-1]
        return str(stock_price)
