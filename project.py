import yfinance as yf
obj = yf.Ticker('MTBV').financials
print(obj)