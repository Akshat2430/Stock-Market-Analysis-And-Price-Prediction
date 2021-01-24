# Plotting candlestick chart
fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
fig.show()

# Decomposition of time series data for forecasting
df_close = df[['Date', 'Close']].copy()
df_close = df_close.set_index('Date')
df_close.head()
decomp = decompose(df_close, period=365)
fig = decomp.plot()
fig.set_size_inches(20, 8)
