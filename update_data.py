
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import time

def get_analysis():
    # Add a retry loop to handle Yahoo Finance connection issues
    for attempt in range(3):
        try:
            print(f"Attempt {attempt + 1} to fetch data...")
            ticker = yf.Ticker("EURUSD=X")
            df = ticker.history(period="5d", interval="1h")
            
            if df.empty:
                print("Empty dataframe received. Retrying...")
                time.sleep(5)
                continue
                
            df = df.dropna()
            
            # Calculate indicators
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # RSI Calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Get the latest row
            latest = df.iloc[-1]
            price = round(latest['Close'], 4)
            
            # --- CURRENT CANDLE EXPLANATION ---
            if latest['Close'] > latest['Open']:
                candle_current = f"The current 1-hour candle is bullish, opening at {latest['Open']:.4f} and currently at {price}. Buyers are in control."
            else:
                candle_current = f"The current 1-hour candle is bearish, opening at {latest['Open']:.4f} and currently at {price}. Sellers are pushing the price down."
            
            # --- SIGNAL LOGIC ---
            if latest['Close'] > latest['EMA_20'] and latest['RSI'] > 50:
                signal = "BUY"
                reason = f"Price is trading above the 20 EMA and RSI is at {latest['RSI']:.1f}, showing bullish momentum."
            elif latest['Close'] < latest['EMA_20'] and latest['RSI'] < 50:
                signal = "SELL"
                reason = f"Price is trading below the 20 EMA and RSI is at {latest['RSI']:.1f}, showing bearish pressure."
            else:
                signal = "HOLD"
                reason = f"Price is consolidating. RSI is neutral at {latest['RSI']:.1f}. Waiting for a clear breakout."
            
            # --- TIMEFRAME ANALYSIS ---
            c1h = f"The 1-hour candle closed at {price}. " + ("It is holding above the 20 EMA, confirming the uptrend." if latest['Close'] > latest['EMA_20'] else "It is rejecting the 20 EMA, confirming the downtrend.")
            c4h = f"The broader 4-hour trend looks " + ("bullish as price holds above key moving averages." if latest['Close'] > latest['EMA_50'] else "bearish as price struggles below key moving averages.")
            
            # --- ENTRY, SL, TP CALCULATION ---
            atr = (latest['High'] - latest['Low']) * 1.5
            entry = price
            sl = round(price - atr, 4) if signal == "BUY" else round(price + atr, 4)
            tp = round(price + (atr * 2), 4) if signal == "BUY" else round(price - (atr * 2), 4)
            
            valid_until = (datetime.utcnow() + timedelta(hours=1)).strftime("%H:%M UTC")
            
            return {
                "signal": signal,
                "reason": reason,
                "price": str(price),
                "entry": str(entry),
                "sl": str(sl),
                "tp": str(tp),
                "valid": valid_until,
                "candle_current": candle_current,
                "candle_1h": c1h,
                "candle_4h": c4h
            }
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(5) # Wait 5 seconds before retrying
            
    print("All attempts failed. Yahoo Finance may be blocking GitHub servers.")
    return None

if __name__ == "__main__":
    data = get_analysis()
    if data:
        with open("data.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Successfully updated data.json with live EUR/USD data.")
    else:
        print("Failed to fetch data from Yahoo Finance.")
