import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time

# Optional imports - wrapped so they don't crash the script
try:
    import pandas_ta_classic as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("pandas_ta_classic not installed - pattern detection disabled")

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("groq not installed - LLM reasoning disabled")


def get_analysis():
    for attempt in range(3):
        try:
            # --- FETCH CANDLES ---
            ticker = yf.Ticker("EURUSD=X")
            df = ticker.history(period="5d", interval="1h")
            if df.empty:
                time.sleep(5)
                continue
            df = df.dropna()

            # --- INDICATORS ---
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            latest = df.iloc[-1]
            price = round(latest['Close'], 4)
            rsi = round(latest['RSI'], 1)
            ema20 = round(latest['EMA_20'], 4)

            # --- RULE 1: NEWS SENTIMENT (optional) ---
            news_bias, news_score = "unavailable", 0
            try:
                from fxnewsbias import Client
                fx = Client()
                s = fx.sentiment()
                news_bias = s.favours("EUR/USD") or "flat"
                news_score = s.spread("EUR/USD")
            except Exception as e:
                print(f"News error (skipping): {e}")

            # --- RULE 2: CANDLESTICK PATTERNS ---
            patterns_list = ["No clear pattern"]
            if HAS_TA:
                try:
                    patterns = df.ta.cdl_pattern(name="all")
                    patterns_list = []
                    for col in patterns.columns:
                        if patterns[col].iloc[-1] != 0:
                            name = col.replace("CDL_", "").replace("_", " ").title()
                            direction = "bullish" if patterns[col].iloc[-1] > 0 else "bearish"
                            patterns_list.append(f"{name} ({direction})")
                    if not patterns_list:
                        patterns_list = ["No clear pattern"]
                except Exception as e:
                    print(f"Pattern error: {e}")

            # --- RULE 3: SESSION TIMING ---
            hour = datetime.utcnow().hour
            if 7 <= hour < 10:
                session, volatility = "London Kill Zone", "high"
            elif 12 <= hour < 15:
                session, volatility = "New York Kill Zone", "high"
            elif 2 <= hour < 5:
                session, volatility = "Asian Kill Zone", "low"
            else:
                session, volatility = "Off-Session", "low"

            # --- TECHNICAL SIGNAL ---
            if latest['Close'] > latest['EMA_20'] and rsi > 50:
                tech_signal = "BUY"
            elif latest['Close'] < latest['EMA_20'] and rsi < 50:
                tech_signal = "SELL"
            else:
                tech_signal = "HOLD"

            # --- RULE 4: LLM VERDICT (optional) ---
            verdict = tech_signal
            llm_reason = f"Technical signal: Price {'above' if latest['Close'] > latest['EMA_20'] else 'below'} 20 EMA, RSI at {rsi}."
            confidence = "MEDIUM"
            
            if HAS_GROQ and os.environ.get("GROQ_API_KEY"):
                try:
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                    prompt = f"""You are a professional EUR/USD trader. Analyze:
Price: {price}, RSI: {rsi}, EMA20: {ema20}
News Bias: {news_bias} (score: {news_score})
Patterns: {', '.join(patterns_list)}
Session: {session} ({volatility} volatility)
Technical Signal: {tech_signal}

Give final verdict (BUY/SELL/HOLD), reason (2 sentences), confidence (LOW/MEDIUM/HIGH).
Respond ONLY with JSON like: {{"verdict": "BUY", "reason": "...", "confidence": "HIGH"}}"""
                    
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1, max_tokens=500,
                        response_format={"type": "json_object"}
                    )
                    result = json.loads(response.choices[0].message.content)
                    verdict = result.get("verdict", tech_signal)
                    llm_reason = result.get("reason", llm_reason)
                    confidence = result.get("confidence", "MEDIUM")
                except Exception as e:
                    print(f"LLM error (using technical signal): {e}")

            # --- CANDLE TEXT ---
            if latest['Close'] > latest['Open']:
                candle_current = f"Bullish candle. Open: {latest['Open']:.4f}, Now: {price}. Buyers in control."
            else:
                candle_current = f"Bearish candle. Open: {latest['Open']:.4f}, Now: {price}. Sellers pushing down."

            c1h = f"1H candle closed at {price}. " + ("Above 20 EMA - uptrend." if latest['Close'] > latest['EMA_20'] else "Below 20 EMA - downtrend.")
            c4h = f"4H trend: " + ("bullish above moving averages." if latest['Close'] > latest['EMA_50'] else "bearish below moving averages.")

            # --- SL/TP ---
            atr = (latest['High'] - latest['Low']) * 1.5
            entry = price
            sl = round(price - atr, 4) if verdict == "BUY" else round(price + atr, 4)
            tp = round(price + (atr * 2), 4) if verdict == "BUY" else round(price - (atr * 2), 4)
            valid_until = (datetime.utcnow() + timedelta(hours=1)).strftime("%H:%M UTC")

            return {
                "signal": verdict,
                "reason": llm_reason,
                "confidence": confidence,
                "price": str(price),
                "entry": str(entry),
                "sl": str(sl),
                "tp": str(tp),
                "valid": valid_until,
                "candle_current": candle_current,
                "candle_1h": c1h,
                "candle_4h": c4h,
                "news_bias": news_bias,
                "news_score": news_score,
                "patterns": patterns_list,
                "session": session,
                "volatility": volatility
            }
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
            time.sleep(5)
    return None


if __name__ == "__main__":
    data = get_analysis()
    if data:
        with open("data.json", "w") as f:
            json.dump(data, f, indent=4)
        print("SUCCESS: data.json updated with live EUR/USD data.")
        print(f"Price: {data['price']} | Signal: {data['signal']} | Confidence: {data['confidence']}")
    else:
        print("FAILED: Could not fetch EUR/USD data.")
