import yfinance as yf
import pandas as pd
import json
import os
import random
from datetime import datetime, timedelta
import time

# Optional imports
try:
    import pandas_ta_classic as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("pandas_ta_classic not installed")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("google-generativeai not installed")


JOKES = [
    "Why did the trader break up with his girlfriend? Because she wanted a long-term commitment, but he only believed in scalping.",
    "I told my wife I was a forex trader. She said, 'So you're unemployed with extra steps?'",
    "Why do forex traders make terrible comedians? Because their timing is always off by a pip.",
    "A trader walked into a bar. He ordered a beer. Then he ordered another. Then he realized he was in a bear market.",
    "My stop loss is like my ex - always triggering at the worst possible moment.",
    "Why don't traders play hide and seek? Because good luck hiding from a volatile market.",
    "I asked the market for a direction. It said 'yes'.",
    "The market can stay irrational longer than you can stay solvent. That's not a joke, that's just life.",
    "What's the difference between a forex trader and a pizza? A pizza can feed a family of four.",
    "I only trade what I see. Sadly, I see four screens and a headache."
]

MOTIVATIONS = [
    "Remember Boss, the market rewards patience, not prediction. Wait for your setup.",
    "Discipline beats conviction. The best trade is often the one you don't take.",
    "Every loss is tuition. The market is the teacher, and you are the student.",
    "The trend is your friend until it bends. Respect your stop loss.",
    "Boss, protect your capital first. Opportunities are infinite, but your account is not.",
    "Rome wasn't built in a day, and fortunes aren't made in one trade. Stay consistent.",
    "Fear and greed are the only two enemies. Master them, and you master the market.",
    "Cut losses short, let winners run. The math will handle the rest.",
    "A calm mind catches more pips than a desperate one.",
    "You don't need to be right every time, Boss. You just need to be profitable overall."
]


def calculate_support_resistance(df):
    recent = df.tail(50)
    recent_high = round(recent['High'].max(), 4)
    recent_low = round(recent['Low'].min(), 4)
    mid = round((recent_high + recent_low) / 2, 4)
    return recent_high, recent_low, mid


def get_market_context(df):
    recent = df.tail(5)
    last_5 = []
    for _, row in recent.iterrows():
        last_5.append({
            "open": round(row['Open'], 4),
            "high": round(row['High'], 4),
            "low": round(row['Low'], 4),
            "close": round(row['Close'], 4),
            "direction": "up" if row['Close'] > row['Open'] else "down"
        })
    ranges = (df['High'] - df['Low']).tail(3).tolist()
    momentum = "expanding" if ranges[-1] > ranges[-2] > ranges[-3] else \
               "contracting" if ranges[-1] < ranges[-2] < ranges[-3] else "mixed"
    return last_5, momentum


def get_analysis():
    for attempt in range(3):
        try:
            ticker = yf.Ticker("EURUSD=X")
            df = ticker.history(period="5d", interval="1h")
            if df.empty:
                time.sleep(5)
                continue
            df = df.dropna()

            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            latest = df.iloc[-1]
            price = round(latest['Close'], 4)
            rsi = round(latest['RSI'], 1)
            ema20 = round(latest['EMA_20'], 4)
            ema50 = round(latest['EMA_50'], 4)
            atr = round(latest['ATR'], 4)

            if pd.isna(atr) or atr < 0.0005:
                recent_ranges = (df['High'] - df['Low']).tail(10)
                atr = round(recent_ranges.mean(), 4)
                if atr < 0.0003:
                    atr = 0.0015

            resistance, support, mid_range = calculate_support_resistance(df)
            if price > resistance * 0.998:
                location = "near recent resistance"
            elif price < support * 1.002:
                location = "near recent support"
            else:
                location = "middle of the recent range"

            last_5, momentum = get_market_context(df)

            news_bias, news_score = "unavailable", 0
            try:
                from fxnewsbias import Client
                fx = Client()
                s = fx.sentiment()
                news_bias = s.favours("EUR/USD") or "flat"
                news_score = s.spread("EUR/USD")
            except Exception as e:
                print(f"News error: {e}")

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

            hour = datetime.utcnow().hour
            if 7 <= hour < 10:
                session, volatility = "London Kill Zone", "high"
            elif 12 <= hour < 15:
                session, volatility = "New York Kill Zone", "high"
            elif 2 <= hour < 5:
                session, volatility = "Asian Kill Zone", "low"
            else:
                session, volatility = "Off-Session", "low"

            if latest['Close'] > latest['EMA_20'] and rsi > 50:
                tech_signal = "BUY"
            elif latest['Close'] < latest['EMA_20'] and rsi < 50:
                tech_signal = "SELL"
            else:
                tech_signal = "HOLD"

            verdict = tech_signal
            llm_reason = f"Technical signal: Price {'above' if latest['Close'] > latest['EMA_20'] else 'below'} 20 EMA, RSI at {rsi}."
            confidence = "MEDIUM"

            if HAS_GEMINI and os.environ.get("GEMINI_API_KEY"):
                try:
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    last_5_text = "\n".join([
                        f"  Candle {i+1}: O={c['open']} H={c['high']} L={c['low']} C={c['close']} ({c['direction']})"
                        for i, c in enumerate(last_5)
                    ])
                    
                    prompt = f"""You are a professional EUR/USD trader with 10 years of experience at a hedge fund. You are analyzing the 1-hour chart for a client named Boss.

MARKET DATA:
- Current Price: {price}
- RSI(14): {rsi}
- EMA20: {ema20}
- EMA50: {ema50}
- ATR(14): {atr}
- Recent Resistance: {resistance}
- Recent Support: {support}
- Mid-Range: {mid_range}
- Price Location: {location}

LAST 5 CANDLES:
{last_5_text}

MOMENTUM: {momentum}
CANDLESTICK PATTERNS: {', '.join(patterns_list)}
NEWS BIAS: {news_bias} (score: {news_score})
SESSION: {session} ({volatility} volatility)
TECHNICAL SIGNAL: {tech_signal}

YOUR TASK:
Give a professional trading verdict. Think like a real trader:
- Is the trend clear or choppy?
- Is the price at a good entry level or exhausted?
- Does the news support the technical setup?
- What invalidates this trade?

RULES:
- If signals conflict (e.g., bullish pattern but bearish news), say HOLD
- If price is mid-range with no clear pattern, say HOLD
- Only BUY or SELL with MEDIUM or HIGH confidence

Respond ONLY with valid JSON (no markdown, no extra text):
{{"verdict": "BUY/SELL/HOLD", "reason": "3-4 sentences explaining the setup, what confirms it, and what would invalidate it. Speak directly to Boss.", "confidence": "LOW/MEDIUM/HIGH"}}"""

                    response = model.generate_content(prompt)
                    raw_text = response.text.strip()
                    # Clean up any markdown formatting
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("```")[1]
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:]
                    raw_text = raw_text.strip()
                    
                    result = json.loads(raw_text)
                    verdict = result.get("verdict", tech_signal)
                    llm_reason = result.get("reason", llm_reason)
                    confidence = result.get("confidence", "MEDIUM")
                    print(f"Gemini verdict: {verdict} ({confidence})")
                except Exception as e:
                    print(f"Gemini error: {e}")

            if latest['Close'] > latest['Open']:
                candle_current = f"Bullish candle. Open: {latest['Open']:.4f}, Now: {price}. Buyers in control."
            else:
                candle_current = f"Bearish candle. Open: {latest['Open']:.4f}, Now: {price}. Sellers pushing down."

            c1h = f"1H candle at {price}. " + ("Above 20 EMA - uptrend." if latest['Close'] > latest['EMA_20'] else "Below 20 EMA - downtrend.")
            c4h = f"4H trend: " + ("bullish above moving averages." if latest['Close'] > latest['EMA_50'] else "bearish below moving averages.")

            entry = price
            if verdict == "BUY":
                sl = round(price - (atr * 1.5), 4)
                tp = round(price + (atr * 3.0), 4)
            elif verdict == "SELL":
                sl = round(price + (atr * 1.5), 4)
                tp = round(price - (atr * 3.0), 4)
            else:
                sl = round(price - (atr * 1.5), 4)
                tp = round(price + (atr * 3.0), 4)

            ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
            valid_until = (ist_now + timedelta(hours=1)).strftime("%H:%M IST")

            if random.random() < 0.5:
                wise_words = f"Joke: {random.choice(JOKES)}"
            else:
                wise_words = f"Motivation: {random.choice(MOTIVATIONS)}"

            return {
                "signal": verdict,
                "reason": llm_reason,
                "confidence": confidence,
                "price": str(price),
                "entry": str(entry),
                "sl": str(sl),
                "tp": str(tp),
                "atr": str(atr),
                "support": str(support),
                "resistance": str(resistance),
                "valid": valid_until,
                "candle_current": candle_current,
                "candle_1h": c1h,
                "candle_4h": c4h,
                "news_bias": news_bias,
                "news_score": news_score,
                "patterns": patterns_list,
                "session": session,
                "volatility": volatility,
                "wise_words": wise_words
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
        print("SUCCESS: data.json updated.")
        print(f"Price: {data['price']} | Signal: {data['signal']} | Confidence: {data['confidence']} | Valid: {data['valid']}")
    else:
        print("FAILED: Could not fetch data.")
