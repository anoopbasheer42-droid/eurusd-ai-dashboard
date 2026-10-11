import yfinance as yf
import pandas as pd
import json
import os
import random
from datetime import datetime, timedelta
import time

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


def load_personal_context():
    try:
        with open("personal_context.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load personal_context.json: {e}")
        return None


def get_wise_words(personal):
    quotes = []
    if personal and isinstance(personal.get("quotes"), list):
        quotes = [q for q in personal["quotes"] if q and isinstance(q, str)]
    
    roll = random.random()
    if quotes and roll < 0.4:
        return f"Quote: \"{random.choice(quotes)}\""
    elif roll < 0.7:
        return f"Joke: {random.choice(JOKES)}"
    else:
        return f"Motivation: {random.choice(MOTIVATIONS)}"


def build_personal_section(personal):
    if not personal:
        return "No personal context provided."
    lines = []
    name = personal.get("name", "Boss")
    lines.append(f"Client's preferred name: {name}")
    family = personal.get("family", {})
    if family:
        lines.append("\nFAMILY:")
        if "wife" in family:
            w = family["wife"]
            lines.append(f"- Wife: {w.get('name')} (currently in {w.get('location')})")
        if "daughters" in family:
            for d in family["daughters"]:
                lines.append(f"- Daughter: {d.get('name')}, {d.get('grade')}, birthday {d.get('birthday')}")
        if "cat" in family:
            lines.append(f"- Cat: {family['cat']}")
        if "parents" in family:
            lines.append(f"- Parents: {', '.join(family['parents'])}")
        if "brothers" in family:
            lines.append(f"- Brothers: {', '.join(family['brothers'])}")
        if "location" in family:
            lines.append(f"- Family lives in {family['location']}")
    routine = personal.get("daily_routine", [])
    if routine:
        lines.append("\nDAILY ROUTINE REMINDERS:")
        for r in routine:
            lines.append(f"- {r}")
    style = personal.get("trading_style", {})
    if style:
        lines.append("\nTRADING STYLE:")
        lines.append(f"- Monthly goal: {style.get('monthly_goal')}")
        lines.append(f"- Preferred sessions: {', '.join(style.get('preferred_sessions', []))}")
        lines.append(f"- Risk tolerance: {style.get('risk_tolerance')}")
    personality = personal.get("ai_personality", {})
    if personality:
        lines.append("\nAI PERSONALITY:")
        lines.append(f"- Tone: {personality.get('tone')}")
        lines.append(f"- Always call the client: {personality.get('should_call_user', 'Boss')}")
    return "\n".join(lines)


def resample_to_4h(df_1h):
    df = df_1h.copy()
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return df_4h


def find_swing_points(df, lookback=60, window=3):
    recent = df.tail(lookback)
    highs = []
    lows = []
    for i in range(window, len(recent) - window):
        high_win = recent['High'].iloc[i-window:i+window+1]
        low_win = recent['Low'].iloc[i-window:i+window+1]
        if recent['High'].iloc[i] == high_win.max():
            highs.append(round(float(recent['High'].iloc[i]), 4))
        if recent['Low'].iloc[i] == low_win.min():
            lows.append(round(float(recent['Low'].iloc[i]), 4))
    return highs[-5:], lows[-5:]


def determine_trend(df):
    highs, lows = find_swing_points(df)
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1] > highs[-2]
        hl = lows[-1] > lows[-2]
        lh = highs[-1] < highs[-2]
        ll = lows[-1] < lows[-2]
        if hh and hl:
            return "Uptrend (higher highs + higher lows)"
        elif lh and ll:
            return "Downtrend (lower highs + lower lows)"
        else:
            return "Ranging/Consolidating"
    return "Unclear"


def candles_to_text(df, count, label):
    recent = df.tail(count)
    lines = [f"{label} (oldest -> newest):"]
    for idx, row in recent.iterrows():
        try:
            time_str = idx.strftime("%m-%d %H:%M")
        except Exception:
            time_str = str(idx)
        direction = "UP" if row['Close'] > row['Open'] else "DOWN"
        lines.append(
            f"  {time_str} | O:{row['Open']:.4f} H:{row['High']:.4f} L:{row['Low']:.4f} C:{row['Close']:.4f} ({direction})"
        )
    return "\n".join(lines)


def candles_to_json(df, count):
    recent = df.tail(count)
    result = []
    for idx, row in recent.iterrows():
        try:
            time_str = idx.strftime("%Y-%m-%d %H:%M")
        except Exception:
            time_str = str(idx)
        result.append({
            "time": time_str,
            "open": round(float(row['Open']), 4),
            "high": round(float(row['High']), 4),
            "low": round(float(row['Low']), 4),
            "close": round(float(row['Close']), 4)
        })
    return result


def call_gemini(prompt):
    if not (HAS_GEMINI and os.environ.get("GEMINI_API_KEY")):
        return None
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model_names = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-flash-latest"
    ]
    for m_name in model_names:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(prompt)
            print(f"Gemini model used: {m_name}")
            return response.text.strip()
        except Exception as e:
            print(f"Model {m_name} failed: {e}")
            continue
    return None


def get_analysis():
    for attempt in range(3):
        try:
            ticker = yf.Ticker("EURUSD=X")

            # Fetch 1H data (30 days)
            df_1h = ticker.history(period="30d", interval="1h")
            if df_1h.empty:
                time.sleep(5)
                continue
            df_1h = df_1h.dropna()

            # Fetch 15m data (5 days)
            df_15m = ticker.history(period="5d", interval="15m")
            if df_15m.empty:
                # Try longer period
                df_15m = ticker.history(period="30d", interval="15m")
            df_15m = df_15m.dropna() if not df_15m.empty else df_1h.tail(20)

            # Fetch 5m data (2 days)
            df_5m = ticker.history(period="2d", interval="5m")
            if df_5m.empty:
                df_5m = ticker.history(period="5d", interval="5m")
            df_5m = df_5m.dropna() if not df_5m.empty else df_1h.tail(30)

            # Fetch Daily (90 days)
            df_daily = ticker.history(period="90d", interval="1d")
            if df_daily.empty:
                df_daily = df_1h.resample('1D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min',
                    'Close': 'last', 'Volume': 'sum'
                }).dropna()
            else:
                df_daily = df_daily.dropna()

            # Resample 1H -> 4H
            df_4h = resample_to_4h(df_1h)

            # Indicators on 1H
            df_1h['EMA_20'] = df_1h['Close'].ewm(span=20, adjust=False).mean()
            df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
            delta = df_1h['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_1h['RSI'] = 100 - (100 / (1 + rs))

            tr1 = df_1h['High'] - df_1h['Low']
            tr2 = (df_1h['High'] - df_1h['Close'].shift()).abs()
            tr3 = (df_1h['Low'] - df_1h['Close'].shift()).abs()
            df_1h['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_1h['ATR'] = df_1h['TR'].rolling(window=14).mean()

            latest = df_1h.iloc[-1]
            price = round(float(latest['Close']), 4)
            rsi = round(float(latest['RSI']), 1)
            ema20 = round(float(latest['EMA_20']), 4)
            ema50 = round(float(latest['EMA_50']), 4)
            atr = round(float(latest['ATR']), 4)

            if pd.isna(atr) or atr < 0.0005:
                recent_ranges = (df_1h['High'] - df_1h['Low']).tail(10)
                atr = round(float(recent_ranges.mean()), 4)
                if atr < 0.0003:
                    atr = 0.0015

            # Support/Resistance
            recent50 = df_1h.tail(50)
            resistance = round(float(recent50['High'].max()), 4)
            support = round(float(recent50['Low'].min()), 4)
            mid_range = round((resistance + support) / 2, 4)

            if price > resistance * 0.998:
                location = "near recent resistance"
            elif price < support * 1.002:
                location = "near recent support"
            else:
                location = "middle of the recent range"

            swing_highs, swing_lows = find_swing_points(df_1h)
            trend = determine_trend(df_1h)

            # Candle text blocks
            candles_5m_text = candles_to_text(df_5m, 30, "5-MINUTE CANDLES")
            candles_15m_text = candles_to_text(df_15m, 20, "15-MINUTE CANDLES")
            candles_1h_text = candles_to_text(df_1h, 30, "1-HOUR CANDLES")
            candles_4h_text = candles_to_text(df_4h, 10, "4-HOUR CANDLES")
            candles_daily_text = candles_to_text(df_daily, 5, "DAILY CANDLES")

            candles_5m_json = candles_to_json(df_5m, 30)
            candles_15m_json = candles_to_json(df_15m, 20)
            candles_1h_json = candles_to_json(df_1h, 30)
            candles_4h_json = candles_to_json(df_4h, 10)
            candles_daily_json = candles_to_json(df_daily, 5)

            # Momentum
            ranges = (df_1h['High'] - df_1h['Low']).tail(3).tolist()
            momentum = "expanding" if ranges[-1] > ranges[-2] > ranges[-3] else \
                       "contracting" if ranges[-1] < ranges[-2] < ranges[-3] else "mixed"

            # News
            news_bias, news_score = "unavailable", 0
            try:
                from fxnewsbias import Client
                fx = Client()
                s = fx.sentiment()
                news_bias = s.favours("EUR/USD") or "flat"
                news_score = s.spread("EUR/USD")
            except Exception as e:
                print(f"News error: {e}")

            # Patterns
            patterns_list = ["No clear pattern"]
            if HAS_TA:
                try:
                    patterns = df_1h.ta.cdl_pattern(name="all")
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

            # Session
            hour = datetime.utcnow().hour
            if 7 <= hour < 10:
                session, volatility = "London Kill Zone", "high"
            elif 12 <= hour < 15:
                session, volatility = "New York Kill Zone", "high"
            elif 2 <= hour < 5:
                session, volatility = "Asian Kill Zone", "low"
            else:
                session, volatility = "Off-Session", "low"

            # Tech signal
            if latest['Close'] > latest['EMA_20'] and rsi > 50:
                tech_signal = "BUY"
            elif latest['Close'] < latest['EMA_20'] and rsi < 50:
                tech_signal = "SELL"
            else:
                tech_signal = "HOLD"

            # Personal context
            personal = load_personal_context()
            personal_section = build_personal_section(personal)

            ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
            ist_hour = ist_now.hour
            if ist_hour < 12:
                time_greeting = "Good morning"
            elif ist_hour < 17:
                time_greeting = "Good afternoon"
            else:
                time_greeting = "Good evening"

            # === BUILD PROMPT ===
            prompt = f"""You are a professional EUR/USD trader AND personal assistant to a client named Boss. You have 10 years of hedge fund experience. Think like a real trader, not a rulebook.

TIME: {ist_now.strftime('%H:%M IST, %A')} — Start with "{time_greeting}".

PERSONAL CONTEXT:
{personal_section}

=== MARKET DATA ===

CURRENT PRICE: {price}

INDICATORS (1H):
- RSI(14): {rsi}
- EMA20: {ema20}
- EMA50: {ema50}
- ATR(14): {atr}

KEY LEVELS:
- Resistance (50H high): {resistance}
- Support (50H low): {support}
- Mid-range: {mid_range}
- Price location: {location}

TREND STRUCTURE:
- Overall: {trend}
- Recent swing highs: {', '.join(str(h) for h in swing_highs) if swing_highs else 'None'}
- Recent swing lows: {', '.join(str(l) for l in swing_lows) if swing_lows else 'None'}

MOMENTUM: {momentum}

CANDLESTICK PATTERNS (latest 1H): {', '.join(patterns_list)}

NEWS BIAS: {news_bias} (score: {news_score})
SESSION: {session} ({volatility} volatility)
TECHNICAL SIGNAL: {tech_signal}

{candles_5m_text}

{candles_15m_text}

{candles_1h_text}

{candles_4h_text}

{candles_daily_text}

=== YOUR TASK ===

You have FIVE timeframes now: 5m, 15m, 1H, 4H, Daily. Use them like a real intraday trader:

- 5m -> ENTRY TIMING. Is the current moment right to enter? Any engulfing/rejection candles?
- 15m -> SHORT-TERM STRUCTURE. Is price making higher lows? Any micro resistance?
- 1H -> MAIN TREND. This is the anchor.
- 4H -> MEDIUM TREND. Confirms or conflicts with 1H.
- Daily -> BIG PICTURE. Weekly context.

Return TWO things:

1) TRADING VERDICT: BUY / SELL / HOLD with reason (3-5 sentences). Explain:
   - What the higher timeframes (1H, 4H, Daily) say about the trend
   - What the lower timeframes (5m, 15m) say about the ENTRY moment
   - If HTF says BUY but LTF says WAIT, tell Boss to wait for confirmation
   - If all 5 align, that's a high-confidence setup
   - What would invalidate this idea

2) PERSONAL MESSAGE: Short, friendly check-in (2-3 sentences) with a greeting, one reminder from Boss's routine, and a warm/funny line.

Respond ONLY with valid JSON:
{{"verdict": "BUY/SELL/HOLD", "reason": "3-5 sentences", "confidence": "LOW/MEDIUM/HIGH", "personal_message": "2-3 sentences"}}"""

            raw_response = call_gemini(prompt)

            verdict = tech_signal
            llm_reason = f"Technical signal: Price {'above' if latest['Close'] > latest['EMA_20'] else 'below'} 20 EMA, RSI at {rsi}."
            confidence = "MEDIUM"
            personal_message = f"{time_greeting}, Boss. Stay disciplined today."

            if raw_response:
                try:
                    clean = raw_response.strip()
                    if clean.startswith("```"):
                        clean = clean.split("```")[1]
                        if clean.startswith("json"):
                            clean = clean[4:]
                    clean = clean.strip()
                    result = json.loads(clean)
                    verdict = result.get("verdict", tech_signal)
                    llm_reason = result.get("reason", llm_reason)
                    confidence = result.get("confidence", "MEDIUM")
                    personal_message = result.get("personal_message", personal_message)
                except Exception as e:
                    print(f"JSON parse error: {e}")
                    print(f"Raw: {raw_response[:300]}")

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

            valid_until = (ist_now + timedelta(hours=1)).strftime("%H:%M IST")

            wise_words = get_wise_words(personal)

            return {
                "signal": verdict,
                "reason": llm_reason,
                "personal_message": personal_message,
                "confidence": confidence,
                "price": str(price),
                "entry": str(entry),
                "sl": str(sl),
                "tp": str(tp),
                "atr": str(atr),
                "support": str(support),
                "resistance": str(resistance),
                "mid_range": str(mid_range),
                "trend": trend,
                "swing_highs": [str(h) for h in swing_highs],
                "swing_lows": [str(l) for l in swing_lows],
                "valid": valid_until,
                "candle_current": candle_current,
                "candle_1h": c1h,
                "candle_4h": c4h,
                "candles_5m": candles_5m_json,
                "candles_15m": candles_15m_json,
                "candles_h1": candles_1h_json,
                "candles_h4": candles_4h_json,
                "candles_daily": candles_daily_json,
                "news_bias": news_bias,
                "news_score": news_score,
                "patterns": patterns_list,
                "session": session,
                "volatility": volatility,
                "wise_words": wise_words
            }
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
    return None


if __name__ == "__main__":
    data = get_analysis()
    if data:
        with open("data.json", "w") as f:
            json.dump(data, f, indent=4)
        print("=" * 60)
        print("SUCCESS: data.json updated.")
        print(f"Price: {data['price']} | Signal: {data['signal']} | Confidence: {data['confidence']}")
        print(f"Trend: {data['trend']}")
        print(f"Reason: {data['reason'][:200]}")
        print("=" * 60)
    else:
        print("FAILED: Could not fetch data.")
