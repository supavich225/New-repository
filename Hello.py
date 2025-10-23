
import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="ตลาดหุ้น — กราฟวันนี้",
    page_icon="📈",
    layout="wide",
)

st.title("📈 ดูกราฟหุ้นของวันนี้")
st.sidebar.header("ตั้งค่า")

ticker = st.sidebar.text_input("สัญลักษณ์หุ้น (Ticker)", value="AAPL").upper()
interval = st.sidebar.selectbox(
    "ช่วงเวลา (interval)",
    ["1m", "2m", "5m", "15m", "30m", "60m"],
    index=0
)
# ใหม่: เลือกประเภทกราฟ (ตั้งค่าเริ่มต้นเป็น Candlestick)
chart_type = st.sidebar.selectbox("ประเภทกราฟ", ["Candlestick", "Area", "Bar", "Line"], index=0)

# --- Range selector (allow Max / custom range) ---
range_option = st.sidebar.selectbox("ช่วงเวลาแสดง (Range)", ["1d", "5d", "1mo", "6mo", "1y", "5y", "Max", "Custom"], index=0)
custom_start = None
custom_end = None
if range_option == "Custom":
    custom_start = st.sidebar.date_input("Start date", value=(datetime.now() - timedelta(days=365)).date())
    custom_end = st.sidebar.date_input("End date", value=datetime.now().date())

# candle & overlay options
show_ma20 = st.sidebar.checkbox("แสดง MA20", value=True)
candle_width_label = "ความกว้างแท่งเทียน"
if 'candle_width' not in st.session_state:
    st.session_state.candle_width = 8

st.sidebar.markdown(candle_width_label)
cw_col1, cw_col2, cw_col3 = st.sidebar.columns([1,1,1])
if cw_col1.button("➖", key="candle_minus"):
    st.session_state.candle_width = max(2, st.session_state.candle_width - 1)
cw_col2.markdown(f"**{st.session_state.candle_width}**", unsafe_allow_html=True)
if cw_col3.button("➕", key="candle_plus"):
    st.session_state.candle_width = min(40, st.session_state.candle_width + 1)

candle_width = st.session_state.candle_width


# --- Top movers recommendation ---
show_tops = st.sidebar.checkbox("แสดง Top movers (แนะนำหุ้น)", value=False)
default_watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
selected_recommendation = None
if show_tops:
    st.sidebar.markdown("**Watchlist เริ่มต้น**")
    st.sidebar.write(", ".join(default_watchlist))
    # Fetch daily prices for last 2 trading days to compute % change
    try:
        tops_df = yf.download(tickers=" ".join(default_watchlist), period="2d", interval="1d", group_by='ticker', progress=False)
    except Exception:
        tops_df = pd.DataFrame()

    movers = []
    if not tops_df.empty:
        # tops_df has a multi-index columns per ticker when multiple tickers requested
        for sym in default_watchlist:
            try:
                if isinstance(tops_df.columns, pd.MultiIndex):
                    sym_df = tops_df[sym]
                else:
                    # single-ticker fallback
                    sym_df = tops_df
                closes = sym_df['Close'].dropna()
                if len(closes) >= 2:
                    prev, last = closes.iloc[-2], closes.iloc[-1]
                    pct = (last - prev) / prev * 100 if prev != 0 else 0.0
                elif len(closes) == 1:
                    prev = closes.iloc[0]
                    last = prev
                    pct = 0.0
                else:
                    prev = last = pct = None
                movers.append({'symbol': sym, 'prev': prev, 'last': last, 'pct': pct})
            except Exception:
                movers.append({'symbol': sym, 'prev': None, 'last': None, 'pct': None})

        movers_df = pd.DataFrame(movers).sort_values('pct', ascending=False, na_position='last')
        st.sidebar.markdown("### Top movers (ตาม % วันนี้)")
        # show as table with select buttons
        for i, row in movers_df.head(5).iterrows():
            sym = row['symbol']
            pct = row['pct']
            last = row['last']
            st.sidebar.write(f"{sym}: {last if pd.notna(last) else '-'}  ({pct:+.2f}%" if pd.notna(pct) else f"{sym}: -")
            if st.sidebar.button(f"Load {sym}", key=f"load_{sym}"):
                selected_recommendation = sym

    else:
        st.sidebar.write("ไม่สามารถดึงข้อมูล Top movers ได้ในขณะนี้")

# If user clicked a recommended ticker button, override ticker
if selected_recommendation:
    ticker = selected_recommendation

period = "1d"  # default for intraday fetch


def compute_fetch_params(range_option, custom_start=None, custom_end=None, interval='1d'):
    """Return (period, start, end, interval) suitable for yfinance download.
    If start/end provided, use them (daily). If range_option == 'Max', return period='max'.
    For short intraday ranges (1d,5d) we keep given interval from sidebar.
    """
    if range_option == 'Custom' and custom_start and custom_end:
        return (None, custom_start, custom_end, '1d')  # ใช้ interval รายวันสำหรับช่วงเวลาที่กำหนดเอง
    if range_option == 'Max':
        return ('max', None, None, '1d')  # ใช้ interval รายวันสำหรับข้อมูลทั้งหมด
    
    # สำหรับช่วงเวลาอื่นๆ
    if range_option in ('1d', '5d'):
        return (range_option, None, None, interval)  # ใช้ interval ที่ผู้ใช้เลือกสำหรับข้อมูลรายวัน
    elif range_option in ('1mo', '6mo', '1y', '5y'):
        # ใช้ interval ที่เหมาะสมตามช่วงเวลา
        if range_option == '1mo':
            suggested_interval = '1h'  # ข้อมูลรายชั่วโมงสำหรับ 1 เดือน
        elif range_option == '6mo':
            suggested_interval = '1d'  # ข้อมูลรายวันสำหรับ 6 เดือน
        else:
            suggested_interval = '1d'  # ข้อมูลรายวันสำหรับ 1 ปีขึ้นไป
        return (range_option, None, None, suggested_interval)
    else:
        return (range_option, None, None, '1d')


@st.cache_data(ttl=60)
def fetch_history(ticker: str, period: str = None, start=None, end=None, interval: str = '1d'):
    """Fetch either intraday or historical daily data depending on params.
    Returns a dataframe with Datetime index.
    """
    try:
        if start is not None or period is None and interval == '1d' and start is None:
            # if start provided use start/end
            df = yf.download(tickers=ticker, start=start, end=end, interval=interval, progress=False)
        else:
            df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    except Exception:
        df = pd.DataFrame()
    # normalize index
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=60)
def fetch_intraday(ticker: str, period: str, interval: str):
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        # ถ้าไม่พบข้อมูลสำหรับ period=1d ลอง period=5d แล้วกรองวันล่าสุด
        try:
            df = yf.download(tickers=ticker, period="5d", interval=interval, progress=False)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            last_day = df.index.date[-1]
            df = df.loc[pd.to_datetime(df.index).date == last_day]
    return df

# determine fetch params from selected range
period_param, start_param, end_param, use_interval = compute_fetch_params(range_option, custom_start, custom_end, interval)
data = fetch_history(ticker, period=period_param, start=start_param, end=end_param, interval=use_interval)

if data is None or data.empty:
    st.warning("ไม่พบข้อมูลสำหรับสัญลักษณ์นี้หรือช่วงเวลา ลองเปลี่ยน ticker/interval.")
else:
    df = data.copy()
    # ensure index is datetime and create a Datetime column after reset
    df.index = pd.to_datetime(df.index)
    df = df.reset_index()
    # normalize column name for datetime (some yfinance returns 'Datetime' or unnamed first column)
    if 'Datetime' not in df.columns:
        df.rename(columns={df.columns[0]: 'Datetime'}, inplace=True)
    # ensure Datetime column is proper datetime dtype
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    except Exception:
        # fallback: create Datetime from index if available
        df['Datetime'] = pd.to_datetime(df.index)

    last_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else None
    change = (last_price - prev_price) if prev_price is not None else 0.0
    pct = (change / prev_price * 100) if prev_price not in (None, 0) else 0.0

    # Vertical zoom: padding around last price (in price units).
    # Use session state and top-right buttons to control vertical padding (±)
    if 'y_padding' not in st.session_state:
        st.session_state.y_padding = 2  # ลดค่า padding เริ่มต้นให้ zoom เข้าใกล้ราคามากขึ้น

    # initial domain computed from session state (buttons will appear in chart area and update this value)
    y_padding = float(st.session_state.y_padding)
    y_min = max(0, last_price - y_padding)
    y_max = last_price + y_padding

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label=f"{ticker} ราคา (ล่าสุด)", value=f"${last_price:,.2f}", delta=f"{pct:+.2f}%")
        # guard caption formatting if Datetime missing
        if 'Datetime' in df.columns and not df['Datetime'].isna().all():
            try:
                last_dt = pd.to_datetime(df['Datetime'].iloc[-1])
                st.caption(f"อัปเดตล่าสุด: {last_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                st.caption("อัปเดตล่าสุด: -")
        else:
            st.caption("อัปเดตล่าสุด: -")
        st.write("ช่วง:", interval, "| จำนวนแถว:", len(df))
        st.write("ประเภทกราฟ:", chart_type)

    with col2:
        # Render small control buttons at top-right of the chart area with the numeric
        # padding value shown between the minus and plus buttons.
        control_col1, control_col2, control_col3, control_col4 = st.columns([0.06, 0.04, 0.06, 0.01])
        with control_col1:
            # Minus: decrease padding
            if st.button("−", key="y_pad_minus"):
                st.session_state.y_padding = max(1, st.session_state.y_padding + 2)
        with control_col2:
            # Display the current numeric padding between the buttons
            control_col2.markdown(f"**{st.session_state.y_padding}**", unsafe_allow_html=True)
        with control_col3:
            # Plus: increase padding
            if st.button("＋", key="y_pad_plus"):
                st.session_state.y_padding = min(1000, st.session_state.y_padding - 2)
        with control_col4:
            # Reset small button
            if st.button("⟳", key="y_pad_reset"):
                st.session_state.y_padding = 10

        # Recompute domain after potential button clicks
        y_padding = float(st.session_state.y_padding)
        y_min = max(0, last_price - y_padding)
        y_max = last_price + y_padding

        # สร้างกราฟตามประเภทที่เลือก
        if chart_type == "Line":
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Datetime:T", title="Time"),
                    y=alt.Y("Close:Q", title="Price (USD)", scale=alt.Scale(domain=[y_min, y_max])),
                    tooltip=[
                        alt.Tooltip("Datetime:T", title="Time"),
                        alt.Tooltip("Close:Q", title="Close"),
                        alt.Tooltip("Open:Q", title="Open"),
                        alt.Tooltip("High:Q", title="High"),
                        alt.Tooltip("Low:Q", title="Low"),
                        alt.Tooltip("Volume:Q", title="Volume"),
                    ],
                )
                .interactive()
            )
        elif chart_type == "Area":
            chart = (
                alt.Chart(df)
                .mark_area(opacity=0.3)
                .encode(
                    x=alt.X("Datetime:T", title="Time"),
                    y=alt.Y("Close:Q", title="Price (USD)", scale=alt.Scale(domain=[y_min, y_max])),
                    tooltip=[
                        alt.Tooltip("Datetime:T", title="Time"),
                        alt.Tooltip("Close:Q", title="Close"),
                    ],
                )
                .interactive()
            )
        elif chart_type == "Bar":
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Datetime:T", title="Time"),
                    y=alt.Y("Close:Q", title="Price (USD)", scale=alt.Scale(domain=[y_min, y_max])),
                    tooltip=[
                        alt.Tooltip("Datetime:T", title="Time"),
                        alt.Tooltip("Close:Q", title="Close"),
                    ],
                )
                .interactive()
            )
        else:  # Candlestick
                # ปรับปรุง: วาดแท่งเทียนแบบมี wicks (high-low) และ bodies (open-close)
                # สี: ถ้า Close >= Open -> เขียว (ขึ้น) else แดง (ลง)
                up_color = "#26a69a"  # green
                down_color = "#ef5350"  # red

                # กำหนดเงื่อนไขสี
                color_cond = alt.condition(alt.datum.Close >= alt.datum.Open, alt.value(up_color), alt.value(down_color))

                # rule = wick (high-low)
                wick = (
                    alt.Chart(df)
                    .mark_rule(size=1)
                    .encode(
                        x='Datetime:T',
                        y=alt.Y('Low:Q', scale=alt.Scale(domain=[y_min, y_max])),
                        y2='High:Q',
                        color=color_cond,
                        tooltip=[
                            alt.Tooltip('Datetime:T', title='Time'),
                            alt.Tooltip('High:Q', title='High'),
                            alt.Tooltip('Low:Q', title='Low'),
                        ],
                    )
                )

                # body = rectangle between Open and Close. Use mark_bar with small continuous width.
                body = (
                    alt.Chart(df)
                    .mark_bar(size=candle_width)
                    .encode(
                        x='Datetime:T',
                        y=alt.Y('Open:Q', scale=alt.Scale(domain=[y_min, y_max])),
                        y2='Close:Q',
                        color=color_cond,
                        tooltip=[
                            alt.Tooltip('Datetime:T', title='Time'),
                            alt.Tooltip('Open:Q', title='Open'),
                            alt.Tooltip('Close:Q', title='Close'),
                            alt.Tooltip('High:Q', title='High'),
                            alt.Tooltip('Low:Q', title='Low'),
                            alt.Tooltip('Volume:Q', title='Volume'),
                        ],
                    )
                )

                # Optional: moving average overlay (simple 20-period) for clarity
                try:
                    df["ma20"] = df["Close"].rolling(window=20, min_periods=1).mean()
                    ma = (
                        alt.Chart(df)
                        .mark_line(color="#1f77b4", strokeWidth=2)
                        .encode(x='Datetime:T', y=alt.Y('ma20:Q', scale=alt.Scale(domain=[y_min, y_max])))
                    )
                    chart = alt.layer(wick, body, ma).properties(height=400).interactive()
                except Exception:
                    chart = alt.layer(wick, body).properties(height=400).interactive()

        st.altair_chart(chart, use_container_width=True)

    st.markdown("### ตารางข้อมูล (ตัวอย่าง 10 แถวล่าสุด)")
    st.dataframe(df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].tail(10), use_container_width=True)

    st.caption("ข้อมูลจาก yfinance (Yahoo Finance). Cache TTL=60s")