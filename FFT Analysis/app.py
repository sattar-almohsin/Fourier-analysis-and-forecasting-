import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import yfinance as yf
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Fourier Analysis Dashboard",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main {
        background-color: #FAFAFA;
    }
    .stButton>button {
        background-color: #F0B90B;
        color: #1E2026;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #D4A009;
        color: #1E2026;
    }
    h1, h2, h3 {
        color: #212833;
    }
    .sidebar .sidebar-content {
        background-color: #1E2026;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-text {
        color: #2EBD85;
        font-weight: bold;
    }
    .warning-text {
        color: #F6465D;
        font-weight: bold;
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #E5E5E5;
    }
</style>
""",
            unsafe_allow_html=True)


def fetch_yfinance_data(symbol, start_date='2000-01-01'):
    try:
        stock_df = yf.download(symbol, start=start_date, progress=False)

        if stock_df.empty:
            return None

        stock_df['mid_price'] = stock_df['Close']
        stock_df.reset_index(inplace=True)

        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = [
                col[0] if col[1] == '' else col[1] for col in stock_df.columns
            ]

        stock_df = stock_df.dropna(subset=['Date', 'mid_price']).copy()
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df.set_index('Date', inplace=True)

        return stock_df[['mid_price']]
    except Exception as e:
        st.error(f"Error fetching yfinance data: {str(e)}")
        return None


def extract_frequency_components(df, top_n=2):
    if len(df) < 2:
        return []

    fft_result = np.fft.fft(df['mid_price'].values)
    frequencies = np.fft.fftfreq(len(fft_result), d=1)
    amplitudes = np.abs(fft_result)

    positive_freq_indices = np.where(frequencies > 0)[0]

    if len(positive_freq_indices) == 0:
        return []

    sorted_indices = np.argsort(amplitudes[positive_freq_indices])[::-1]
    top_indices = min(top_n, len(sorted_indices))
    top_frequencies = positive_freq_indices[sorted_indices[:top_indices]]

    component_params = []
    for freq_idx in top_frequencies:
        amplitude = 2 * amplitudes[freq_idx] / len(fft_result)
        phase = np.angle(fft_result[freq_idx])
        frequency = frequencies[freq_idx]
        period_days = 1 / frequency if frequency != 0 else np.inf
        component_params.append({
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase,
            'period_days': period_days
        })

    return component_params


def analyze_periods(stock_df, start_date, period_length, num_periods,
                    intervals, top_n_components):
    all_components = []
    period_dates = []
    cumulative_days = 0
    period_info = []

    for i in range(num_periods):
        cumulative_days += intervals[i]
        period_end = start_date - timedelta(days=cumulative_days)
        period_start = period_end - timedelta(days=period_length)

        try:
            period_data = stock_df.loc[period_start:period_end].copy()

            if len(period_data) < 2:
                continue

            components = extract_frequency_components(period_data,
                                                      top_n=top_n_components)

            if components:
                all_components.extend(components)
                period_dates.append((period_start, period_end))
                period_info.append({
                    'period_num': i + 1,
                    'start': period_start,
                    'end': period_end,
                    'components': components,
                    'data_points': len(period_data)
                })
        except Exception as e:
            continue

    return all_components, period_dates, period_info


def create_composite_wave(all_components, period_length, period_dates):
    time_points = np.arange(period_length)
    composite_wave = np.zeros(period_length)

    for comp in all_components:
        amplitude = comp['amplitude']
        frequency = comp['frequency']
        phase = comp['phase']

        wave = amplitude * np.sin(2 * np.pi * frequency * time_points + phase)
        composite_wave += wave

    if len(period_dates) > 0:
        composite_wave = composite_wave / len(period_dates)

    return composite_wave


def calculate_price_predictions(stock_df, start_date, composite_wave,
                                period_dates):
    try:
        last_known_price = stock_df.loc[:start_date]['mid_price'].iloc[
            -1] if start_date in stock_df.index else stock_df[
                'mid_price'].iloc[-1]
    except:
        last_known_price = stock_df['mid_price'].iloc[-1]

    all_period_prices = []
    for p_start, p_end in period_dates:
        period_prices = stock_df.loc[p_start:p_end]['mid_price'].values
        all_period_prices.extend(period_prices)

    if len(all_period_prices) > 0:
        mean_price = np.mean(all_period_prices)
        std_price = np.std(all_period_prices)
    else:
        mean_price = last_known_price
        std_price = 0

    predicted_prices_from_last = last_known_price + composite_wave
    predicted_prices_from_mean = mean_price + composite_wave

    if std_price > 0 and np.std(composite_wave) > 0:
        normalized_wave = (composite_wave / np.std(composite_wave)) * std_price
        predicted_prices_normalized = mean_price + normalized_wave
    else:
        predicted_prices_normalized = predicted_prices_from_mean

    return {
        'last_price': last_known_price,
        'mean_price': mean_price,
        'std_price': std_price,
        'from_last': predicted_prices_from_last,
        'from_mean': predicted_prices_from_mean,
        'normalized': predicted_prices_normalized
    }


st.title("📈 Fourier Analysis Dashboard")
st.markdown("Advanced Time Series Analysis with FFT")

with st.sidebar:
    st.markdown("### ⚙️ Analysis Parameters")

    data_source = st.radio("Data Source", ["yFinance (Stocks)", "Upload CSV"],
                           help="Select where to fetch your price data from")

    if data_source == "yFinance (Stocks)":
        symbol = st.text_input(
            "Stock Symbol",
            value="GLD",
            help="Enter a stock symbol (e.g., GLD, AAPL, TSLA)")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="CSV must have 'Date' and 'mid_price' columns")

    st.markdown("---")
    st.markdown("### 📅 Time Period Settings")

    start_date = st.date_input(
        "Analysis Start Date",
        value=datetime.now().date(),
        help="The reference date from which to predict future periods")

    period_length = st.number_input(
        "Period Length (Days)",
        min_value=10,
        max_value=365,
        value=88,
        help="Length of each analysis period in days")

    num_periods = st.number_input(
        "Number of Historical Periods",
        min_value=1,
        max_value=10,
        value=5,
        help="How many historical periods to analyze")

    st.markdown("### ⏱️ Period Intervals")
    st.markdown("*Days between each period*")

    intervals = []
    for i in range(int(num_periods)):
        if i == 0:
            label = f"Start Date to Period 1"
        else:
            label = f"Period {i} to Period {i+1}"

        interval = st.number_input(
            label,
            min_value=1,
            value=88,
            key=f"interval_{i}",
            help=f"Days backward from previous reference point")
        intervals.append(interval)

    st.markdown("---")
    st.markdown("### 🌊 Frequency Components")

    top_n_components = st.number_input(
        "Components per Period",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of top frequency components to extract from each period")

    st.markdown("---")
    analyze_button = st.button("Run Analysis", use_container_width=True)

if analyze_button:
    stock_df = None

    with st.spinner("Fetching data..."):
        if data_source == "yFinance (Stocks)":
            stock_df = fetch_yfinance_data(symbol)
            data_label = f"{symbol} (Stock)"
        else:
            if uploaded_file is not None:
                try:
                    stock_df = pd.read_csv(uploaded_file)
                    stock_df = stock_df.dropna(
                        subset=['Date', 'mid_price']).copy()
                    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                    stock_df.set_index('Date', inplace=True)
                    stock_df = stock_df[['mid_price']]
                    data_label = "Uploaded CSV"
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
            else:
                st.warning("Please upload a CSV file")

    if stock_df is not None and not stock_df.empty:
        st.success(
            f"✅ Successfully loaded {len(stock_df)} data points from {data_label}"
        )

        start_date_dt = pd.to_datetime(start_date)

        cumulative_lookback = sum(intervals) + period_length
        earliest_required_date = start_date_dt - timedelta(
            days=cumulative_lookback)
        earliest_available_date = stock_df.index.min()
        latest_available_date = stock_df.index.max()
        available_days = (latest_available_date - earliest_available_date).days

        st.info(
            f"📅 Data available from **{earliest_available_date.strftime('%Y-%m-%d')}** to **{latest_available_date.strftime('%Y-%m-%d')}** ({available_days} days)"
        )

        if start_date_dt > latest_available_date:
            st.error(
                f"❌ Analysis start date ({start_date}) is beyond the available data range. Please choose a date on or before {latest_available_date.strftime('%Y-%m-%d')}."
            )
            st.stop()

        if earliest_required_date < earliest_available_date:
            missing_days = (earliest_available_date -
                            earliest_required_date).days
            st.warning(
                f"⚠️ Requested analysis requires data from **{earliest_required_date.strftime('%Y-%m-%d')}**, but data is only available from **{earliest_available_date.strftime('%Y-%m-%d')}** ({missing_days} days short). Some periods may be skipped. Consider reducing period intervals or the number of periods."
            )

        with st.spinner("Analyzing frequency components..."):
            all_components, period_dates, period_info = analyze_periods(
                stock_df, start_date_dt, period_length, num_periods, intervals,
                top_n_components)

        if len(period_dates) < num_periods:
            st.warning(
                f"⚠️ Only {len(period_dates)} out of {num_periods} requested periods had sufficient data for analysis. Consider adjusting your parameters."
            )

        if len(all_components) > 0:
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Periods Analyzed", len(period_dates))
            with col2:
                st.metric("Total Components", len(all_components))
            with col3:
                st.metric("Prediction Days", period_length)
            with col4:
                avg_period = np.mean([
                    comp['period_days'] for comp in all_components
                    if comp['period_days'] != np.inf
                ])
                st.metric("Avg Period (Days)", f"{avg_period:.1f}")

            composite_wave = create_composite_wave(all_components,
                                                   period_length, period_dates)
            future_dates = pd.date_range(start=start_date_dt,
                                         periods=period_length,
                                         freq='D')
            predictions = calculate_price_predictions(stock_df, start_date_dt,
                                                      composite_wave,
                                                      period_dates)

            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Historical Periods", " Frequency Components",
                "🔮 Price Predictions", "📋 Detailed Results"
            ])

            with tab1:
                st.markdown("### Historical Periods Used in Analysis")

                fig1 = go.Figure()

                two_years_ago = start_date_dt - timedelta(days=730)
                historical_data = stock_df.loc[two_years_ago:start_date_dt][
                    'mid_price']

                if len(historical_data) > 0:
                    fig1.add_trace(
                        go.Scatter(x=historical_data.index,
                                   y=historical_data.values,
                                   mode='lines',
                                   name='Historical Price',
                                   line=dict(color='#999999', width=1),
                                   opacity=0.5))

                colors = [
                    '#F0B90B', '#2EBD85', '#F6465D', '#1E90FF', '#FF69B4',
                    '#00CED1', '#FFD700', '#FF6347', '#7B68EE', '#32CD32'
                ]

                for i, (p_start, p_end) in enumerate(period_dates):
                    period_data = stock_df.loc[p_start:p_end]['mid_price']
                    fig1.add_trace(
                        go.Scatter(x=period_data.index,
                                   y=period_data.values,
                                   mode='lines',
                                   name=f'Period {i+1}',
                                   line=dict(color=colors[i % len(colors)],
                                             width=2)))

                fig1.add_shape(type="line",
                               x0=start_date_dt,
                               x1=start_date_dt,
                               y0=0,
                               y1=1,
                               yref="paper",
                               line=dict(color="#F6465D", width=2,
                                         dash="dash"))

                fig1.add_annotation(x=start_date_dt,
                                    y=1,
                                    yref="paper",
                                    text="Start Date",
                                    showarrow=False,
                                    yshift=10,
                                    font=dict(color="#F6465D", size=12))

                fig1.update_layout(
                    title="Historical Prices & Analyzed Periods",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    legend=dict(orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1))

                st.plotly_chart(fig1, use_container_width=True)

            with tab2:
                st.markdown("### Individual Frequency Components")

                fig2 = go.Figure()

                time_points = np.arange(period_length)

                for i, comp in enumerate(all_components[:20]):
                    amplitude = comp['amplitude']
                    frequency = comp['frequency']
                    phase = comp['phase']
                    period_days = comp['period_days']

                    wave = amplitude * np.sin(2 * np.pi * frequency *
                                              time_points + phase)

                    fig2.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=wave,
                            mode='lines',
                            name=f'f={frequency:.4f} (P={period_days:.1f}d)',
                            opacity=0.6,
                            line=dict(width=1)))

                fig2.add_trace(
                    go.Scatter(x=time_points,
                               y=composite_wave,
                               mode='lines',
                               name='Composite Wave',
                               line=dict(color='#1E2026', width=3),
                               opacity=1))

                fig2.update_layout(
                    title=
                    "Individual Frequency Components (Top 20) & Composite Wave",
                    xaxis_title="Days",
                    yaxis_title="Amplitude",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    showlegend=True)

                st.plotly_chart(fig2, use_container_width=True)

            with tab3:
                st.markdown("### Price Predictions")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Last Known Price",
                              f"${predictions['last_price']:.2f}")
                with col2:
                    st.metric("Historical Mean",
                              f"${predictions['mean_price']:.2f}")
                with col3:
                    st.metric("Std Deviation",
                              f"${predictions['std_price']:.2f}")

                fig3 = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("Prediction from Last Price",
                                    "Normalized Prediction"),
                    vertical_spacing=0.12,
                    row_heights=[0.5, 0.5])

                fig3.add_trace(go.Scatter(x=future_dates,
                                          y=predictions['from_last'],
                                          mode='lines',
                                          name='Prediction',
                                          line=dict(color='#2EBD85', width=2)),
                               row=1,
                               col=1)

                upper_bound = predictions['from_last'] + predictions[
                    'std_price']
                lower_bound = predictions['from_last'] - predictions[
                    'std_price']

                fig3.add_trace(go.Scatter(x=future_dates,
                                          y=upper_bound,
                                          mode='lines',
                                          name='Upper Bound',
                                          line=dict(width=0),
                                          showlegend=False),
                               row=1,
                               col=1)

                fig3.add_trace(go.Scatter(x=future_dates,
                                          y=lower_bound,
                                          mode='lines',
                                          name='Confidence Band',
                                          line=dict(width=0),
                                          fillcolor='rgba(46, 189, 133, 0.2)',
                                          fill='tonexty'),
                               row=1,
                               col=1)

                fig3.add_hline(y=predictions['last_price'],
                               line_dash="dash",
                               line_color="#F6465D",
                               annotation_text=
                               f"Last Price: ${predictions['last_price']:.2f}",
                               row=1,
                               col=1)

                fig3.add_trace(go.Scatter(x=future_dates,
                                          y=predictions['normalized'],
                                          mode='lines',
                                          name='Normalized Prediction',
                                          line=dict(color='#F0B90B', width=2)),
                               row=2,
                               col=1)

                peaks, _ = find_peaks(predictions['normalized'])
                valleys, _ = find_peaks(-predictions['normalized'])

                if len(peaks) > 0:
                    fig3.add_trace(go.Scatter(
                        x=future_dates[peaks],
                        y=predictions['normalized'][peaks],
                        mode='markers',
                        name='Predicted Peaks',
                        marker=dict(color='#F6465D',
                                    size=10,
                                    symbol='triangle-down')),
                                   row=2,
                                   col=1)

                if len(valleys) > 0:
                    fig3.add_trace(go.Scatter(
                        x=future_dates[valleys],
                        y=predictions['normalized'][valleys],
                        mode='markers',
                        name='Predicted Valleys',
                        marker=dict(color='#2EBD85',
                                    size=10,
                                    symbol='triangle-up')),
                                   row=2,
                                   col=1)

                resistance = np.max(predictions['normalized'])
                support = np.min(predictions['normalized'])

                fig3.add_hline(
                    y=resistance,
                    line_dash="dot",
                    line_color="#F6465D",
                    annotation_text=f"Resistance: ${resistance:.2f}",
                    row=2,
                    col=1)

                fig3.add_hline(y=support,
                               line_dash="dot",
                               line_color="#2EBD85",
                               annotation_text=f"Support: ${support:.2f}",
                               row=2,
                               col=1)

                fig3.update_xaxes(title_text="Date", row=1, col=1)
                fig3.update_xaxes(title_text="Date", row=2, col=1)
                fig3.update_yaxes(title_text="Price", row=1, col=1)
                fig3.update_yaxes(title_text="Price", row=2, col=1)

                fig3.update_layout(height=800,
                                   hovermode='x unified',
                                   template='plotly_white',
                                   showlegend=True)

                st.plotly_chart(fig3, use_container_width=True)

                st.markdown("#### 📊 Prediction Summary")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Price Range:**")
                    st.markdown(
                        f"- Maximum: <span class='success-text'>${np.max(predictions['normalized']):.2f}</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"- Minimum: <span class='warning-text'>${np.min(predictions['normalized']):.2f}</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"- Average: ${np.mean(predictions['normalized']):.2f}"
                    )

                with col2:
                    st.markdown("**Key Points:**")
                    if len(peaks) > 0:
                        st.markdown(f"- Predicted Peaks: {len(peaks)}")
                        st.markdown(
                            f"- First Peak: {future_dates[peaks[0]].strftime('%Y-%m-%d')}"
                        )
                    if len(valleys) > 0:
                        st.markdown(f"- Predicted Valleys: {len(valleys)}")
                        st.markdown(
                            f"- First Valley: {future_dates[valleys[0]].strftime('%Y-%m-%d')}"
                        )

            with tab4:
                st.markdown("### Extracted Frequency Components by Period")

                for period in period_info:
                    with st.expander(
                            f"📅 Period {period['period_num']}: {period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')} ({period['data_points']} points)"
                    ):
                        components_data = []
                        for j, comp in enumerate(period['components'], 1):
                            components_data.append({
                                'Component':
                                j,
                                'Frequency':
                                f"{comp['frequency']:.6f}",
                                'Period (Days)':
                                f"{comp['period_days']:.2f}"
                                if comp['period_days'] != np.inf else "∞",
                                'Amplitude':
                                f"{comp['amplitude']:.4f}",
                                'Phase (rad)':
                                f"{comp['phase']:.4f}"
                            })

                        df_components = pd.DataFrame(components_data)
                        st.dataframe(df_components, use_container_width=True)

                st.markdown("---")
                st.markdown("### 💾 Download Results")

                predictions_df = pd.DataFrame({
                    'Date':
                    future_dates,
                    'Composite_Wave':
                    composite_wave,
                    'Price_From_Last':
                    predictions['from_last'],
                    'Price_Normalized':
                    predictions['normalized'],
                    'Upper_Bound':
                    predictions['normalized'] + predictions['std_price'],
                    'Lower_Bound':
                    predictions['normalized'] - predictions['std_price']
                })

                csv_predictions = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv_predictions,
                    file_name=f"fourier_predictions_{start_date}.csv",
                    mime="text/csv",
                    use_container_width=True)

                components_export = []
                for period in period_info:
                    for comp in period['components']:
                        components_export.append({
                            'Period':
                            period['period_num'],
                            'Period_Start':
                            period['start'].strftime('%Y-%m-%d'),
                            'Period_End':
                            period['end'].strftime('%Y-%m-%d'),
                            'Frequency':
                            comp['frequency'],
                            'Period_Days':
                            comp['period_days'],
                            'Amplitude':
                            comp['amplitude'],
                            'Phase':
                            comp['phase']
                        })

                df_components_export = pd.DataFrame(components_export)
                csv_components = df_components_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download Components (CSV)",
                    data=csv_components,
                    file_name=f"fourier_components_{start_date}.csv",
                    mime="text/csv",
                    use_container_width=True)

        else:
            st.error(
                "❌ No frequency components could be extracted. Please check your parameters and try again."
            )

    elif stock_df is None:
        st.error(
            "❌ Failed to load data. Please check your input and try again.")
else:
    st.info(
        "👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to begin."
    )

    st.markdown("---")
    st.markdown("### 📖 How to Use")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1️⃣ Select Data Source")
        st.markdown("""
        - **yFinance**: Stock market data (GLD, AAPL, TSLA, etc.)
        - **CSV Upload**: Custom data with Date and mid_price columns
        """)

    with col2:
        st.markdown("#### 2️⃣ Configure Parameters")
        st.markdown("""
        - Set your analysis start date
        - Define period length (e.g., 88 days)
        - Choose number of historical periods
        - Set intervals between periods
        - Select components per period
        """)

    with col3:
        st.markdown("#### 3️⃣ Analyze & Predict")
        st.markdown("""
        - Click 'Run Analysis' button
        - View historical patterns
        - Examine frequency components
        - Get price predictions
        - Download results as CSV
        """)
