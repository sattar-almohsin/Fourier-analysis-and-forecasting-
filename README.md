# Fourier-analysis-and-forecasting-

An advanced interactive web-based dashboard for financial time-series
analysis using Fourier Transform techniques.
The application is built with Streamlit and provides a
TradingView-inspired interface for analyzing price cycles, extracting
dominant frequencies, and generating forward projections based on
composite wave reconstruction.

------------------------------------------------------------------------

Note:
> This tool is designed for users interested in financial market analysis
through time-cycle methods.  
> The accuracy of any projection depends on the user's understanding of market cycles
and their own cycle-based research for the specific asset or market being analyzed.

------------------------------------------------------------------------

Features

Multi-Source Data Input

-   Yahoo Finance support via yfinance
-   CSV file upload for custom datasets
-   Automatic validation of data availability for selected analysis
    intervals

Advanced Fourier Analysis

-   FFT-based extraction of dominant frequency components
-   Peak/valley detection for identifying significant cycles
-   Signal reconstruction into a composite wave for forecasting
-   Three prediction modes:
    -   From last price
    -   From mean
    -   Normalized reconstruction

Interactive Visualizations

-   High-quality Plotly charts
-   Frequency domain visualization
-   Historical pattern analysis
-   Real-time response to parameter changes
-   Vertical markers for cycle peaks & troughs

User-Friendly Interface

-   Streamlit sidebar with real-time controls
-   Parameter tooltips and descriptions
-   Two-panel responsive layout
-   Tabbed results: Predictions, Components, Charts

Data Export

-   Downloadable CSV outputs for:
    -   Predictions
    -   Extracted frequency components

------------------------------------------------------------------------

System Architecture

Frontend

-   Streamlit for building interactive UI
-   Plotly for high-resolution, interactive charts

Backend / Processing

-   SciPy: FFT, peak detection
-   NumPy: numerical operations
-   Pandas: time-series data manipulation
-   Pure functional pipeline:
    Fetch → Process → Fourier Transform → Reconstruct → Visualize

Data Sources

-   Yahoo Finance (via yfinance) — no API key required
-   CSV file upload for custom market data

------------------------------------------------------------------------

Installation

Requirements

Python 3.11+

1. Clone the repository

git clone https://github.com/sattar-almohsin/Fourier-analysis-and-forecasting-.git
cd Fourier-analysis-and-forecasting-

2. Install dependencies

pip install -r requirements.txt

Or with pyproject.toml: pip install .

3. Run the app

streamlit run app.py

------------------------------------------------------------------------

Tech Stack

-   Python
-   Streamlit
-   NumPy
-   Pandas
-   SciPy
-   Plotly
-   yfinance

------------------------------------------------------------------------

Author

Developed by Sattar Almohsin
Developer & Engineer — AI, Energy, and Data Systems
