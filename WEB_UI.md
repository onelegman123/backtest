# Streamlit Web UI

## Local run

From the project root, create and activate a virtual environment if needed, then install dependencies:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

The app uses the existing network-only yfinance and TWSE providers. Each operation may take time because it downloads current OHLCV data. Individual unavailable tickers are reported without stopping batch scans.

## Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select the repository and branch, and set the main file path to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

No API key is configured by this project. Availability of free upstream market-data endpoints can vary.
