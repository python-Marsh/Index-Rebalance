import re

import pandas as pd
import yfinance as yf


def fetch_ftse_tickers():
    """Fetch the top 350 tickers from wikipedia and summarise into a DataFrame.

    Returns:
        pd.DataFrame: DataFrame including the top 350 tickers with the following columns:
            - company
            - ticker
            - ftse industry classification benchmark sector
            - constituent: True if the ticker is a constituent of FTSE 100, otherwise False
            - current rank
    """
    # Get FTSE 100
    tables = pd.read_html("https://en.wikipedia.org/wiki/FTSE_100_Index")
    ftse_100 = tables[4]
    ftse_100["constituent"] = True
    ftse_100.columns = [re.sub(r"\[\d+\]", "", col).lower() for col in ftse_100.columns]
    # Get FTSE 250
    tables = pd.read_html("https://en.wikipedia.org/wiki/FTSE_250_Index")
    ftse_250 = tables[3]
    ftse_250["constituent"] = False  # FTSE 250 tickers are not constituents of FTSE 100
    ftse_250.columns = [re.sub(r"\[\d+\]", "", col).lower() for col in ftse_250.columns]
    # Concatenate FSTE 100 and FSTE 250
    ftse_df = pd.concat([ftse_100, ftse_250], ignore_index=True)
    ftse_df["current rank"] = range(1, len(ftse_df) + 1)
    ftse_df.set_index("current rank")
    return ftse_df


def fetch_yf_data(ftse_df: pd.DataFrame):
    """Fetch market data from yahoo finance.

    Args:
        ftse_df (pd.DataFrame): DataFrame including the top 350 tickers with the following columns:
            - company
            - ticker
            - ftse industry classification benchmark sector
            - constituent: True if the ticker is a constituent of FTSE 100, otherwise False
            - current rank
    Returns:
        pd.DataFrame: DataFrame including the top 350 tickers with the following columns:
            - ticker
            - company
            - price
            - market cap
            - float shares
            - constituent: True if the ticker is a constituent of FTSE 100, otherwise False
    """
    ftse_tickers = ftse_df["ticker"].tolist()
    company_names = ftse_df["company"].tolist()
    constituent = ftse_df["constituent"].tolist()

    # Append '.L' for Yahoo Finance (London Exchange)
    ftse_tickers = [t + ".L" for t in ftse_tickers]

    # Fetch data from yfinance
    data = []
    for ticker, company, cons in zip(ftse_tickers, company_names, constituent):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            info_dict = {
                "ticker": ticker,
                "company": company,
                "price": info.get("currentPrice"),
                "market cap": info.get("marketCap"),
                "float shares": info.get("floatShares"),
                "constituent": cons,
            }
            data.append(info_dict)
        except Exception as e:
            print(f"Error fetching {company}: {e}")

    # Convert to DataFrame and sort
    market_data_df = pd.DataFrame(data)
    market_data_df = market_data_df.sort_values(by="market cap", ascending=False)
    market_data_df.index = range(1, len(market_data_df) + 1)
    return market_data_df


def fetch_market_data():
    """_summary_

    Returns:
        pd.DataFrame: DataFrame including the top 350 tickers with the following columns:
            - ticker
            - company
            - price
            - market cap
            - float shares
            - constituent: True if the ticker is a constituent of FTSE 100, otherwise False
            - volatility
            - drift
    """
    ftse_df = fetch_ftse_tickers()
    market_data_df = fetch_yf_data(ftse_df)
    market_data_df["volatility"] = 0.2
    market_data_df["drift"] = 0
    return market_data_df
