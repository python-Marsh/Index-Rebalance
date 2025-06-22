import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
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


def get_drift_and_volatility(
    tickers: List[str], lookback_days: int = 252
) -> Dict[str, Tuple[float, float]]:
    """Computes the annualized drift and volatility for multiple stocks using historical
    data.

    Args:
        tickers (List[str]): List of ticker symbols (e.g., ['AAPL', 'MSFT']).
        lookback_days (int): Number of trading days to look back (default 252 ~ 1 year).

    Returns:
        Dict[str, Tuple[float, float]]: Dictionary mapping ticker to (annualized_drift,
        annualized_volatility)
    """
    results = {}
    current_date = pd.to_datetime(datetime.today())
    start_date = current_date - pd.tseries.offsets.BDay(lookback_days)

    # Download all tickers at once
    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=current_date.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
    )

    # If only one ticker, data won't have multi-level columns, fix this:
    if isinstance(tickers, str) or len(tickers) == 1:
        tickers = [tickers]

    for ticker in tickers:
        try:
            if ticker not in data.columns.levels[0]:
                raise ValueError(f"No data found for ticker {ticker}")

            ticker_data = data[ticker]

            # Prefer 'Adj Close', fallback to 'Close'
            if "Close" in ticker_data.columns:
                price_col = "Close"
            else:
                raise ValueError(f"No usable price column found for {ticker}.")

            ticker_data = ticker_data.dropna(subset=[price_col])
            available_days = len(ticker_data)

            if available_days < 2:
                raise ValueError(
                    f"Not enough data to compute drift and volatility for {ticker}"
                    "(only {available_days} day(s) available)."
                )

            # Compute daily log returns
            ticker_data["LogReturn"] = np.log(
                ticker_data[price_col] / ticker_data[price_col].shift(1)
            )
            log_returns = ticker_data["LogReturn"].dropna()

            daily_drift = log_returns.mean()
            daily_volatility = log_returns.std()

            annual_drift = daily_drift * 252
            annual_volatility = daily_volatility * np.sqrt(252)

            print(f"[INFO] Using {available_days} trading days of data for {ticker}")
            results[ticker] = (annual_drift, annual_volatility)

        except Exception as e:
            print(f"[ERROR] Could not process {ticker}: {e}")

    return results


def add_drift_and_volatility(
    df: pd.DataFrame, lookback_days: int = 252
) -> pd.DataFrame:
    """Takes a DataFrame of tickers and adds annualized drift and volatility columns.

    Args:
        df (pd.DataFrame): DataFrame with the following columns:
            - ticker
            - company
            - price
            - market cap
            - float shares
            - constituent: True if the ticker is a constituent of FTSE 100, otherwise False
        lookback_days (int): Number of trading days to look back for calculation (default 252).

    Returns:
        pd.DataFrame: The original DataFrame with two new columns 'drift' and 'volatility'.
    """

    tickers = df["ticker"].tolist()

    # Reuse the get_drift_and_volatility function you have or the version that downloads all
    # tickers at once:
    drift_vol_dict = get_drift_and_volatility(tickers, lookback_days=lookback_days)

    # Map results back to DataFrame, assign NaN for tickers with missing data
    df["drift"] = df["ticker"].map(
        lambda x: drift_vol_dict.get(x, (float("nan"), float("nan")))[0]
    )
    df["volatility"] = df["ticker"].map(
        lambda x: drift_vol_dict.get(x, (float("nan"), float("nan")))[1]
    )

    return df


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
    market_data_df = add_drift_and_volatility(market_data_df)
    return market_data_df
