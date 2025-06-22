from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd


def get_next_ftse_review_date(input_date):
    """Given a date string in the format '%Y-%m-%d', return the next upcoming FTSE UK
    Index Series review cutoff date as a string in the same format.

    FTSE UK indexes are reviewed quarterly in March, June, September, and December.
    The review is based on data from the end of the trading day on the Tuesday
    before the first Friday of the review month.

    Parameters:
        input_date (str): The input date in 'YYYY-MM-DD' format.

    Returns:
        str: The next review cutoff date (Tuesday before the first Friday of the
             review month) in 'YYYY-MM-DD' format.
             Returns None if no date is found within the next 2 years (very unlikely).
    """
    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    review_months = [3, 6, 9, 12]

    def get_review_cutoff(year, month):
        # First Friday of the review month
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
        # Tuesday before that Friday
        review_cutoff = first_friday - timedelta(days=3)
        return review_cutoff

    for offset in range(0, 24):  # Check next 2 years if needed
        month = review_months[offset % 4]
        year = input_date.year + (offset // 4)
        cutoff_date = get_review_cutoff(year, month)
        if input_date < cutoff_date:
            return cutoff_date.strftime("%Y-%m-%d")

    return None  # Fallback


get_next_ftse_review_date("2024-09-27")


def gbm(
    stock_price: List[float],
    volatility: List[float],
    current_date: str,
    rank_date: str,
    drift: List[float],
) -> List[float]:
    """Vectorized Geometric Brownian Motion (GBM) for multiple stocks with lists of
    parameters.

    Args:
        stock_price (List[float]): current prices of each stock
        volatility (List[float]): annual volatility for each stock
        current_date (str): current date in format '%Y-%m-%d'
        rank_date (str): future date in format '%Y-%m-%d'
        drift (List[float]): expected annual return for each stock

    Returns:
        List[float]: simulated prices at rank_date
    """
    # Convert lists to NumPy arrays
    stock_price = np.array(stock_price, dtype=float)
    volatility = np.array(volatility, dtype=float)
    drift = np.array(drift, dtype=float)

    n = len(stock_price)

    # Time setup
    current_date = pd.to_datetime(current_date)
    rank_date = pd.to_datetime(rank_date)
    days = len(pd.bdate_range(start=current_date, end=rank_date))
    dt = 1 / 252

    # Random normal draws: shape (n_stocks, n_days)
    Z = np.random.standard_normal((n, days))

    # GBM formula components
    mu_term = (drift - 0.5 * volatility**2)[:, None] * dt  # shape (n, 1)
    vol_term = (volatility[:, None] * np.sqrt(dt)) * Z  # shape (n, days)

    # Compute cumulative log returns
    log_returns = mu_term + vol_term  # shape (n, days)

    # Compute estimated prices
    estimated_stock_prices = stock_price[:] * np.exp(
        np.sum(log_returns, axis=-1)
    )  # shape (n)

    # Return final simulated price for each stock
    return estimated_stock_prices


def calculate_position(df: pd.DataFrame):
    """Compute and rank tickers by estimated market cap, add the following additional
    columns:

    - number of shares
    - estimated market cap
    - estimated market cap rank
    """
    df["number of shares"] = df["market cap"] / df["price"]
    df["estimated market cap"] = df["estimated stock price"] * df["number of shares"]
    df["estimated market cap rank"] = df["estimated market cap"].rank(
        ascending=False, method="min"
    )
    return df


def apply_ftse100_review_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the FTSE review rules
    1. A company will be inserted at the periodic review if it rises above 90th or above
    2. A company will be deleted at the periodic review if it falls below 111th or below
    3. A constant number of constituents will be maintained for the FTSE 100. Where a
    greater number of companies qualify to be inserted in an index than those qualifying
    to be deleted, the lowest-ranking constituents presently included in the index will
    be deleted to ensure that an equal number of companies are inserted and deleted at
    the periodic review. Likewise, where a greater number of companies qualify to be
    deleted than those qualifying to be inserted, the securities of the highest-ranking
    companies that are presently not included in the index will be inserted to match the
    number of companies being deleted at the periodic review.

    Based on the rules, add the following additional columns:
    - potential insert: True if the company is not currently an constituent and rises
    above 90th, otherwise False
    - potential delete: True if the company is currently an constituent and falls below
    111th, otherwise False
    - forced delete: True if the company is one of the lowest-ranking constituents
    presently included to be deleted when the number of potential insert exceeds the
    number of potential delete
    - forced insert: True if the company is one of the highest-ranking non-constituents
    presently included to be inserted when the number of potential delete exceeds the
    number of potential insert
    - revision decision: the value could be one of
        ['stay', 'added', 'removed', 'not included']

    Args:
        df (pd.DataFrame): DataFrame which column 'estimated market cap rank' indicates
        its position.
    """
    df = df.sort_values("estimated market cap rank").reset_index(drop=True)

    # Step 1: flag potential adds and deletes
    df["potential insert"] = (df["constituent"] == 0) & (
        df["estimated market cap rank"] <= 90
    )
    df["potential delete"] = (df["constituent"] == 1) & (
        df["estimated market cap rank"] >= 111
    )

    insert_candidates = df[df["potential insert"]]
    delete_candidates = df[df["potential delete"]]

    num_inserts = len(insert_candidates)
    num_deletes = len(delete_candidates)
    df["forced delete"] = False
    df["forced insert"] = False

    # Step 2: Enforce constant 100 membership by balancing inserts/deletes
    if num_inserts > num_deletes:
        extra = num_inserts - num_deletes
        # Remove more lowest-ranked current members
        current_constituents = df[(df["constituent"] == 1) & (~df["potential delete"])]
        extra_deletes = current_constituents.sort_values(
            "estimated market cap rank", ascending=False
        ).head(extra)
        df.loc[extra_deletes.index, "forced delete"] = True
    elif num_deletes > num_inserts:
        extra = num_deletes - num_inserts
        # Add more highest-ranked non-members
        non_constituents = df[(df["constituent"] == 0) & (~df["potential insert"])]
        extra_inserts = non_constituents.sort_values(
            "estimated market cap rank", ascending=True
        ).head(extra)
        df.loc[extra_inserts.index, "forced insert"] = True
    else:
        df["forced delete"] = False
        df["forced insert"] = False

    # Step 3: Set review decision
    def review_decision(row):
        if row["constituent"] == 1:
            if row.get("potential delete", False) or row.get("forced delete", False):
                return "removed"
            else:
                return "stay"
        else:
            if row.get("potential insert", False) or row.get("forced insert", False):
                return "added"
            else:
                return "not included"

    df["review decision"] = df.apply(review_decision, axis=1)
    assert len(df[df["review decision"] == "added"]) == len(
        df[df["review decision"] == "removed"]
    ), (
        f"{len(df[df['review decision'] == 'added'])} addition and "
        f"{len(df[df['review decision'] == 'removed'])} remove"
    )
    return df


def single_simulation(df: pd.DataFrame, rank_date: str):
    """Run a single simulation based on Geometric Brownian Motion (GBM)

    Step 1 - Run GBM to estimate the share price at rank date considering volatility &
    drift
    Step 2 - Calculate the market cap and rank accordingly
    Step 3 - Apply FTSE rules to determine the additions and deletions

    Args:
        pd.DataFrame: DataFrame including the top 350 tickers with the following
        columns:
            - ticker
            - company
            - price
            - market cap
            - float shares
            - constituent: True if the ticker is a constituent of FTSE 100, otherwise
            False
            - volatility
            - drift
        rank_date (str): the rank date of FTSE 100 review event

    Returns:
        df (pd.DataFrame): DataFrame with the following addtional columns:
        - number of shares
        - estimated market cap
        - estimated market cap rank
        - potential insert: True if the company is not currently an constituent and
        rises above 90th, otherwise False
        - potential delete: True if the company is currently an constituent and falls
        below 111th, otherwise False
        - forced delete: True if the company is one of the lowest-ranking constituents
        presently included to be deleted when the number of potential insert exceeds
        the number of potential delete
        - forced insert: True if the company is one of the highest-ranking
        non-constituents presently included to be inserted when the number of potential
        delete exceeds the number of potential insert
        - revision decision: the value could be one of
            ['stay', 'added', 'removed', 'not included']
    """

    current_date = datetime.today().strftime("%Y-%m-%d")
    df["estimated stock price"] = gbm(
        stock_price=df["price"],
        volatility=df["volatility"],
        current_date=current_date,
        rank_date=rank_date,
        drift=df["drift"],
    )
    df = calculate_position(df)
    df = apply_ftse100_review_rules(df)
    return df


def multiple_simulation(df: pd.DataFrame, rank_date: str, n: int):
    """Run multiple simulations based on Geometric Brownian Motion (GBM)

    Step 1 - Run GBM to estimate the share price at rank date considering volatility
    & drift
    Step 2 - Calculate the market cap and rank accordingly
    Step 3 - Apply FTSE rules to determine the additions and deletions
    Step 4 - Iterate until the number of simulation (n) is reached

    Args:
       pd.DataFrame: DataFrame including the top 350 tickers with the following columns:
            - ticker
            - company
            - price
            - market cap
            - float shares
            - constituent: True if the ticker is a constituent of FTSE 100, otherwise
            False
            - volatility
            - drift
        rank_date (str): the rank date of FTSE 100 review event
        n (int): number of simulations to run

    Returns:
        df (pd.DataFrame): DataFrame with the following columns:
            - ticker
            - company
            - price
            - market cap
            - volatility
            - drift
            - added: probability of addition recorded across all simulations.
            - removed: probability of deletions recorded across all simulations.
    """
    # Copy market data to be presented
    columns_to_copy = [
        "ticker",
        "company",
        "price",
        "market cap",
        "volatility",
        "drift",
    ]
    simulation_df = df[columns_to_copy].copy()
    # Initialise 'added' and 'removed' column
    simulation_df["added"] = 0
    simulation_df["removed"] = 0
    # Run n simulations
    for _ in range(n):
        df_sim = single_simulation(df, rank_date)

        # Extract tickers that were added or removed
        added_tickers = df_sim.loc[df_sim["review decision"] == "added", "ticker"]
        removed_tickers = df_sim.loc[df_sim["review decision"] == "removed", "ticker"]

        # Increment corresponding entries in simulation_df using .loc with ticker match
        simulation_df.loc[simulation_df["ticker"].isin(added_tickers), "added"] += 1
        simulation_df.loc[simulation_df["ticker"].isin(removed_tickers), "removed"] += 1

    added_df = simulation_df[simulation_df["added"] > 0]
    added_df["action"] = "addition"
    added_df["probability"] = added_df["added"] / n
    removed_df = simulation_df[simulation_df["removed"] > 0]
    removed_df["action"] = "delete"
    removed_df["probability"] = removed_df["removed"] / n
    simulation_df = pd.concat(
        [
            added_df[["ticker", "company", "action", "probability"]],
            removed_df[["ticker", "company", "action", "probability"]],
        ],
        ignore_index=True,
    )
    return simulation_df
