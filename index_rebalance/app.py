import copy
from datetime import datetime

import streamlit as st

from index_rebalance.utils.fetch_data import fetch_market_data
from index_rebalance.utils.simulation import (
    get_next_ftse_review_date,
    multiple_simulation,
)


def main():
    # Gather market data for all tickers
    if "fetched_market_data" not in st.session_state:
        st.session_state.fetched_market_data = fetch_market_data()
    if "market_data" not in st.session_state:
        st.session_state.market_data = copy.deepcopy(
            st.session_state.fetched_market_data
        )
    if "task" not in st.session_state:
        st.session_state.task = None
    if "n_simulation" not in st.session_state:
        st.session_state.n_simulation = 100

    task_options = ["Current Constituent", "General Forcast"]
    st.session_state.task = st.sidebar.selectbox(
        label="Task",
        options=task_options,
        index=(
            task_options.index(st.session_state.task) if st.session_state.task else None
        ),
        placeholder="Selet task",
    )

    if st.session_state.task is None:
        st.title("What are you lookng for")
    elif st.session_state.task == "Current Constituent":
        st.title("Current Constituent")
        st.write(
            f'Addition Threshold: {st.session_state.market_data.iloc[89]["market cap"]}'
        )
        st.write(
            f'Deletion Threshold: {st.session_state.market_data.iloc[109]["market cap"]}'
        )

        def reset_market_data():
            st.session_state.market_data = copy.deepcopy(
                st.session_state.fetched_market_data
            )

        st.button(label="reset", on_click=reset_market_data)
        # st.dataframe(
        #     st.session_state.market_data[
        #         ['ticker', 'company', 'price', 'market cap', 'constituent', 'volatility', 'drift']
        #     ]
        # )
        edited_df = st.data_editor(
            st.session_state.market_data[
                [
                    "ticker",
                    "company",
                    "price",
                    "market cap",
                    "constituent",
                    "volatility",
                    "drift",
                ]
            ],
            disabled=["ticker", "company"],
        )
        for col in ["price", "market cap", "constituent", "volatility", "drift"]:
            st.session_state.market_data[col] = edited_df[col]

    elif st.session_state.task == "General Forcast":
        st.title("General Forcast")
        st.session_state.n_simulation = st.slider(
            label="Number of simulations",
            min_value=100,
            max_value=1000,
            value=st.session_state.n_simulation,
        )
        current_date = datetime.today().strftime("%Y-%m-%d")
        rank_date = get_next_ftse_review_date(current_date)
        simulation_df = multiple_simulation(
            df=copy.deepcopy(st.session_state.market_data),
            rank_date=rank_date,
            n=st.session_state.n_simulation,
        )
        st.dataframe(simulation_df)


if __name__ == "__main__":
    main()
