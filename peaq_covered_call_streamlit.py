import ccxt
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_crypto(symbol, lookback, timeframe):
        """retrieve a coin df based on X days for a timeframe
        eg 1h, 2h, 1d"""

        exchange = ccxt.kucoin()
        to_datetime = dt.datetime.now()
        from_datetime = to_datetime - dt.timedelta(days=lookback)
        from_datetime = from_datetime.strftime("%Y-%m-%d %H:%M:%S")
        from_timestamp = exchange.parse8601(from_datetime)
        data = exchange.fetch_ohlcv(
            symbol, timeframe, from_timestamp, limit=10000
            )
        header = ["time", "open", "high", "low", "close", "volumeto"]
        df = pd.DataFrame(data, columns=header).set_index("time")
        df.index = pd.to_datetime(df.index / 1000, unit="s")
        from_datetime = df.index[-1]
        from_datetime = from_datetime.strftime("%Y-%m-%d %H:%M:%S")
        from_timestamp = exchange.parse8601(from_datetime)
        from_datetime = df.index[-1]
        while from_datetime < to_datetime - dt.timedelta(days=0):
            data = exchange.fetch_ohlcv(
                symbol, timeframe, from_timestamp, limit=10000
                )
            header = ["time", "open", "high", "low", "close", "volumeto"]
            df1 = pd.DataFrame(data, columns=header).set_index("time")
            df1.index = pd.to_datetime(df1.index / 1000, unit="s")
            df1 = df1.iloc[1:, :]
            if len(df1)>0:
                from_datetime = df1.index[-1]
                from_datetime = from_datetime.strftime("%Y-%m-%d %H:%M:%S")
                from_timestamp = exchange.parse8601(from_datetime)
                df = pd.concat([df, df1])
                from_datetime = df.index[-1]
            else:
                break

        df.columns = map(str.lower, df.columns)
        df['symbol'] = symbol
        df['volume'] = df.close*df.volumeto
        df['returns'] = df.close.pct_change()
        df = df.dropna()
        return df

def rolling_N_return_distribution(df, N_periods,log_return=False, show_plot=True):
    """
    Calculate and visualize the distribution of rolling N_periods returns for a given OHLCV dataframe.

    Parameters:
    - df: DataFrame with datetime index and 'close' column (daily frequency)
    - log_return: If True, use log returns. If False, use simple % returns.
    - show_plot: If True, plot histogram.

    Returns:
    - summary: Descriptive statistics of the rolling 30-day returns
    - rolling_returns: Series of the rolling 30-day returns
    """
    # Resample to daily close prices
    daily_close = df['close'].resample('1D').last().dropna()

    # Compute rolling 30-day returns
    if log_return:
        rolling_returns = np.log(daily_close / daily_close.shift(N_periods)).dropna()
    else:
        rolling_returns = daily_close.pct_change(periods=N_periods).dropna()

    # Descriptive stats
    summary = rolling_returns.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

    # Optional plot
    if show_plot:
        rolling_returns.hist(bins=50, figsize=(10, 5), alpha=0.7)
        plt.axvline(rolling_returns.quantile(0.25), color='orange', label='25th percentile')
        plt.axvline(rolling_returns.median(), color='blue', label='Median')
        plt.axvline(rolling_returns.quantile(0.75), color='green', label='75th percentile')
        plt.title(f"Rolling {N_periods}-Day Return Distribution")
        plt.xlabel(f"{N_periods}-day return" + (" (log)" if log_return else " (%)"))
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return summary, rolling_returns

def allocate_tokens_by_strike(rolling_returns, strike_levels, premiums=None):
    """
    Dynamically allocate token weights to strike prices based on return history.

    Parameters:
    - rolling_returns: pd.Series of rolling 30D returns
    - strike_levels: list of strike levels as % of spot (e.g., [0.9, 0.95, 1.0, 1.05])
    - premiums: optional dict of {strike_level: premium}. If None, use simple decay model.

    Returns:
    - Dict of strike_level: token allocation weight (sums to 1)
    """

    strike_probs = {}
    value_scores = {}

    # Estimate exercise probabilities from return distribution
    for strike in strike_levels:
        # P(price return > strike % gain) = option gets exercised
        strike_return_threshold = strike - 1.0  # e.g., 0.9 strike → -10% return
        exercise_prob = (rolling_returns > strike_return_threshold).mean()
        # exercise_prob = number of occurences in the dataset that call was exercised / len(dataset) 
        strike_probs[strike] = exercise_prob
        # Estimate or use actual premium
        if premiums is None:
            # Simplified premium proxy (e.g., ATM = 0.07, decays from there)
            premium = max(0.01, 0.15 - abs(strike - 1.0) * 0.25)  # crude function
        else:
            premium = premiums.get(strike, 0.01)

        # Value = expected premium kept = (1 - P[exec]) × premium
        expected_value = (1 - exercise_prob) * premium
        value_scores[strike] = expected_value

    # Normalize to get weights
    total_score = sum(value_scores.values())
    weights = {k: v / total_score for k, v in value_scores.items()}

    return {
        "weights": weights,
        "exercise_probs": strike_probs,
        "value_scores": value_scores
    }

symbol = "PEAQ/USDT"
lookback = 270
timeframe = "1d"
df = fetch_crypto(symbol, lookback, timeframe)

rolling_N_days = [30, 60, 90, 180]

strike_levels = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]

# Real premiums from Keyrock quote sheet, with different expiry
premiums = [{
    0.85: 0.1783,
    0.9: 0.1433,   
    0.95: 0.1130,
    1.0: 0.0875,
    1.05: 0.0666,
    1.1: 0.0499,
    1.15: 0.0368,
    1.2: 0.0268,
    1.25: 0.0192,
    1.3: 0.0136
}, 
{
    0.85: 0.2030,
    0.9: 0.1716,   
    0.95: 0.1439,
    1.0: 0.1197,
    1.05: 0.0989,
    1.1: 0.0811,
    1.15: 0.0662,
    1.2: 0.0537,
    1.25: 0.0434,
    1.3: 0.0349
}, {
    0.85: 0.2226,
    0.9: 0.1931,   
    0.95: 0.1668,
    1.0: 0.1433,
    1.05: 0.1227,
    1.1: 0.1047,
    1.15: 0.0891,
    1.2: 0.0756,
    1.25: 0.0639,
    1.3: 0.0540
}, {
    0.85: 0.2639,
    0.9: 0.2372,   
    0.95: 0.2129,
    1.0: 0.1773,
    1.05: 0.1570,
    1.1: 0.1388,
    1.15: 0.1225,
    1.2: 0.1080,
    1.25: 0.0952,
    1.3: 0.0838
}]

exec_probs = []

for i in range(len(rolling_N_days)):
    test = rolling_N_return_distribution(df, rolling_N_days[i], log_return=False, show_plot=True)
    rolling_returns = test[1]
    result = allocate_tokens_by_strike(rolling_returns, strike_levels, premiums[i])

    print(f"Rolling {rolling_N_days[i]} Days")
    print("Strike\tExec%\tPrem\tScore\tWeight")
    tmp_probs = []
    for k in strike_levels:
        print(f"{k:.2f}\t{result['exercise_probs'][k]*100:.1f}%\t{premiums[i][k]:.3f}\t{result['value_scores'][k]:.4f}\t{result['weights'][k]*100:.1f}%")
        tmp_probs.append({k: result['exercise_probs'][k]})
    exec_probs.append(tmp_probs)

import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

# ---- CONFIG ----
st.set_page_config(page_title="PEAQ Covered Calls Simulator", layout="wide")
st.title("📈 PEAQ Covered Call Expected Value (All Tenors)")

# ---- INPUTS ----
# Uses the latest close from your df as spot
current_spot = float(df["close"].iloc[-1])

# Total PEAQ tokens you plan to allocate
total_peaq = st.number_input("Total $PEAQ tokens to allocate", min_value=0, value=1_200_000, step=50_000, help="Total tokens split across all tenors & strikes")

st.caption(f"Current Spot (from data): **{current_spot:,.4f} USDT**")

# ---- PREPARE OPTION UNIVERSE ----
# We expect:
# - rolling_N_days: list like [30,60,90,180]
# - strike_levels: e.g. [0.85, 0.9, ... , 1.3]
# - premiums: list of dicts (same length as rolling_N_days) mapping strike -> premium (as fraction of spot, e.g. 0.0875 = 8.75%)
# - exec_probs: list of lists-of-dicts (same length as rolling_N_days); each inner list same order as strike_levels with {strike: prob}
#   Example element: exec_probs[i][j] = { strike_levels[j]: probability }

# Build a tidy dataframe of all (tenor, strike) with premium & exercise probability
rows = []
for i, tenor in enumerate(rolling_N_days):
    # Build a dict strike -> exec_prob for quick lookup
    tenor_exec_map = {}
    for entry in exec_probs[i]:
        # each entry is {strike: prob}
        for k, v in entry.items():
            tenor_exec_map[float(k)] = float(v)

    for strike in strike_levels:
        prem = float(premiums[i][strike])  # fraction of spot
        p_ex = float(tenor_exec_map[strike])  # probability 0..1
        rows.append({
            "tenor_days": tenor,
            "strike_mult": strike,
            "premium_frac": prem,
            "p_exercise": p_ex,
        })

opt_df = pd.DataFrame(rows).sort_values(["tenor_days", "strike_mult"]).reset_index(drop=True)


# Compute per-token EV (in units of spot) for reference
opt_df["sale_per_token_in_spot"] = opt_df["strike_mult"] + opt_df["premium_frac"]
#opt_df["ev_per_token_in_spot"] = opt_df["premium_frac"] + opt_df["p_exercise"] * opt_df["strike_mult"]
#opt_df["premium_only_in_spot"] = opt_df["premium_frac"]

# ---- UI: ALLOCATION SLIDERS ----
st.markdown("### Allocation by Tenor & Strike")
st.write("Adjust how you'd like to allocate your **$PEAQ tokens** across all available tenors and strikes.")

alloc_perc = {}
total_cols = 4  # 4 columns works nicely on wide layout

for tenor in rolling_N_days:
    with st.expander(f"Tenor: {tenor} days", expanded=(tenor == rolling_N_days[0])):
        tenor_df = opt_df[opt_df["tenor_days"] == tenor].copy()
        cols = st.columns(total_cols)
        for idx, row in tenor_df.iterrows():
            label = f"{tenor}D @ {row['strike_mult']:.2f}x  (Prem {row['premium_frac']*100:.2f}%, p_exec {row['p_exercise']*100:.1f}%)"
            col = cols[idx % total_cols]
            with col:
                alloc_perc[(tenor, row["strike_mult"])] = st.slider(
                    label, min_value=0, max_value=100, value=0, step=1, key=f"slider_{tenor}_{row['strike_mult']}"
                )

# ---- VALIDATION ----
total_allocation = sum(alloc_perc.values())
st.markdown(f"**Total Allocation: {total_allocation}%**")
if total_allocation != 100:
    st.warning("Please ensure your total allocation equals 100% across all tenors & strikes.")
    st.stop()

# ---- CALCULATIONS ----
# Convert percentages to token counts
alloc_tokens = {
    k: (v / 100.0) * total_peaq
    for k, v in alloc_perc.items()
}

# Merge allocations back to the option table
alloc_rows = []
for (tenor, strike), pct in alloc_perc.items():
    sub = opt_df[(opt_df["tenor_days"] == tenor) & (opt_df["strike_mult"] == strike)].iloc[0]
    tokens = alloc_tokens[(tenor, strike)]

    p_ex = sub["p_exercise"]
    prem = sub["premium_frac"]
    #ev_per_token_spot = sub["ev_per_token_in_spot"]  # premium + p_ex * strike
    #prem_only_spot = sub["premium_only_in_spot"]

    # Convert to USDT
    #ev_usdt = tokens * ev_per_token_spot * current_spot
    #prem_only_usdt = tokens * prem_only_spot * current_spot
    sale_per_token_spot = sub["sale_per_token_in_spot"]     # strike + premium
    sale_usdt           = tokens * sale_per_token_spot * current_spot

    #spot_mult_pred = float(row["spot_mult_pred"])  # per-tenor, same across strikes
    ##spot_rev_usdt  = tokens * spot_mult_pred * current_spot

    alloc_rows.append({
        "Tenor (D)": tenor,
        "Strike (x Spot)": strike,
        "p_exercise": p_ex,
        "Premium (% of Spot)": prem * 100.0,
        "Allocation (%)": pct,
        "Tokens Allocated": tokens,
        #"EV per Token (in Spot)": ev_per_token_spot,
        #"EV (USDT)": ev_usdt,
        "Sale per Token relative to spot": sale_per_token_spot,
        "Sale (USDT, if exercised)": sale_usdt,
        #"Predicted Spot Multiplier": spot_mult_pred,                     
        #"Predicted Spot Revenue (USDT)": spot_rev_usdt
        #"Premium-kept-only EV (USDT)": prem_only_usdt,
    })

alloc_df = pd.DataFrame(alloc_rows).sort_values(["Tenor (D)", "Strike (x Spot)"])
#total_ev_usdt = alloc_df["EV (USDT)"].sum()
#total_prem_only_usdt = alloc_df["Premium-kept-only EV (USDT)"].sum()
total_sale_usdt = alloc_df["Sale (USDT, if exercised)"].sum()
#total_spot_rev  = alloc_df["Predicted Spot Revenue (USDT)"].sum()

# ---- SUMMARY KPIs ----
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total Tokens", f"{total_peaq:,.0f}")
with k2:
    st.metric("Total Sale Proceeds (USDT, if exercised)", f"{total_sale_usdt:,.2f}")
#with k3: st.metric("Predicted Spot Sale (USDT)", f"{total_spot_rev:,.2f}")

# ---- PIE: Allocation Share ----
st.markdown("### Allocation Mix")
labels = [f"{r['Tenor (D)']}D @ {r['Strike (x Spot)']:.2f}x" for _, r in alloc_df.iterrows()]
values = alloc_df["Allocation (%)"].tolist()
fig_alloc = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
fig_alloc.update_layout(title_text="Token Allocation by Tenor & Strike")
st.plotly_chart(fig_alloc, use_container_width=True)

# ---- BAR: EV by Bucket ----
st.markdown("### Expected Value by Bucket (USDT)")
fig_ev = go.Figure(data=[go.Bar(x=labels, y=alloc_df["Sale (USDT, if exercised)"] )])
fig_ev.update_layout(yaxis_title="Sale Proceeds (USDT)", xaxis_title="Tenor @ Strike")
st.plotly_chart(fig_ev, use_container_width=True)

# ---- DETAILS TABLE ----
st.markdown("### Details")
st.dataframe(
    alloc_df[[
        "Tenor (D)", "Strike (x Spot)", "p_exercise", "Premium (% of Spot)",
        "Allocation (%)", "Tokens Allocated",
        "Sale per Token relative to spot", "Sale (USDT, if exercised)",
        #"Predicted Spot Multiplier", "Predicted Spot Revenue (USDT)"
    ]]
)
st.caption("Sale per Token relative to spot if exercised = Strike + Premium. Sale (USDT) = Tokens × (Strike + Premium) × Spot.")

