import streamlit as st
import plotly.graph_objects as go

# ---- Initialize session state for saving history ----
if "allocation_history" not in st.session_state:
    st.session_state.allocation_history = []


# ---- CONFIG ----
st.set_page_config(page_title="PEAQ Covered Calls Simulator", layout="centered")
st.title("📈 PEAQ Covered Call Profit Explorer")

# ---- PARAMETERS ----
total_peaq = 1_200_000  # total tokens

# Strike price options and their corresponding fixed premiums
strike_options = {
    "90%": (0.90*0.15, 13.00),
    "95%": (0.95*0.15, 9.70),
    "100%": (1.00*0.15, 7.45),
    "105%": (1.05*0.15, 5.50),
    "110%": (1.10*0.15, 3.86),
}

st.markdown("Adjust how you'd like to allocate your **1.2M $PEAQ tokens** across the available strike prices:")

# ---- SLIDERS FOR ALLOCATION ----
allocations = {}
cols = st.columns(len(strike_options))
for idx, (label, _) in enumerate(strike_options.items()):
    with cols[idx]:
        allocations[label] = st.slider(
            label, 0, 100, 0, 1, key=f"slider_{label}"
        )

# ---- VALIDATION ----
total_allocation = sum(allocations.values())
st.markdown(f"**Total Allocation: {total_allocation}%**")

if total_allocation != 100:
    st.warning("Please ensure your total allocation equals 100%.")
    st.stop()

# ---- PROFIT CALCULATION ----
labels = []
values = []
profits = []

total_profit = 0

for label, percent in allocations.items():
    strike, premium = strike_options[label]
    tokens = total_peaq * (percent / 100.0)
    profit = tokens * (strike * (1 + premium / 100.0))
    labels.append(f"{label} (Prem {premium}%)")
    values.append(percent)
    profits.append(profit)
    total_profit += profit
    

# ---- PIE CHART ----
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
fig.update_layout(title_text='Token Allocation by Strike Price')

st.plotly_chart(fig, use_container_width=True)

# ---- PREMIUM VS STRIKE PROFIT ----
# Compute premium and strike parts
total_premium_part = 0
total_conversion_part = 0

for label, percent in allocations.items():
    strike, premium = strike_options[label]
    tokens = total_peaq * (percent / 100)
    total_premium_part += tokens * strike * (premium / 100)
    total_conversion_part += tokens * strike

# Percent split
upfront_percent = (total_premium_part / total_profit) * 100
conversion_percent = 100 - upfront_percent

# Format text labels
premium_label = f"${total_premium_part:,.0f} ({upfront_percent:.1f}%)"
conversion_label = f"${total_conversion_part:,.0f} ({conversion_percent:.1f}%)"

# Bar chart showing profit composition
st.markdown("### 💡 Profit Composition")

fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(
    y=["Total Profit"],
    x=[total_premium_part],
    name="Upfront Premium",
    orientation='h',
    marker_color="#1f77b4",
    text=[premium_label],
    textposition='inside',
    hovertemplate='Upfront Premium: %{x:,.0f} USDT<extra></extra>',
))

fig_bar.add_trace(go.Bar(
    y=["Total Profit"],
    x=[total_conversion_part],
    name="Strike Conversion",
    orientation='h',
    marker_color="#2ca02c",
    text=[conversion_label],
    textposition='inside',
    hovertemplate='Strike Conversion: %{x:,.0f} USDT<extra></extra>',
))

fig_bar.update_layout(
    barmode='stack',
    height=180,
    xaxis_title="USDT",
    title="Total Profit Breakdown",
    showlegend=True,
    margin=dict(t=40, b=40)
)

st.plotly_chart(fig_bar, use_container_width=True)


# ---- TOTAL PROFIT & SAVE BUTTON ----
st.success(f"💰 **Total Profit: ${total_profit:,.2f} USDT**")

# Save allocation to session state
if st.button("📥 Save This Allocation"):
    saved_entry = {
        "Allocation (%)": allocations.copy(),
        "Total Profit": round(total_profit, 2)
    }
    st.session_state.allocation_history.append(saved_entry)
    st.success("✅ Allocation saved!")

# ---- OPTIONAL BAR CHART ----
with st.expander("🔍 View Profit per Strike"):
    bar_fig = go.Figure(data=[go.Bar(x=labels, y=profits)])
    bar_fig.update_layout(title="Profit by Strike Price", yaxis_title="Profit (USDT)", xaxis_title="Strike Price")
    st.plotly_chart(bar_fig, use_container_width=True)

# ---- SHOW ALLOCATION HISTORY ----
if st.session_state.allocation_history:
    st.markdown("### 🧾 Saved Allocation History")

    # Flatten history into table format
    rows = []
    for i, entry in enumerate(st.session_state.allocation_history, 1):
        row = {"#": i, "Total Profit (USDT)": entry["Total Profit"]}
        row.update(entry["Allocation (%)"])
        rows.append(row)

    st.dataframe(rows, use_container_width=True)


# ---- FOOTER ----
st.caption("Built with ❤️ using Streamlit + Plotly")
