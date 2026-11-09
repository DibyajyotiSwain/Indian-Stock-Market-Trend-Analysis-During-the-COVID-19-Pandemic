"""Indian Stock Market Analytics Dashboard

Python | Pandas | Plotly | Dash | Scikit-learn

Run:  python app.py
Needs: data/All_Stocks_Data_Cleaned.csv  and  assets/style.css
Open:  http://127.0.0.1:8050
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

# ------------------------------------------------------------------ config
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "All_Stocks_Data_Cleaned.csv"
REQUIRED_COLUMNS = ["Symbol", "Date", "Closing_Price"]

MAX_ML_ROWS = 60_000
MAX_BOX_POINTS = 50_000

INK, MUTED, GRID = "#1b2559", "#6b7490", "#e3e7f0"
GAIN, LOSS, SAFFRON, BLUE = "#1a9b6c", "#d9534f", "#f28c28", "#3c5bd4"
GRAPH_CFG = {"displayModeBar": False}

RANGES = {"1M": 30, "6M": 182, "1Y": 365, "5Y": 1825, "All": None}


# ------------------------------------------------------------------ data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"\nCould not find the data file:\n  {path}\n"
                 "Create a 'data' folder next to app.py and put "
                 "All_Stocks_Data_Cleaned.csv inside it.\n")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"\nThe CSV is missing these columns: {missing}\n"
                 f"Columns found: {list(df.columns)}\n")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Closing_Price"] = pd.to_numeric(df["Closing_Price"], errors="coerce")
    df = (df.dropna(subset=["Symbol", "Date", "Closing_Price"])
            .sort_values(["Symbol", "Date"]).reset_index(drop=True))

    # Use the supplied daily return, or work it out from prices.
    if "Daily_Return" in df.columns:
        df["Daily_Return"] = pd.to_numeric(df["Daily_Return"], errors="coerce")
    else:
        df["Daily_Return"] = df.groupby("Symbol")["Closing_Price"].pct_change()

    price = df.groupby("Symbol")["Closing_Price"]
    df["Year"] = df["Date"].dt.year
    df["MA_7D"] = price.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["MA_30D"] = price.transform(lambda s: s.rolling(30, min_periods=1).mean())
    df["Drawdown"] = (df["Closing_Price"] / price.cummax() - 1) * 100
    return df


def build_stock_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stock: return, risk and consistency numbers."""
    stats = (df.dropna(subset=["Daily_Return"]).groupby("Symbol")["Daily_Return"]
               .agg(Avg_Return="mean", Volatility="std", Trading_Days="count",
                    Positive_Days=lambda x: int((x > 0).sum()),
                    Negative_Days=lambda x: int((x < 0).sum())))
    stats["Avg_Return_Pct"] = stats["Avg_Return"] * 100
    stats["Volatility_Pct"] = stats["Volatility"] * 100
    stats["Max_Drawdown"] = df.groupby("Symbol")["Drawdown"].min()
    first = df.groupby("Symbol")["Closing_Price"].first()
    last = df.groupby("Symbol")["Closing_Price"].last()
    stats["Total_Return"] = (last / first - 1) * 100
    return stats.dropna(subset=["Volatility"]).reset_index()


def build_yearly_returns(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (df.groupby(["Symbol", "Year"])["Closing_Price"]
                .agg(First="first", Last="last").reset_index())
    yearly["Yearly_Return"] = (yearly["Last"] / yearly["First"] - 1) * 100
    return yearly


# ------------------------------------------------------------------ machine learning
ML_FEATURES = ["Lag_Return_1D", "Lag_Return_5D", "Price_Momentum_7D",
               "Price_Momentum_30D", "Volatility_7D", "Volatility_30D"]


def train_model(df: pd.DataFrame) -> dict | None:
    """Random Forest predicting next-day direction, with a time-based split."""
    d = df[["Symbol", "Date", "Closing_Price", "Daily_Return"]].copy()
    ret = d.groupby("Symbol")["Daily_Return"]
    price = d.groupby("Symbol")["Closing_Price"]

    d["Lag_Return_1D"] = ret.shift(1)
    d["Lag_Return_5D"] = ret.shift(5)
    d["Price_Momentum_7D"] = d["Closing_Price"] / price.shift(7) - 1
    d["Price_Momentum_30D"] = d["Closing_Price"] / price.shift(30) - 1
    d["Volatility_7D"] = ret.transform(lambda s: s.rolling(7).std())
    d["Volatility_30D"] = ret.transform(lambda s: s.rolling(30).std())
    d["Next_Day_Return"] = ret.shift(-1)

    d = d.dropna(subset=ML_FEATURES + ["Next_Day_Return"])
    d = d.sort_values("Date").tail(MAX_ML_ROWS)
    if len(d) < 200:
        return None

    d["Target"] = (d["Next_Day_Return"] > 0).astype(int)
    split = int(len(d) * 0.8)
    X_tr, X_te = d[ML_FEATURES].iloc[:split], d[ML_FEATURES].iloc[split:]
    y_tr, y_te = d["Target"].iloc[:split], d["Target"].iloc[split:]
    if y_tr.nunique() < 2:
        return None

    model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42,
                                   n_jobs=-1, class_weight="balanced").fit(X_tr, y_tr)
    pred = model.predict(X_te)
    share_up = y_te.mean()
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "baseline": max(share_up, 1 - share_up),
        "matrix": confusion_matrix(y_te, pred, labels=[0, 1]),
        "importance": pd.DataFrame({"Feature": ML_FEATURES,
                                    "Importance": model.feature_importances_})
                        .sort_values("Importance"),
    }


# ------------------------------------------------------------------ prepare everything once
print("Loading data and training the model, one moment...")
df = load_data(DATA_PATH)
stats = build_stock_stats(df)
yearly = build_yearly_returns(df)
ml = train_model(df)

years = sorted(int(y) for y in df["Year"].unique())
symbols = sorted(df["Symbol"].astype(str).unique())
stats_by_symbol = stats.set_index("Symbol")


# ------------------------------------------------------------------ figure helpers
def style(fig: go.Figure, title: str | None = None, unified: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=15, color=INK)),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, system-ui, sans-serif", color=INK),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig)


def hbar(data, x, y, color, title, x_label) -> go.Figure:
    fig = px.bar(data, x=x, y=y, orientation="h", labels={x: x_label, y: "Stock"})
    fig.update_traces(marker_color=color)
    fig.update_yaxes(title=None)
    return style(fig, title)


# ------------------------------------------------------------------ static figures
def risk_return_figure() -> go.Figure:
    fig = px.scatter(stats, x="Volatility_Pct", y="Avg_Return_Pct", hover_name="Symbol",
                     labels={"Volatility_Pct": "Volatility (% daily)",
                             "Avg_Return_Pct": "Average return (% daily)"})
    fig.update_traces(marker=dict(size=9, color=BLUE, opacity=0.7,
                                  line=dict(width=1, color="white")))
    return style(fig, "Risk vs return, every stock")


def volatility_figure() -> go.Figure:
    top = stats.nlargest(10, "Volatility_Pct").sort_values("Volatility_Pct")
    return hbar(top, "Volatility_Pct", "Symbol", SAFFRON,
                "10 most volatile stocks", "Volatility (% daily)")


def drawdown_figure() -> go.Figure:
    worst = stats.nsmallest(10, "Max_Drawdown").sort_values("Max_Drawdown", ascending=False)
    return hbar(worst, "Max_Drawdown", "Symbol", LOSS,
                "10 deepest drawdowns", "Maximum drawdown (%)")


def importance_figure() -> go.Figure:
    return hbar(ml["importance"], "Importance", "Feature", BLUE,
                "What the model relies on", "Importance")


def confusion_figure() -> go.Figure:
    labels = ["Down", "Up"]
    fig = px.imshow(ml["matrix"], text_auto=True, x=labels, y=labels,
                    color_continuous_scale=["#eef1f8", BLUE],
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"})
    fig.update_coloraxes(showscale=False)
    return style(fig, "Predicted vs actual direction (test set)")


# ------------------------------------------------------------------ layout components
def kpi(label, value, note="", tone=""):
    return html.Div(className=f"kpi-card {tone}".strip(), children=[
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(note, className="kpi-note"),
    ])


def section(title, description):
    return html.Div(className="section-head", children=[
        html.H2(title, className="section-title"),
        html.P(description, className="section-description"),
    ])


def card(title, child, description=None, class_name=""):
    kids = [html.Div(title, className="card-title")]
    if description:
        kids.append(html.P(description, className="card-description"))
    kids.append(child)
    return html.Div(className=f"dashboard-card {class_name}".strip(), children=kids)


def graph(graph_id=None, figure=None, height=None):
    props = {"config": GRAPH_CFG, "className": "chart"}
    if graph_id:
        props["id"] = graph_id
    if figure is not None:
        props["figure"] = figure
    if height:
        props["style"] = {"height": height}
    return dcc.Graph(**props)


def dropdown(dropdown_id, options, value, searchable=False):
    return dcc.Dropdown(id=dropdown_id, options=options, value=value,
                        searchable=searchable, clearable=False, className="dropdown")


def ml_section():
    if ml is None:
        return [section("Machine learning", "Not enough data to train the model."),
                card("Model", html.P("The dataset needs at least a few hundred rows with "
                                     "valid returns to train the classifier."))]
    return [
        section("Machine learning",
                "A Random Forest guesses whether a stock rises the next day. "
                "It is tested on the most recent 20% of the data."),
        html.Div(className="kpi-grid", children=[
            kpi("Accuracy", f"{ml['accuracy']:.1%}",
                f"Always guessing the common outcome scores {ml['baseline']:.1%}", "accent"),
            kpi("Precision", f"{ml['precision']:.1%}", "Of predicted up days"),
            kpi("Recall", f"{ml['recall']:.1%}", "Of actual up days"),
            kpi("F1 score", f"{ml['f1']:.1%}", "Precision and recall combined"),
        ]),
        html.Div(className="two-column", children=[
            card("Feature importance", graph(figure=importance_figure()),
                 "Which inputs the model relies on most."),
            card("Confusion matrix", graph(figure=confusion_figure()),
                 "How often each prediction was right or wrong."),
        ]),
        card("How to read this", html.Ul(className="notes", children=[
            html.Li("Inputs: 1-day and 5-day lagged returns, 7-day and 30-day momentum, "
                    "and 7-day and 30-day volatility."),
            html.Li("Daily direction is close to a coin flip. Compare accuracy with the "
                    "baseline before drawing conclusions."),
            html.Li("This is a learning project, not investment advice."),
        ])),
    ]


# ------------------------------------------------------------------ app + layout
app = Dash(__name__, title="Indian Stock Market Analytics")
server = app.server

app.layout = html.Div(className="page", children=[
    html.Div(className="flag-stripe"),
    html.Div(className="dashboard-container", children=[

        html.Header(className="header", children=[
            html.Div([
                html.H1("Indian Stock Market Analytics", className="brand-title"),
                html.P("Prices, returns, risk and a next-day direction model "
                       "across the full dataset.", className="brand-subtitle"),
            ]),
            html.Div(f"{df['Date'].min():%b %Y} to {df['Date'].max():%b %Y}",
                     className="header-badge"),
        ]),

        section("Market overview", "The dataset at a glance."),
        html.Div(className="kpi-grid", children=[
            kpi("Stocks", f"{df['Symbol'].nunique():,}", "Unique symbols", "accent"),
            kpi("Trading records", f"{len(df):,}", "Daily observations"),
            kpi("Average close", f"₹{df['Closing_Price'].mean():,.2f}", "Across all stocks"),
            kpi("Median close", f"₹{df['Closing_Price'].median():,.2f}", "Less skewed by outliers"),
            kpi("Average volatility", f"{stats['Volatility_Pct'].mean():.2f}%",
                "Std. dev. of daily return"),
        ]),

        section("Stock explorer", "Pick a stock and a time range."),
        html.Div(className="controls", children=[
            card("Stock", dropdown("stock-selector", symbols, symbols[0], searchable=True),
                 class_name="control"),
            card("Time range", dcc.RadioItems(
                id="range-selector",
                options=[{"label": k, "value": k} for k in RANGES],
                value="All", className="pills", inline=True),
                class_name="control"),
        ]),
        html.Div(id="stock-kpis", className="kpi-grid four"),
        html.Div(className="two-column", children=[
            card("Closing price", graph("stock-price-chart"),
                 "Close with 7-day and 30-day moving averages."),
            card("Daily return", graph("stock-return-chart"),
                 "Day-to-day percentage change."),
        ]),

        section("Performance", "Compare stocks within one calendar year."),
        html.Div(className="controls", children=[
            card("Year", dropdown("year-selector",
                                  [{"label": str(y), "value": y} for y in years], years[-1]),
                 class_name="control"),
        ]),
        html.Div(className="two-column", children=[
            card("Top 10 performers", graph("top-chart")),
            card("Bottom 10 performers", graph("bottom-chart")),
        ]),
        html.Div(className="two-column", children=[
            card("Return distribution", graph("return-boxplot"),
                 "Spread of daily returns for the selected year."),
            card("Up vs down days", graph("trading-day-pie"),
                 "For the stock chosen in the explorer."),
        ]),

        section("Risk", "Volatility, drawdowns and the risk and return trade-off."),
        html.Div(className="two-column", children=[
            card("Risk vs return", graph(figure=risk_return_figure())),
            card("Highest volatility", graph(figure=volatility_figure())),
        ]),
        card("Maximum drawdown", graph(figure=drawdown_figure()),
             "Largest fall from a previous peak."),

        section("Correlation", "How the year's best performers move together."),
        card("Return correlation", graph("correlation-heatmap", height="560px"),
             "Top 15 stocks of the selected year, by yearly return."),

        *ml_section(),

        html.Footer("Python, Pandas, Plotly, Dash and Scikit-learn", className="footer"),
    ]),
])


# ------------------------------------------------------------------ callbacks
@app.callback(
    Output("stock-kpis", "children"),
    Output("stock-price-chart", "figure"),
    Output("stock-return-chart", "figure"),
    Output("trading-day-pie", "figure"),
    Input("stock-selector", "value"),
    Input("range-selector", "value"),
)
def update_stock_explorer(symbol, time_range):
    data = df[df["Symbol"] == symbol]
    if data.empty or symbol not in stats_by_symbol.index:
        blank = empty_figure("No data for this stock")
        return [], blank, blank, blank

    s = stats_by_symbol.loc[symbol]
    total = s["Total_Return"]

    kpis = [
        kpi("Latest close", f"₹{data['Closing_Price'].iloc[-1]:,.2f}",
            f"{data['Date'].iloc[-1]:%d %b %Y}", "accent"),
        kpi("Total return", f"{total:+.1f}%", "First to last close",
            "gain" if total >= 0 else "loss"),
        kpi("Volatility", f"{s['Volatility_Pct']:.2f}%", "Daily std. dev."),
        kpi("Max drawdown", f"{s['Max_Drawdown']:.1f}%", "Worst fall from a peak", "loss"),
    ]

    # Apply the time range to the charts only.
    days = RANGES.get(time_range)
    view = data if days is None else data[data["Date"] >= data["Date"].max() - pd.Timedelta(days=days)]

    price = go.Figure()
    for col, name, color, width in (("Closing_Price", "Close", INK, 2),
                                    ("MA_7D", "7-day MA", SAFFRON, 1.3),
                                    ("MA_30D", "30-day MA", BLUE, 1.3)):
        price.add_trace(go.Scatter(x=view["Date"], y=view[col], name=name, mode="lines",
                                   line=dict(color=color, width=width),
                                   hovertemplate=f"{name}: ₹%{{y:,.2f}}<extra></extra>"))
    style(price, f"{symbol} price trend", unified=True).update_yaxes(title="Price (₹)")

    returns = px.line(view, x="Date", y="Daily_Return")
    returns.update_traces(line=dict(color=BLUE, width=1),
                          hovertemplate="%{y:.2%}<extra></extra>")
    style(returns, f"{symbol} daily returns", unified=True).update_yaxes(
        tickformat=".1%", title=None)

    pie = px.pie(names=["Up days", "Down days"],
                 values=[s["Positive_Days"], s["Negative_Days"]], hole=0.62,
                 color_discrete_sequence=[GAIN, LOSS])
    pie.update_traces(textinfo="label+percent", marker=dict(line=dict(color="white", width=2)))
    style(pie, f"{symbol} up vs down days").update_layout(showlegend=False)

    return kpis, price, returns, pie


@app.callback(
    Output("top-chart", "figure"),
    Output("bottom-chart", "figure"),
    Output("return-boxplot", "figure"),
    Output("correlation-heatmap", "figure"),
    Input("year-selector", "value"),
)
def update_yearly_views(year):
    y = yearly[yearly["Year"] == year]
    if y.empty:
        blank = empty_figure(f"No data for {year}")
        return blank, blank, blank, blank

    top = y.nlargest(10, "Yearly_Return").sort_values("Yearly_Return")
    bottom = y.nsmallest(10, "Yearly_Return").sort_values("Yearly_Return")
    top_fig = hbar(top, "Yearly_Return", "Symbol", GAIN, f"Top 10 in {year}", "Return (%)")
    bottom_fig = hbar(bottom, "Yearly_Return", "Symbol", LOSS, f"Bottom 10 in {year}", "Return (%)")

    year_data = df[df["Year"] == year].dropna(subset=["Daily_Return"])
    if year_data.empty:
        box = empty_figure(f"No return data for {year}")
    else:
        if len(year_data) > MAX_BOX_POINTS:
            year_data = year_data.sample(MAX_BOX_POINTS, random_state=42)
        box = px.box(year_data, y="Daily_Return", points=False,
                     labels={"Daily_Return": "Daily return"})
        box.update_traces(marker_color=BLUE, line_color=BLUE)
        style(box, f"Daily return distribution, {year}").update_yaxes(tickformat=".1%")

    top_symbols = y.nlargest(15, "Yearly_Return")["Symbol"]
    pivot = (df[(df["Year"] == year) & df["Symbol"].isin(top_symbols)]
             .pivot_table(index="Date", columns="Symbol", values="Daily_Return"))
    if pivot.shape[1] < 2:
        heat = empty_figure(f"Not enough stocks to correlate in {year}")
    else:
        heat = px.imshow(pivot.corr(), text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r", labels={"color": "Correlation"})
        style(heat, f"Return correlation of {year}'s top 15")

    return top_fig, bottom_fig, box, heat


if __name__ == "__main__":
    app.run(debug=True, port=8050)