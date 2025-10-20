from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)
app.secret_key = "inventory_secret_key"

# =====================================================
# File Paths
# =====================================================
DATA_PATH = "retail_store_inventory.csv"
USER_PATH = "users.csv"

# Create users.csv if not exists
if not os.path.exists(USER_PATH):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_PATH, index=False)

# =====================================================
# Load and Prepare Dataset
# =====================================================
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date")

# Detect sales column
possible_sales_cols = [
    c for c in df.columns
    if any(keyword in c.lower() for keyword in ["sale", "quantity", "unit sold", "units sold"])
]
if not possible_sales_cols:
    raise ValueError(f"No usable sales column found. Columns: {df.columns.tolist()}")

df.rename(columns={possible_sales_cols[0]: "Sales"}, inplace=True)

# =====================================================
# Helper Functions
# =====================================================
def generate_forecast_chart(history_df, forecast_df):
    plt.figure(figsize=(10, 5))
    plt.style.use("dark_background")

    plt.plot(history_df["Date"], history_df["Sales"], label="Historical Sales", color="#00BFFF", linewidth=2)
    plt.plot(forecast_df["Date"], forecast_df["Predicted_Sales"], label="Forecasted Sales", color="#FF69B4", linestyle="--", linewidth=2)

    plt.title("🔮 Forecasted Sales", fontsize=14, color="white")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.legend(facecolor="#1E1E1E", edgecolor="white")
    plt.grid(alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return chart_base64


def forecast_sales(data, days=30):
    if len(data) < 10:
        forecast_values = np.repeat(data["Sales"].mean(), days)
    else:
        try:
            model = ExponentialSmoothing(
                data["Sales"],
                trend="add",
                seasonal=None,
                initialization_method="estimated"
            ).fit()
            forecast_values = model.forecast(days)
        except Exception as e:
            print("⚠️ Forecast failed:", e)
            forecast_values = np.repeat(data["Sales"].mean(), days)

    start_date = pd.Timestamp.today().normalize()
    future_dates = pd.date_range(start=start_date, periods=days, freq="D")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Sales": forecast_values.round(2)
    })
    return forecast_df

# =====================================================
# AUTH ROUTES
# =====================================================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        users = pd.read_csv(USER_PATH)
        if username in users["username"].values:
            flash("⚠️ Username already exists. Try a different one.")
            return redirect(url_for("signup"))

        users.loc[len(users)] = [username, password]
        users.to_csv(USER_PATH, index=False)
        flash("✅ Signup successful! Please log in.")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        users = pd.read_csv(USER_PATH)
        user = users[(users["username"] == username) & (users["password"] == password)]

        if not user.empty:
            session["user"] = username
            flash(f"👋 Welcome back, {username}!")
            return redirect(url_for("index"))
        else:
            flash("❌ Invalid username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("👋 You have been logged out.")
    return redirect(url_for("login"))


# =====================================================
# MAIN FORECAST ROUTE (Store S005 Fixed)
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    forecast_summary = None
    forecast_table = None
    chart_url = None
    error = None

    if request.method == "POST":
        try:
            product_num = int(request.form.get("product_id"))
            days = int(request.form.get("days", 30))

            store_id = "S005"  # ✅ Fixed store
            product_id = f"P{product_num:04d}"

            data = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].copy()
            if data.empty:
                raise ValueError(f"⚠️ No data found for Store {store_id} and Product {product_id}")

            forecast_df = forecast_sales(data, days)
            forecast_table = forecast_df.to_dict(orient="records")

            total_sales = int(data["Sales"].sum())
            recommended_inventory = int(forecast_df["Predicted_Sales"].sum())
            status = "Sufficient" if recommended_inventory < total_sales else "Low"

            forecast_summary = {
                "store_id": store_id,
                "total_inventory": total_sales,
                "recommended_inventory": recommended_inventory,
                "status": status,
                "forecast_days": days
            }

            chart_url = generate_forecast_chart(data, forecast_df)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        forecast=forecast_summary,
        forecast_table=forecast_table,
        chart_url=chart_url,
        error=error,
        user=session.get("user")
    )

# =====================================================
# CSV Download Route (Store S005 Fixed)
# =====================================================
@app.route("/download_csv", methods=["POST"])
def download_csv():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        product_num = int(request.form.get("product_id"))
        days = int(request.form.get("days", 30))

        store_id = "S005"
        product_id = f"P{product_num:04d}"

        data = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].copy()
        if data.empty:
            return "No data found for that Store/Product", 404

        forecast_df = forecast_sales(data, days)

        csv_buffer = io.StringIO()
        forecast_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        filename = f"forecast_{store_id}_{product_id}.csv"
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return f"Error generating CSV: {e}", 500


# =====================================================
# Run App
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
