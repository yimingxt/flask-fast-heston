import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


app = Flask(__name__)

size_max = (50000, 252)
np.random.seed(321)
z1_seed = np.random.normal(size=size_max)
np.random.seed(42)
z2_seed = np.random.normal(size=size_max)

def heston_call(x, dt = 1/252, M = 1, crn = True, div = 0.0):
    moneyness, T, K, V0, r, theta, kappa, rho, sigma = x
    S0 = K/(1-moneyness)
    sqrt_dt = np.sqrt(dt)
    N = int(252*T)
    
    # initialize arrays to store simulation results
    S = np.zeros((M, N+1))
    V = np.zeros((M, N+1))
    S[:,0] = S0
    V[:,0] = V0

    # crn is used to generate random basis functions
    if crn == True:
        z1 = z1_seed[:M,:N]
        z1_1 = z2_seed[:M,:N]
        z2 = rho * z1 + np.sqrt(1 - rho**2) * z1_1
    else:
        z1 = np.random.normal(size=(M, N))
        z1_1 = np.random.normal(size=(M, N))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * z1_1

    for t in range(1, N+1):
        S[:,t] = S[:,t-1] * np.exp((r - div- 0.5 * V[:,t-1]) * dt + np.sqrt(V[:,t-1]) * sqrt_dt * z1[:,t-1])
        V[:,t] = np.maximum(V[:,t-1] + kappa * (theta - V[:,t-1]) * dt + sigma * np.sqrt(V[:,t-1]) * sqrt_dt * z2[:,t-1], 0)
    
    return (np.maximum(S[:,-1]-K, 0))*np.exp(-r*T)

def heston_surrogate(x, n = 100):
    z_coef = np.array([1.24234437e-02, 2.50074219e-03, 1.99543160e-05, 1.39012038e-02,
                       1.74953633e-03, 1.55454810e-02, 8.49722989e-03, 1.72073735e-05,
                       1.79158347e-05, 1.70682275e-05, 1.38366386e-02, 1.12745600e-02,
                       1.01252501e-03, 1.55727361e-02, 1.83334069e-05, 1.76621553e-05,
                       3.15874548e-03, 4.10585917e-03, 2.19644487e-02, 1.51170724e-02,
                       1.83566249e-02, 1.69608025e-05, 1.36526135e-02, 6.45631579e-03,
                       6.27139313e-03, 1.66996419e-02, 1.36803545e-02, 3.08605260e-02,
                       9.40810401e-03, 1.85626264e-05, 1.69728125e-05, 1.61353533e-02,
                       2.22511542e-03, 7.94663209e-03, 1.17124378e-02, 1.07862046e-02,
                       2.72123661e-03, 1.06750844e-02, 8.18326435e-04, 1.05648696e-02,
                       1.87455624e-02, 1.70703530e-05, 1.70597560e-02, 2.81090772e-03,
                       2.58437423e-02, 1.75963864e-02, 1.01068143e-02, 1.52472436e-02,
                       1.76659631e-02, 5.43261196e-03, 1.00329752e-02, 1.70028170e-05,
                       1.08894873e-02, 1.75359555e-05, 1.52092180e-02, 9.73404692e-03,
                       1.35048884e-02, 1.72029529e-05, 4.74339316e-04, 1.70763304e-05,
                       2.08739667e-03, 1.41876608e-02, 3.72535444e-02, 2.00594405e-03,
                       9.43701264e-03, 9.45299668e-03, 1.13144092e-02, 1.73031626e-05,
                       1.24260117e-02, 1.87990997e-02, 6.04808179e-03, 2.11543095e-02,
                       1.16893795e-02, 1.03529725e-02, 1.80323538e-05, 5.45470471e-03,
                       1.87956291e-02, 1.12581923e-02, 2.90831860e-02, 2.51577801e-02,
                       9.19951880e-03, 1.08764643e-02, 2.03647139e-02, 6.31057214e-03,
                       1.37077829e-02, 1.33471220e-02, 1.10780261e-03, 1.76466756e-02,
                       4.88789831e-03, 2.21396251e-03, 3.07516812e-03, 5.65978114e-03,
                       1.34683090e-02, 1.64096874e-02, 1.95897809e-03, 1.27002892e-03,
                       2.48877613e-02, 2.70093414e-02, 2.12971989e-02, 1.41903350e-02])
    res = np.sum(heston_call(x, M = n)*z_coef)
    return res

def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-3):
    if ticker == 'AAPL':
        params = [0.0874, 0.0455, 0.0441, 7.5102, -0.7597, 0.4053]
    elif ticker == 'AMZN':
        params = [0.1957, 0.0455, 0.0793, 18.3979, -0.95, 0.726]
    elif ticker == 'GOOGL':
        params = [0.2286, 0.0455, 0.0727, 24.8393, -0.95 , 0.6639]
    elif ticker == 'MSFT':
        params = [0.0791, 0.0455, 0.0458, 13.8177, -0.95, 0.3772]
    elif ticker == 'NVDA':
        params = [0.2437, 0.0455, 0.2548, 2, -0.95, 0.2583]
    else:
        return 'Company information not available!'
    spot = strike/(1-moneyness)
    delta_moneyness = delta/spot
    x = np.array([moneyness, maturity, strike] + params)
    x_plus = np.array([moneyness + delta_moneyness, maturity, strike] + params)
    x_minus = np.array([moneyness - delta_moneyness, maturity, strike] + params)
    return heston_surrogate(x), min((heston_surrogate(x_plus) - heston_surrogate(x_minus))/(2*delta),1.0)


def count_us_trading_days(start_date, end_date):
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    return len(trading_days) - 1
    

# Create an API endpoint
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract form data from POST request
        company = request.form.get("Company")
        today = request.form.get("Date of Today")
        maturity = request.form.get("Maturity")
        hour = float(request.form.get("Market Hour"))
        moneyness = float(request.form.get("Moneyness"))/100
        strike = float(request.form.get("Strike"))
        
        # Call your function with the extracted data
        today_date = datetime.strptime(today, "%Y-%m-%d")
        maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
        time_to_maturity = max((count_us_trading_days(today_date, maturity_date)-(hour-9.5)/6.5)/252, 0)
        P, D = fast_heston(time_to_maturity, moneyness, strike, company)
        
        # Return the result to the user
        return render_template("result.html", price=np.round(P,2), delta=np.round(D,5))
    else:
        # Handle GET request (show the form to the user)
        return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
