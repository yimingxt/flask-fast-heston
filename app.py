import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


app = Flask(__name__)

size_max = (100000, 252)
np.random.seed(321)
z1_seed = np.random.normal(size=size_max)
np.random.seed(365)
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
    z_coef = array([0.00000000e+00, 1.24987304e-13, 5.72021408e-04, 2.60810320e-02,
                   1.12266673e-02, 1.04868469e-02, 5.67180016e-03, 1.22623096e-13,
                   2.32785697e-13, 1.01549297e-13, 1.27238336e-13, 0.00000000e+00,
                   1.47228040e-02, 3.39589269e-02, 7.07155790e-03, 0.00000000e+00,
                   2.92806937e-02, 0.00000000e+00, 0.00000000e+00, 2.16720749e-02,
                   4.43233511e-03, 5.04355397e-13, 3.04689639e-03, 4.92289687e-13,
                   4.13288428e-03, 5.03871222e-13, 3.11171554e-02, 1.70996582e-02,
                   5.06285953e-13, 3.70291107e-15, 5.06090912e-13, 3.00921419e-02,
                   1.39737344e-13, 1.33555482e-02, 4.48404607e-03, 4.59222667e-13,
                   3.83937565e-13, 1.30673957e-02, 0.00000000e+00, 1.99758601e-02,
                   4.50368713e-04, 1.20445952e-13, 1.34619456e-13, 4.99377706e-13,
                   2.50526255e-02, 4.77940377e-03, 1.53271681e-02, 1.34884940e-13,
                   8.83305811e-03, 2.63243349e-02, 1.88150480e-02, 5.23091826e-13,
                   2.73788762e-02, 1.05144936e-13, 3.15746084e-03, 1.06957912e-02,
                   2.10428688e-02, 1.18983368e-13, 3.27692131e-03, 1.51856658e-13,
                   1.30206177e-13, 3.26466480e-02, 6.20439966e-02, 2.04931089e-02,
                   2.91979687e-03, 1.67960938e-02, 9.64135539e-03, 7.61607426e-03,
                   6.25062393e-03, 1.96668243e-02, 4.10854792e-02, 2.68555729e-02,
                   1.22937303e-02, 1.45789036e-13, 0.00000000e+00, 9.55283609e-03,
                   4.71848836e-02, 1.15261781e-02, 1.60278902e-02, 1.52357189e-02,
                   1.36495098e-13, 6.51665745e-03, 1.25796251e-13, 1.73791424e-13,
                   1.54200384e-02, 2.44379071e-02, 9.95963308e-14, 2.36573964e-02,
                   3.98692079e-13, 1.79850238e-13, 0.00000000e+00, 2.94107579e-02,
                   8.50476894e-03, 5.31827543e-03, 1.27707673e-13, 9.35386761e-03,
                   5.03294503e-13, 2.79718567e-02, 8.20467865e-03, 1.66868600e-02])
    res = np.sum(heston_call(x, M = n)*z_coeff)
    return res

def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-4):
    if ticker == 'AAPL':
        params = [0.0877, 0.043, 0.01, 3.8151, 0.695, 0.0641]
    elif ticker == 'AMD':
        params = [0.2585, 0.043, 0.1761, 7.7102, 0.2663, 0.6555]
    elif ticker == 'AMZN':
        params = [0.1283, 0.043, 0.0777, 7.2155, -0.3203, 0.9691]
    elif ticker == 'GOOGL':
        params = [0.1245, 0.043, 0.0686, 7.0206, -0.1578, 0.796]
    elif ticker == 'META':
        params = []
    elif ticker == 'MSFT':
        params = [0.0518, 0.043, 0.0455, 7.2704, -0.1999, 0.6112]
    elif ticker == 'NVDA':
        params = [0.2214, 0.043, 0.1958, 1.1037, 0.9, 0.212]
    elif ticker == 'TSLA':
        params = []
    else:
        return 'Company information not available!'
    spot = strike/(1-moneyness)
    delta_moneyness = delta/spot
    x = np.array([moneyness, maturity, strike] + params)
    x_plus = np.array([moneyness + delta_moneyness, maturity, strike] + params)
    x_minus = np.array([moneyness - delta_moneyness, maturity, strike] + params)
    return heston_surrogate(x), (heston_surrogate(x_plus) - heston_surrogate(x_minus))/(2*delta)


def count_us_trading_days(start_date, end_date):
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    return len(trading_days)
    

# Create an API endpoint
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract form data from POST request
        company = request.form.get("Company")
        today = request.form.get("Date of Today")
        maturity = request.form.get("Maturity")
        moneyness = float(request.form.get("Moneyness"))/100
        strike = float(request.form.get("Strike"))
        
        # Call your function with the extracted data
        today_date = datetime.strptime(today, "%Y-%m-%d")
        maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
        time_to_maturity = count_us_trading_days(today_date, maturity_date)/252
        P, D = fast_heston(time_to_maturity, moneyness, strike, company)
        
        # Return the result to the user
        return render_template("result.html", price=np.round(P,2), delta=np.round(D,5))
    else:
        # Handle GET request (show the form to the user)
        return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
