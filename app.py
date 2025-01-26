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
    z_coeff = np.array([9.67721386e-03, 1.10570431e-15, 1.54790937e-03, 2.69149822e-02,
                       1.97910476e-16, 1.40157558e-02, 1.06053578e-16, 0.00000000e+00,
                       5.37495912e-03, 3.11841208e-16, 1.74914390e-16, 1.20579363e-03,
                       6.15432724e-03, 5.03502350e-02, 1.24624488e-02, 0.00000000e+00,
                       1.04115473e-02, 1.78613118e-02, 8.88006863e-03, 9.67873296e-03,
                       1.33681816e-02, 9.99901461e-03, 8.32098437e-03, 9.51542969e-03,
                       1.46759356e-02, 1.04305854e-02, 1.70846164e-02, 2.79696132e-02,
                       3.10586124e-02, 1.32409730e-03, 1.37491307e-02, 3.25344227e-02,
                       3.13789866e-16, 2.52826291e-02, 1.70126948e-02, 6.24676005e-03,
                       0.00000000e+00, 2.49269421e-16, 0.00000000e+00, 2.11138637e-02,
                       2.30097682e-16, 2.00123900e-16, 0.00000000e+00, 9.77197309e-03,
                       1.58019014e-02, 5.62860255e-03, 2.37591930e-02, 0.00000000e+00,
                       1.45154300e-02, 2.76151715e-16, 2.69699963e-02, 1.21119691e-02,
                       1.54293837e-02, 2.16686727e-16, 4.52055179e-02, 7.14652867e-03,
                       1.95317032e-02, 0.00000000e+00, 3.01881915e-15, 0.00000000e+00,
                       1.10848345e-16, 0.00000000e+00, 4.43420107e-02, 3.83539194e-03,
                       1.22623176e-16, 1.75432152e-02, 9.19043948e-03, 0.00000000e+00,
                       2.05983798e-02, 1.35443121e-02, 4.85891007e-02, 2.74801910e-02,
                       1.66884748e-02, 9.92810073e-04, 1.38417667e-15, 1.02470431e-02,
                       3.09788075e-02, 1.62897640e-02, 1.79810309e-02, 2.73274608e-02,
                       0.00000000e+00, 2.02622147e-02, 1.55302494e-16, 0.00000000e+00,
                       5.58036496e-03, 2.65381048e-02, 8.30543928e-16, 1.17231893e-02,
                       5.43430470e-04, 1.56310353e-03, 1.31864117e-02, 1.55759774e-02,
                       4.68443469e-03, 8.40312714e-03, 7.65669577e-03, 1.00958892e-02,
                       1.08574034e-02, 1.94540168e-02, 4.11427659e-16, 1.51606923e-02])
    res = np.sum(heston_call(x, M = n)*z_coeff)
    return res

def fast_heston(maturity, moneyness, strike, ticker):
    if ticker == 'AAPL':
        params = [0.0843, 0.046, 0.0438, 6.3848, -0.5597, 0.5308]
    elif ticker == 'AMD':
        params = [0.2515, 0.046, 0.1767, 6.7325, -0.1474, 0.8557]
    elif ticker == 'AMZN':
        params = [0.1142, 0.046, 0.0797, 6.2283, -0.3936, 0.679]
    elif ticker == 'GOOGL':
        params = [0.1148, 0.046, 0.071, 7.2975, -0.3939, 0.5692]
    elif ticker == 'MSFT':
        params = [0.0446, 0.046, 0.0632, 0.9824, -0.0415, 0.3024]
    else:
        return 'Company information not available!'
    x = np.array([moneyness, maturity, strike] + params)
    return heston_surrogate(x)


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
        result = np.round(fast_heston(time_to_maturity, moneyness, strike, company), 2)
        
        # Return the result to the user
        return render_template("result.html", result=result)
    else:
        # Handle GET request (show the form to the user)
        return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
