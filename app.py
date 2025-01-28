import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


app = Flask(__name__)

size_max = (50000, 252)
np.random.seed(10)
z1_seed = np.random.normal(size=size_max)
np.random.seed(23)
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
    z_coef = np.array([0.01774259, 0.0093835 , 0.01506296, 0.00418113, 0.00747904,
                       0.01324557, 0.01370538, 0.01608237, 0.00875524, 0.01353247,
                       0.01884955, 0.00716181, 0.00467708, 0.00034029, 0.00017855,
                       0.01036511, 0.033842  , 0.01436941, 0.00013521, 0.01035658,
                       0.02484013, 0.0001352 , 0.00629473, 0.01248017, 0.00013916,
                       0.01328922, 0.01834557, 0.01068036, 0.01375051, 0.01334279,
                       0.00284158, 0.00042095, 0.0141786 , 0.00357659, 0.00608243,
                       0.01266867, 0.01241325, 0.00013941, 0.0001391 , 0.00652472,
                       0.00544468, 0.01044808, 0.0155387 , 0.00013495, 0.00645875,
                       0.01994568, 0.01338059, 0.02176663, 0.0168812 , 0.00147789,
                       0.00013658, 0.02040937, 0.01318371, 0.00013597, 0.01093453,
                       0.03736791, 0.00725481, 0.00013766, 0.0001347 , 0.01787668,
                       0.00014371, 0.009234  , 0.02348773, 0.01352525, 0.02695809,
                       0.01101099, 0.00013697, 0.00489508, 0.01312232, 0.01624259,
                       0.0206106 , 0.00964935, 0.02058435, 0.01151233, 0.00508988,
                       0.01817442, 0.00013442, 0.00421665, 0.00362187, 0.00568436,
                       0.00013448, 0.01132879, 0.01409695, 0.01470136, 0.00013426,
                       0.01532582, 0.01427104, 0.00948644, 0.00801685, 0.01014479,
                       0.00014008, 0.00848569, 0.01556867, 0.02641422, 0.00013681,
                       0.01143541, 0.00013665, 0.0001357 , 0.00013906, 0.00697549])
    res = np.sum(heston_call(x, M = n)*z_coef)
    return res

def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-3):
    if ticker == 'AAPL':
        params = [0.1007, 0.044, 0.01, 4.0345, -0.95, 0.144]
    elif ticker == 'AMD':
        params = [0.2585, 0.043, 0.1761, 7.7102, 0.2663, 0.6555]
    elif ticker == 'AMZN':
        params = [0.1283, 0.043, 0.0777, 7.2155, -0.3203, 0.9691]
    elif ticker == 'GOOGL':
        params = [0.1245, 0.043, 0.0686, 7.0206, -0.1578, 0.796]
    elif ticker == 'META':
        params = [0.1839, 0.043, 0.1214, 14.9241, 0.7659, 0.4687]
    elif ticker == 'MSFT':
        params = [0.0518, 0.043, 0.0455, 7.2704, -0.1999, 0.6112]
    elif ticker == 'NVDA':
        params = [0.2214, 0.043, 0.1958, 1.1037, 0.9, 0.212]
    elif ticker == 'TSLA':
        params = [0.3951, 0.043, 0.3171, 10.6073, 0.8928, 1.0395]
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
