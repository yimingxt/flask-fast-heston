import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


app = Flask(__name__)

size_max = (50000, 252)
np.random.seed(1995)
z1_seed = np.random.normal(size=size_max)
np.random.seed(1023)
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

def heston_surrogate(x, n = 150):
    z_coef = np.array([0.00470824, 0.01508166, 0.00010205, 0.01442324, 0.01090443,
                       0.00013603, 0.00010668, 0.01753078, 0.00054518, 0.00010212,
                       0.00010172, 0.00010162, 0.00190762, 0.00488583, 0.00010409,
                       0.00010479, 0.00097565, 0.00038527, 0.00680624, 0.00677312,
                       0.00177999, 0.00843283, 0.00683524, 0.01562553, 0.01646652,
                       0.00010149, 0.01145947, 0.00010146, 0.00804963, 0.00010245,
                       0.00010163, 0.00015643, 0.00033157, 0.00010342, 0.02868731,
                       0.01196265, 0.00010211, 0.0104091 , 0.00024439, 0.00497308,
                       0.00654882, 0.0001043 , 0.00354703, 0.00595327, 0.01848993,
                       0.00850148, 0.01122092, 0.01511375, 0.01524433, 0.01780864,
                       0.01263502, 0.00732282, 0.00107356, 0.01063644, 0.00010162,
                       0.00596675, 0.00010283, 0.00010271, 0.00604979, 0.00042201,
                       0.00010345, 0.00010158, 0.02201811, 0.01056277, 0.00937681,
                       0.00713623, 0.00845416, 0.00010477, 0.00010141, 0.01252065,
                       0.02245554, 0.00010262, 0.01381156, 0.00010596, 0.01485632,
                       0.0205505 , 0.02093414, 0.00166105, 0.00081711, 0.00309026,
                       0.00010423, 0.00545746, 0.00261876, 0.01407074, 0.00245444,
                       0.01306865, 0.020263  , 0.01910271, 0.00010535, 0.0001035 ,
                       0.00010588, 0.00010155, 0.00010181, 0.00010439, 0.01191475,
                       0.0012034 , 0.00463531, 0.00019328, 0.01022683, 0.0007239 ,
                       0.01078488, 0.01223952, 0.00226947, 0.00763425, 0.01803247,
                       0.01277858, 0.02215373, 0.01777122, 0.01238598, 0.00010425,
                       0.00010243, 0.01221351, 0.00037103, 0.0072554 , 0.00577627,
                       0.02328465, 0.00135359, 0.02465319, 0.01121302, 0.00346197,
                       0.01859238, 0.00725932, 0.00972466, 0.00205838, 0.00036258,
                       0.00010199, 0.00290817, 0.00010684, 0.01200076, 0.00010272,
                       0.00265545, 0.00010252, 0.01696641, 0.01583413, 0.01326084,
                       0.0077226 , 0.0010571 , 0.00147751, 0.00010495, 0.02352502,
                       0.00010456, 0.00010504, 0.01284247, 0.00568519, 0.00010707,
                       0.00010319, 0.00010526, 0.00350929, 0.00128872, 0.00010179])
    res = np.sum(heston_call(x, M = n)*z_coef)
    return res

def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-3):
    if ticker == 'AAPL':
        params = [0.1098, 0.0455, 0.0497, 15.4368, -0.8392, 0.342]
    elif ticker == 'AMZN':
        params = [0.1683, 0.0455, 0.1108, 6.2479, -0.553, 0.8523]
    elif ticker == 'GOOGL':
        params = [0.2244, 0.045, 0.0692, 40.931, -0.5849, 0.7488]
    elif ticker == 'MSFT':
        params = [0.0466, 0.045, 0.0504, 2.9866, -0.95, 0.137]
    elif ticker == 'NVDA':
        params = [0.212, 0.0455, 0.2515, 37.2135, -0.95, 0.1158]
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
        hour = request.form.get()
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
