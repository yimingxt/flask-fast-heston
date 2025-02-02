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

def heston_surrogate(x, n = 150):
    z_coef = np.array([0.00678803, 0.0061588 , 0.00094408, 0.00859097, 0.00560614,
                       0.00088986, 0.00318883, 0.00801287, 0.01031345, 0.0018972 ,
                       0.00059306, 0.00656632, 0.0057684 , 0.00332978, 0.00854257,
                       0.01526496, 0.0090497 , 0.01610391, 0.00697003, 0.00643827,
                       0.00065206, 0.00535093, 0.00577953, 0.00929864, 0.00018795,
                       0.01185445, 0.00888193, 0.0177295 , 0.0072948 , 0.00438666,
                       0.00018433, 0.02159607, 0.00044821, 0.00163027, 0.00953149,
                       0.00789663, 0.00583585, 0.00256749, 0.0001876 , 0.00937318,
                       0.00987156, 0.00122417, 0.00070771, 0.00616223, 0.00776028,
                       0.00598889, 0.00866551, 0.00076019, 0.01562233, 0.00986792,
                       0.00907794, 0.00043717, 0.00422205, 0.00027478, 0.00476272,
                       0.00628006, 0.01772647, 0.00420815, 0.00020205, 0.00306931,
                       0.00026306, 0.00232469, 0.01969909, 0.00857517, 0.00157419,
                       0.00032301, 0.00923868, 0.00018606, 0.00529468, 0.01268841,
                       0.00623911, 0.0079664 , 0.00734915, 0.00617052, 0.00110318,
                       0.0001906 , 0.01685842, 0.01950074, 0.01696988, 0.00813128,
                       0.00556284, 0.00587532, 0.0058069 , 0.01055807, 0.01224175,
                       0.0178982 , 0.00200476, 0.00456134, 0.00211072, 0.00018708,
                       0.01710286, 0.0004731 , 0.00029726, 0.01202912, 0.01353653,
                       0.00183874, 0.00155013, 0.01348808, 0.01057   , 0.00599455,
                       0.00018693, 0.00449763, 0.01185971, 0.00074736, 0.00023464,
                       0.00694137, 0.00821582, 0.00019229, 0.00611728, 0.0110698 ,
                       0.00033777, 0.00790481, 0.00131177, 0.01246271, 0.00019503,
                       0.0058742 , 0.00575706, 0.01168535, 0.00956895, 0.01027392,
                       0.00414993, 0.00909277, 0.00454582, 0.00656449, 0.00306922,
                       0.00841165, 0.00394918, 0.00665493, 0.00087516, 0.01786162,
                       0.00672411, 0.00611825, 0.00018496, 0.00581738, 0.00088533,
                       0.00272049, 0.00655724, 0.01124385, 0.00089261, 0.01259821,
                       0.01025911, 0.00919923, 0.01528662, 0.01023138, 0.01216286,
                       0.01029534, 0.00344483, 0.00787894, 0.00149226, 0.01172604])
    res = np.sum(heston_call(x, M = n)*z_coef)
    return res

def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-4):
    if ticker == 'AAPL':
        params = [0.0508, 0.045, 0.0692, 4.4939, -0.3942, 0.6793]
    elif ticker == 'AMZN':
        params = [0.1167, 0.0455, 0.0899, 2.6865, -0.3577, 0.5343]
    elif ticker == 'GOOGL':
        params = [0.1944, 0.0455, 0.079, 40.9352, -0.99 , 0.6956]
    elif ticker == 'MSFT':
        params = [0.05, 0.0455, 0.0562, 4.2598, -0.1654, 0.6916]
    elif ticker == 'NVDA':
        params = [0.967, 0.0455, 0.234, 24.0423, -0.99, 1.0]
    else:
        return 'Company information not available!'
    spot = strike/(1-moneyness)
    x = np.array([moneyness, maturity, strike] + params)
    x_plus = np.array([moneyness + delta, maturity, strike] + params)
    x_minus = np.array([moneyness - delta, maturity, strike] + params)
    return heston_surrogate(x), (heston_surrogate(x_plus) - heston_surrogate(x_minus))/(2*delta)*strike/(spot**2)


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
