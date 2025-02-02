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

def heston_surrogate(x, n = 200):
    z_coef = np.array([0.0057066 , 0.00478367, 0.00024394, 0.00451472, 0.00030314,
                       0.00326808, 0.00100725, 0.00281425, 0.00183978, 0.00023513,
                       0.00065962, 0.01002534, 0.00631757, 0.00832904, 0.00069742,
                       0.01318742, 0.00399338, 0.00587676, 0.0151281 , 0.00027905,
                       0.00383456, 0.00055534, 0.00368629, 0.00534812, 0.00528123,
                       0.00449712, 0.00629902, 0.01280469, 0.00643876, 0.00383156,
                       0.00023561, 0.01554065, 0.00873032, 0.00023513, 0.00446679,
                       0.00593992, 0.00023525, 0.0103678 , 0.00023556, 0.00797098,
                       0.01196738, 0.00023513, 0.00672519, 0.00194916, 0.01113458,
                       0.00771591, 0.00742643, 0.00471773, 0.00650065, 0.01255569,
                       0.00023513, 0.01143147, 0.00541515, 0.0041285 , 0.00447401,
                       0.00025912, 0.00637424, 0.00139617, 0.00239794, 0.00966525,
                       0.00125548, 0.00714522, 0.01409868, 0.00799388, 0.00886114,
                       0.00023538, 0.00344586, 0.00023513, 0.00396043, 0.00999976,
                       0.00749941, 0.00463964, 0.00672153, 0.0055167 , 0.00196734,
                       0.00023513, 0.00996241, 0.00954354, 0.01117649, 0.00938713,
                       0.0058357 , 0.00420476, 0.00513069, 0.00974373, 0.00347876,
                       0.01630624, 0.00062432, 0.00048914, 0.00023513, 0.00025979,
                       0.01310654, 0.00617475, 0.00039741, 0.00498197, 0.00449541,
                       0.00023997, 0.00196955, 0.00710017, 0.00701623, 0.00226573,
                       0.00492545, 0.00422402, 0.00872037, 0.00092424, 0.00662356,
                       0.00526703, 0.0023688 , 0.00198997, 0.00182888, 0.00190441,
                       0.00215614, 0.00721555, 0.00124913, 0.00867996, 0.00817154,
                       0.00753688, 0.0021688 , 0.007533  , 0.00681112, 0.00549043,
                       0.00391   , 0.00127732, 0.00241675, 0.00662816, 0.0071236 ,
                       0.00684624, 0.0064169 , 0.0110664 , 0.00023775, 0.01099067,
                       0.00514317, 0.00335353, 0.00076791, 0.00432666, 0.01058152,
                       0.00060751, 0.00515022, 0.00567172, 0.00023573, 0.00289219,
                       0.00895708, 0.00526035, 0.00395183, 0.00484049, 0.00348758,
                       0.00683419, 0.00978282, 0.00403886, 0.00023656, 0.00702438,
                       0.0005793 , 0.00402568, 0.00324678, 0.00670035, 0.00023835,
                       0.0095298 , 0.00724941, 0.00169614, 0.00051696, 0.00589271,
                       0.00346083, 0.00442665, 0.00527983, 0.00641216, 0.00698679,
                       0.00790771, 0.01137686, 0.00023982, 0.00023579, 0.00941609,
                       0.00087098, 0.00117903, 0.00494886, 0.00620031, 0.00415478,
                       0.00253459, 0.00086936, 0.00740253, 0.00372934, 0.00403918,
                       0.01733016, 0.00412402, 0.00350724, 0.00665391, 0.00877986,
                       0.00067747, 0.00024237, 0.00023513, 0.00083421, 0.00995662,
                       0.00459506, 0.00023513, 0.00023737, 0.00918468, 0.00328696,
                       0.00023629, 0.00412969, 0.00626361, 0.00201092, 0.0108552 ])
    res = np.sum(heston_call(x, M = n)*z_coef)
    return res

def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-4):
    if ticker == 'AAPL':
        params = [0.0508, 0.045, 0.1318, 0.7712, -0.4744, 0.4438]
    elif ticker == 'AMZN':
        params = [0.1117, 0.0455, 0.0713, 0.9648, -0.4538, 0.3708]
    elif ticker == 'GOOGL':
        params = [0.1708, 0.0455, 0.0764, 40.938, -0.6361, 0.7]
    elif ticker == 'MSFT':
        params = [0.0849, 0.0455, 0.0471, 41.2207, -0.6826, 0.8639]
    elif ticker == 'NVDA':
        params = [0.851, 0.0455, 0.2291, 20.3732, -0.3925, 1.0]
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
