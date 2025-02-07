import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


app = Flask(__name__)

n_intra = 4
size_max = (10000, 252*n_intra)
np.random.seed(321)
z1_seed = np.random.normal(size=size_max)
np.random.seed(42)
z2_seed = np.random.normal(size=size_max)

def heston_call(x, dt = 1/(252*n_intra), M = 1, crn = True, div = 0.0):
    moneyness, T, K, V0, r, theta, kappa, rho, sigma = x
    S0 = K/(1-moneyness)
    sqrt_dt = np.sqrt(dt)
    N = int(252*n_intra*T)
    
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
    z_coef = array([3.90214460e-03, 1.59167495e-05, 5.79015589e-03, 9.63102479e-03,
                   4.22292492e-04, 1.07686571e-02, 6.49015668e-03, 1.79358722e-03,
                   9.05472691e-03, 1.75393594e-03, 1.72424308e-03, 5.03916932e-03,
                   1.59167618e-05, 4.95428490e-05, 1.59167121e-05, 4.60306014e-03,
                   4.87583788e-04, 9.55152446e-04, 7.43583970e-03, 3.19279045e-03,
                   5.48742374e-03, 9.29478914e-03, 1.17258246e-02, 4.16305968e-03,
                   1.33817756e-02, 1.72162075e-03, 1.74786457e-03, 2.24005432e-03,
                   1.52848602e-03, 1.42971732e-02, 1.59166295e-05, 1.15930844e-03,
                   7.06510265e-03, 3.67992639e-03, 2.11043854e-03, 1.10327553e-02,
                   5.99214827e-03, 5.93314861e-03, 6.36038764e-03, 3.42478404e-03,
                   1.98149260e-03, 7.92301291e-03, 6.12075952e-03, 1.30649141e-02,
                   4.62735486e-05, 1.59167364e-05, 5.96279546e-03, 1.59165005e-05,
                   3.08095823e-05, 9.88982003e-03, 5.05780514e-03, 1.03406333e-02,
                   1.31179820e-02, 1.21415734e-03, 8.55047536e-03, 4.59378142e-03,
                   1.03766662e-02, 9.15239486e-03, 8.10313363e-05, 2.22695418e-03,
                   1.11125301e-02, 6.68815959e-03, 6.89744928e-03, 1.78611555e-03,
                   8.12978674e-03, 1.20269766e-02, 6.27939254e-03, 2.06670386e-02,
                   1.13551695e-03, 8.82088749e-03, 1.54914079e-02, 1.46230748e-02,
                   7.55961028e-03, 3.25287091e-03, 2.38691767e-03, 6.51583136e-03,
                   8.16428241e-03, 1.12769317e-03, 7.27546569e-03, 1.00794329e-02,
                   2.13335017e-02, 3.22801737e-03, 3.36586576e-03, 8.37611157e-04,
                   5.81252205e-03, 6.97002978e-03, 3.06631867e-03, 1.15981342e-02,
                   6.78154992e-04, 5.62247506e-03, 5.09249860e-03, 2.76992321e-03,
                   1.59165865e-05, 8.15056330e-03, 1.45208300e-03, 1.59165160e-05,
                   1.25506238e-02, 7.15952185e-03, 4.87879256e-05, 5.40458622e-03,
                   3.17055614e-05, 1.59164982e-05, 1.31953282e-03, 5.25924696e-03,
                   1.59165028e-05, 7.47256855e-03, 1.54818534e-02, 1.51825814e-03,
                   7.68529136e-03, 1.20963302e-02, 7.84414632e-04, 6.00789166e-03,
                   3.64513297e-03, 1.59167853e-05, 3.06152929e-03, 8.20006875e-04,
                   1.59165537e-05, 2.10432014e-04, 1.14597129e-02, 7.50894823e-05,
                   3.59077566e-04, 6.10049440e-03, 4.76642253e-03, 3.00477200e-03,
                   2.85010281e-03, 4.28053992e-03, 1.86020896e-02, 5.01779725e-03,
                   1.59167745e-05, 5.42002054e-03, 1.35377589e-02, 6.40975716e-05,
                   2.30185250e-03, 3.76859567e-03, 1.71117022e-05, 1.63692893e-03,
                   9.43710373e-03, 9.68004303e-03, 5.05940151e-03, 1.59167999e-05,
                   6.66178775e-03, 2.57952607e-05, 8.59925884e-03, 8.35863784e-03,
                   2.52645390e-05, 3.59063180e-05, 1.06077025e-03, 7.70956873e-03,
                   5.79973784e-03, 5.44880623e-03, 3.28388852e-03, 7.87874463e-04,
                   9.70155406e-03, 5.53697517e-03, 1.59167805e-05, 2.64074649e-05,
                   1.59166150e-05, 1.59165793e-05, 3.52913559e-03, 6.20205138e-03,
                   1.10935107e-02, 6.21453467e-03, 7.17117744e-04, 8.37159996e-03,
                   1.03512990e-02, 7.63822960e-03, 1.59165557e-05, 1.59167086e-05,
                   2.04089770e-03, 7.07181505e-03, 1.24920246e-02, 1.59165195e-05,
                   3.13448996e-03, 3.61029533e-03, 4.67549447e-03, 5.83338409e-03,
                   4.10326986e-03, 6.07285928e-05, 1.10281805e-02, 4.43474249e-03,
                   1.59166129e-05, 4.49247914e-04, 7.56853035e-03, 5.10783587e-03,
                   6.19901331e-03, 1.86867669e-05, 2.52919988e-03, 1.32539400e-02,
                   7.96060034e-03, 4.73667138e-05, 4.54102353e-03, 2.70720989e-05,
                   3.45114333e-03, 1.16345854e-02, 1.12046183e-02, 1.59165380e-05,
                   2.28047250e-03, 1.32637791e-02, 1.59165414e-05, 1.59165343e-05])
    res = np.sum(heston_call(x, M = n)*z_coef)
    return res


def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-4):
    if ticker == 'AAPL':
        params = [0.0505, 0.0665, 0.01, 4.0353, -0.9497, 0.0883]
    elif ticker == 'AMZN':
        params = [0.4969, 0.045, 0.018, 45.0695, 0.7871, 0.0161]
    elif ticker == 'GOOGL':
        params = [0.0725, 0.044, 0.0641, 4.0317, -0.948, 0.01]
    elif ticker == 'MSFT':
        params = [0.0745, 0.045, 0.0274, 37.9282, 0.4342, 0.01]
    elif ticker == 'NVDA':
        params = [0.851, 0.0455, 0.2291, 20.3732, -0.3925, 1.0]
    else:
        return 'Ticker information not available!'
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
        P_h, D_h = fast_heston(time_to_maturity, moneyness, strike, company)
        
        # Return the result to the user
        return render_template("result.html", price_h=np.round(P_h,2), delta_h=np.round(D_h,5))
    else:
        # Handle GET request (show the form to the user)
        return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
