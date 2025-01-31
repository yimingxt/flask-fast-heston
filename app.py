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

def heston_surrogate(x, n = 300):
    z_coef = np.array([0.0068661 , 0.00274478, 0.00017912, 0.0006712 , 0.00062454,
                       0.00265002, 0.00017912, 0.00499044, 0.00017912, 0.00017913,
                       0.00324162, 0.00536756, 0.00323873, 0.00479659, 0.00499251,
                       0.00489509, 0.00017913, 0.00424575, 0.00912039, 0.00080799,
                       0.00629373, 0.00017913, 0.00385396, 0.00807368, 0.00017912,
                       0.00505934, 0.00069304, 0.00698157, 0.00209309, 0.00427691,
                       0.00017913, 0.00622447, 0.00294584, 0.00396794, 0.00303339,
                       0.00017913, 0.00017913, 0.01241959, 0.00017912, 0.00292003,
                       0.00017912, 0.00135019, 0.00767019, 0.00511249, 0.01147909,
                       0.00719823, 0.00508938, 0.01033756, 0.0029133 , 0.00478055,
                       0.00017913, 0.0048075 , 0.00744708, 0.00198893, 0.00323207,
                       0.00146121, 0.0052834 , 0.00111943, 0.00591464, 0.0002204 ,
                       0.00017913, 0.00389111, 0.0049366 , 0.00439469, 0.00411429,
                       0.00070071, 0.00158272, 0.00351541, 0.00324755, 0.00578716,
                       0.00179683, 0.00017913, 0.00389168, 0.00241336, 0.00017912,
                       0.00125781, 0.01040022, 0.00056611, 0.00443137, 0.0134913 ,
                       0.01328333, 0.00476657, 0.01058176, 0.01171123, 0.01073891,
                       0.00688679, 0.00084163, 0.00017912, 0.00054281, 0.00463588,
                       0.00926843, 0.00719523, 0.00017912, 0.00536674, 0.00344511,
                       0.00017912, 0.00017913, 0.00366104, 0.00707164, 0.00077954,
                       0.00114055, 0.00200028, 0.00274804, 0.00299597, 0.00353411,
                       0.00017913, 0.00102348, 0.00324555, 0.00267074, 0.00086837,
                       0.00246502, 0.00378995, 0.00177352, 0.00169318, 0.00516666,
                       0.00465617, 0.00115756, 0.00017913, 0.00017913, 0.00619117,
                       0.00161694, 0.00017913, 0.00080616, 0.00481872, 0.00539533,
                       0.00103961, 0.00017912, 0.00270367, 0.00035882, 0.00357162,
                       0.00818524, 0.00357414, 0.00017912, 0.00583317, 0.00318009,
                       0.00141734, 0.00323525, 0.00877335, 0.00017912, 0.00078976,
                       0.00692365, 0.0053684 , 0.00693463, 0.00207938, 0.00095507,
                       0.00370101, 0.0052559 , 0.00574849, 0.00103037, 0.01013128,
                       0.00123932, 0.00215374, 0.00017912, 0.01628611, 0.00017912,
                       0.01185747, 0.00456521, 0.0004328 , 0.00476127, 0.00186718,
                       0.00535643, 0.00017912, 0.00017913, 0.0003015 , 0.0022849 ,
                       0.00543849, 0.0050426 , 0.00017912, 0.00213031, 0.01488649,
                       0.00017913, 0.00017912, 0.00017913, 0.00157198, 0.00052944,
                       0.00017912, 0.00017912, 0.00680132, 0.00783567, 0.00043516,
                       0.01159424, 0.00017912, 0.00017912, 0.00210016, 0.00621701,
                       0.00260646, 0.00105719, 0.00040346, 0.00017913, 0.00665898,
                       0.00323948, 0.00474798, 0.00017912, 0.00017913, 0.00027721,
                       0.00037883, 0.00038788, 0.00017912, 0.00361924, 0.00904944,
                       0.01013753, 0.00041292, 0.00126847, 0.00403394, 0.00757829,
                       0.00485607, 0.00041564, 0.00147016, 0.00254715, 0.00048378,
                       0.00524344, 0.00582976, 0.00017913, 0.00075021, 0.00017913,
                       0.00224406, 0.00306198, 0.0044783 , 0.00148511, 0.00239695,
                       0.0067082 , 0.00020686, 0.00017912, 0.00265184, 0.00017912,
                       0.0089875 , 0.00313153, 0.00225406, 0.00404858, 0.00100448,
                       0.00017912, 0.01028425, 0.00437195, 0.00290949, 0.00017913,
                       0.00090593, 0.00076805, 0.00358317, 0.00017913, 0.00434549,
                       0.00494044, 0.00550168, 0.0009271 , 0.00017912, 0.01100777,
                       0.00954452, 0.00017912, 0.00307155, 0.00273514, 0.00302488,
                       0.00390784, 0.00248432, 0.00017912, 0.00539555, 0.003144  ,
                       0.00017912, 0.006106  , 0.00030552, 0.00248607, 0.01158313,
                       0.00609241, 0.00017912, 0.00144108, 0.00577753, 0.0066376 ,
                       0.00462451, 0.00089503, 0.00133244, 0.00358784, 0.00141818,
                       0.00025585, 0.00765815, 0.00017912, 0.00017912, 0.00017912,
                       0.00211806, 0.00611642, 0.00070737, 0.00492091, 0.00017913,
                       0.00017912, 0.00066922, 0.00278532, 0.00384231, 0.00619197,
                       0.00017912, 0.00046962, 0.00647349, 0.00017912, 0.00593318,
                       0.00719   , 0.0031274 , 0.00481018, 0.00150822, 0.00967129,
                       0.00017912, 0.00017912, 0.00271934, 0.0013924 , 0.00230894])
    res = np.sum(heston_call(x, M = n)*z_coef)
    return res

def fast_heston(maturity, moneyness, strike, ticker, delta = 1e-4):
    if ticker == 'AAPL':
        params = [0.1137, 0.0455, 0.0437, 13.8709, -0.95, 0.3643]
    elif ticker == 'AMZN':
        params = [0.2367, 0.0455, 0.0773, 22.0887, -0.95, 0.7719]
    elif ticker == 'GOOGL':
        params = [0.3183, 0.0455, 0.0719, 37.969, -0.95 , 0.625]
    elif ticker == 'MSFT':
        params = [0.1148, 0.0455, 0.0436, 27.0047, -0.95, 0.451]
    elif ticker == 'NVDA':
        params = [0.2627, 0.0455, 0.2476, 2, -0.4136, 0.6307]

    else:
        return 'Company information not available!'
    spot = strike/(1-moneyness)
    x = np.array([moneyness, maturity, strike] + params)
    x_plus = np.array([moneyness + delta, maturity, strike] + params)
    x_minus = np.array([moneyness - delta, maturity, strike] + params)
    return heston_surrogate(x), (heston_surrogate(x_plus) - heston_surrogate(x_minus))/(2*delta)*K/(S0**2)


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
