import numpy as np
import os
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

size_max = (1000, 252)
np.random.seed(365)
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
    z_coeff = np.array([0.00000000e+00, 1.00730936e-02, 1.30193607e-02, 0.00000000e+00,
                        1.91051523e-02, 5.59764909e-03, 0.00000000e+00, 0.00000000e+00,
                        1.22269687e-02, 9.94997887e-03, 1.15970171e-02, 1.47011939e-03,
                        1.88173553e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        7.76062977e-03, 5.59814532e-03, 4.21451933e-03, 7.73668248e-03,
                        9.69719451e-03, 1.39378294e-02, 1.89738137e-02, 1.47448746e-16,
                        9.95012903e-03, 5.86999617e-03, 2.59489844e-02, 0.00000000e+00,
                        7.20899824e-03, 8.31967301e-04, 2.22385962e-02, 0.00000000e+00,
                        1.42267555e-02, 2.95300723e-03, 5.64972006e-03, 2.08411623e-02,
                        3.37406663e-04, 4.65490813e-16, 9.80588939e-03, 0.00000000e+00,
                        1.96301213e-02, 1.66917523e-02, 0.00000000e+00, 9.33463283e-03,
                        5.20615175e-03, 0.00000000e+00, 3.71369922e-02, 6.28835424e-04,
                        0.00000000e+00, 0.00000000e+00, 2.68925545e-02, 1.19287932e-01,
                        3.05492907e-03, 0.00000000e+00, 5.47614695e-03, 8.86087619e-03,
                        1.69112888e-16, 2.06366579e-02, 6.91877652e-02, 1.54211357e-03,
                        1.97039096e-02, 1.14581744e-02, 3.12633393e-16, 9.19108756e-16,
                        1.06151706e-02, 1.87725969e-02, 1.81911774e-16, 0.00000000e+00,
                        4.69514153e-16, 6.92754287e-03, 6.68311922e-03, 1.78890580e-15,
                        2.11650722e-02, 1.21748919e-02, 6.70205997e-03, 8.51362749e-03,
                        1.10841848e-03, 3.82988541e-03, 1.32917933e-02, 1.73557331e-02,
                        1.58154074e-02, 6.54544941e-03, 1.66134937e-02, 1.13032688e-03,
                        2.25847248e-02, 0.00000000e+00, 3.70268181e-03, 1.46518603e-02,
                        0.00000000e+00, 6.31854795e-02, 2.29327603e-04, 1.18796831e-02,
                        2.59116339e-02, 6.77120809e-03, 1.46225518e-02, 8.54550223e-03,
                        2.16734979e-16, 1.17745149e-02, 2.18256218e-16, 6.26069890e-16])
    res = np.sum(heston_call(x, M = n)*z_coeff)
    return res

def fast_heston(maturity, moneyness, strike, ticker):
    if ticker == 'AAPL':
        params = [0.1161, 0.048, 0.048, 26.3169, -0.5891, 0.5851]
    elif ticker == 'GOOGL':
        params = [0.1824, 0.048, 0.0756, 39.7622, -0.3923, 0.8]
    elif ticker == 'AMZN':
        params = [0.1636, 0.048, 0.0819, 35.3728, -0.5614, 0.8]
    else:
        return 'Company information does not exist!'
    x = np.array([moneyness, maturity, strike] + params)
    return heston_surrogate(x)
    

# Create an API endpoint
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract form data from POST request
        company = request.form.get("Company")
        maturity = float(request.form.get("Time to Maturity"))
        moneyness = float(request.form.get("Moneyness"))
        strike = float(request.form.get("Strike"))
        
        # Call your function with the extracted data
        result = fast_heston(maturity, moneyness, strike, company)
        
        # Return the result to the user
        return render_template("result.html", result=result)
    else:
        # Handle GET request (show the form to the user)
        return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
