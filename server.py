import base64
import json
import math
import os
import time
from io import BytesIO
import matplotlib
import seaborn as sns
import yfinance as yf
from flask import Flask, request, jsonify
from math import log, e
from scipy.stats import norm
from scipy import stats
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import math
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
matplotlib.use('Agg')

app = Flask(__name__)

# Global variables to store user entries
ticker = ""
prices = []
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime.now()


@app.after_request  # Enable CORS for all routes
def enable_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/', methods=['GET', 'POST', 'OPTIONS'])  # Your existing route handling
def process_data():
    try:

        if request.method == 'OPTIONS':
            # Respond to preflight request
            return '', 200

        print("BEFORE getting the query")
        data = request.get_json()
        print(f"after getting the query: {data}")
        model = data.get('model')
        print(f"Received model: {model}")
        ticker = data.get('ticker', '')
        print(f"Received ticker: {ticker}")
        if model == "":
            print("model is empty")
            return jsonify({'error': 'model is empty'}), 400

        elif ticker != "" and model == 'MonteCarlo':
            print("we are in monte carlo")
            # Update global variables with user entries
            ticker = data.get('ticker', '')
            start_date = dt.datetime.strptime(data.get('start_date'), '%Y-%m-%d')
            end_date = dt.datetime.strptime(data.get('end_date'), '%Y-%m-%d')

            prices = import_stock_data(ticker, start_date, end_date)
            days_difference = (end_date - start_date).days

            try:
                result = simulate_MonteCarlo(prices, days_difference, 1000, 'log', True)
                return jsonify({'result': result})
            except Exception as ex:
                print(f"Error in simulate_MonteCarlo: {str(ex)}")
                return jsonify({'error': 'An error occurred in simulate_MonteCarlo'}), 500

        elif ticker != "" and model == 'BlackScholes':
            print("we are in BlackScholes")
            divedend_str = data.get('divedend')
            volatility_str = data.get('volatility')
            strike_str = data.get('strike')
            period_str = data.get('period')
            crr = data.get('crr')
            interest_rate_str = data.get('interest_rate')
            start_date = dt.datetime.strptime(data.get('start_date'), '%Y-%m-%d')
            end_date = dt.datetime.strptime(data.get('end_date'), '%Y-%m-%d')
            stock_price = import_stock_data(ticker, start_date, end_date)
            price = stock_price.iloc[-1]

            divedend = float(divedend_str) if divedend_str is not None else None
            volatility = float(volatility_str) if volatility_str is not None else None
            strike = float(strike_str) if strike_str is not None else None
            period = float(period_str) if period_str is not None else None
            interest_rate = float(interest_rate_str) if interest_rate_str is not None else None

            print(model, ticker, volatility, strike, period, interest_rate, start_date, end_date)

            try:
                result = black_scholes_merton(price, strike, interest_rate, period, volatility, divedend, crr)
                return jsonify({'result': result})
            except Exception as ex:
                print(f"Error in black_scholes_merton: {str(ex)}")
                return jsonify({'error': 'An error occurred in black_scholes_merton'}), 500

        # generate_and_plot_trees - binomial & trinomial
        elif ticker != "" and model == 'binomial_and_trinomial':
            print("we are in binomial_&_trinomial")
            # Parameters
            start_date = dt.datetime.strptime(data.get('start_date'), '%Y-%m-%d')
            end_date = dt.datetime.strptime(data.get('end_date'), '%Y-%m-%d')
            stock_price = import_stock_data(ticker, start_date, end_date)
            S0 = stock_price.iloc[-1]# Initial stock price
            K = data.get('strike')  # Strike price
            T = data.get('time')  # Time to expiration in years
            r = data.get('interest_rate')  # Risk-free interest rate

            n = data.get('number_of_steps')  # Number of steps
            option_type = data.get('option_type')
            american = data.get('method_type')


            strike = float(K) if K is not None else None
            S0 = float(S0) if S0 is not None else None
            T = int(T) if T is not None else None
            r = float(r) if r is not None else None

            n = int(n) if n is not None else None
            american = bool(american) if american is not None else None

            print(S0, K, T, r, n, option_type, american)

            try:
                result = generate_and_plot_trees(S0, strike, T, r, n, option_type, american)
                print(result)
                return jsonify({'result': result})
            except Exception as ex:
                print(f"Error in binomial_and_trinomial: {str(ex)}")
                return jsonify({'error': 'An error occurred in binomial_and_trinomial'}), 500

        elif model == 'vasicek':
            # Handle Vasicek simulation
            result, status_code = Vasicek_simulation(data)
            return jsonify(result), status_code

        elif ticker == "" and model =="YC":
            result, status_code = all_plots()

            print(jsonify(result))
            return result, status_code


        else:
            return jsonify({'error': 'Invalid model specified'}), 400

    except Exception as ex:
        return jsonify({'error': str(ex)}), 500



# binomial & trinomial


def generate_binomial_tree(S0, K, T, r, n, option_type, american):
    """Generates a binomial tree for American option pricing.

    Args:
        S0 (float): Initial stock price
        K (float): Option strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate (as a decimal)
        u (float): Up factor (e.g., 1.1 for 10% increase)
        d (float): Down factor (e.g., 0.9 for 10% decrease)
        n (int): Number of time steps
        option_type (str): 'call' or 'put'
        american (bool): If True, calculate American option; otherwise, European.

    Returns:
        tuple: Stock price tree, Option value tree
    """
    dt = T / n  # Time step

    u = np.exp(sigma * np.sqrt(2 * dt))  # Up factor
    d = 1 / u  # Down factor
    q = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability of up move

    m = 1  # Middle factor (price stays the same)


    discount = np.exp(-r * dt)  # Discount factor per time step

    # Step 1: Generate stock price tree
    stock_tree = []
    for i in range(n + 1):
        level = [S0 * (u**j) * (d**(i-j)) for j in range(i + 1)]
        stock_tree.append(level)

    # Step 2: Generate option value tree
    option_tree = [[0.0 for _ in range(i + 1)] for i in range(n + 1)]

    # Calculate option value at maturity (leaf nodes)
    for j in range(n + 1):
        if option_type == "call":
            option_tree[n][j] = max(stock_tree[n][j] - K, 0)  # Call option payoff
        elif option_type == "put":
            option_tree[n][j] = max(K - stock_tree[n][j], 0)  # Put option payoff

    # Step 3: Backward induction to calculate option value at each node
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = (q * option_tree[i + 1][j + 1] + (1 - q) * option_tree[i + 1][j]) * discount
            if option_type == "call":
                intrinsic_value = max(stock_tree[i][j] - K, 0)
            elif option_type == "put":
                intrinsic_value = max(K - stock_tree[i][j], 0)

            if american:
                # American option: Take the maximum of immediate exercise value and continuation value
                option_tree[i][j] = max(continuation_value, intrinsic_value)
            else:
                # European option: Only continuation value matters
                option_tree[i][j] = continuation_value

    return stock_tree, option_tree


def plot_binomial_tree_with_option(tree, option_tree, option_type):
    """Plots the binomial tree with stock prices and option values at each node.

    Args:
        tree (list of lists): Stock price tree.
        option_tree (list of lists): Option value tree.
        option_type (str): 'call' or 'put' for labeling.
    """
    n = len(tree) - 1  # Number of steps

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, level in enumerate(tree):
        x_positions = np.arange(-i, i + 1, 2)
        y_positions = [n - i] * len(level)

        # Plot stock prices at nodes
        ax.scatter(x_positions, y_positions, color='cyan', s=100, zorder=3)

        # Label nodes with stock prices
        for j, price in enumerate(level):
            ax.text(x_positions[j], y_positions[j], f'S: {price:.2f}', ha='center', va='center', fontsize=9, color='black', zorder=4)

        # Label nodes with option prices
        for j, option_price in enumerate(option_tree[i]):
            ax.text(x_positions[j], y_positions[j] - 0.5, f'V: {option_price:.2f}', ha='center', va='center', fontsize=9, color='black', zorder=4)

        # Plot connecting lines
        if i > 0:
            prev_x_positions = np.arange(-(i-1), (i-1) + 1, 2)
            for j in range(len(level)):
                ax.plot([prev_x_positions[j // 2], x_positions[j]], [y_positions[0] + 1, y_positions[0]], color='black', zorder=1)

    # Formatting the plot
    ax.set_title(f'Binomial Tree with Asset Prices and {option_type.capitalize()} Option Values')
    ax.set_axis_off()  # Hide axes


    # Encode the call plot as a base64 string
    img_buffer_call = BytesIO()
    plt.savefig(img_buffer_call, format='png')
    img_buffer_call.seek(0)
    binomial_plot = base64.b64encode(img_buffer_call.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the call figure to free memory

    return binomial_plot




def generate_trinomial_tree(S0, K, T, r, n, option_type, american):
    """Generates a trinomial tree for American option pricing.

    Args:
        S0 (float): Initial stock price
        K (float): Option strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate (as a decimal)
        u (float): Up factor (e.g., 1.1 for 10% increase)
        d (float): Down factor (e.g., 0.9 for 10% decrease)
        n (int): Number of time steps
        option_type (str): 'call' or 'put'
        american (bool): If True, calculate American option; otherwise, European.

    Returns:
        tuple: Stock price tree, Option value tree
    """
    dt = T / n  # Time step
    m = 1  # Middle factor (price stays the same)
    u = np.exp(sigma * np.sqrt(2 * dt))  # Up factor
    d = 1 / u  # Down factor

    q_up = ((np.exp(r * dt) - d) * (m - d)) / ((u - d) * (u - m))
    q_down = ((u - np.exp(r * dt)) * (np.exp(r * dt) - d)) / ((u - d) * (u - m))
    q_mid = 1 - q_up - q_down
    discount = np.exp(-r * dt)  # Discount factor


    # Step 1: Generate stock price tree
    stock_tree = []
    for i in range(n + 1):
        level = [S0 * (u**j) * (m**(i - j)) * (d**(i - 2*j)) for j in range(i + 1)]
        stock_tree.append(level)

    # Step 2: Generate option value tree
    option_tree = [[0.0 for _ in range(i + 1)] for i in range(n + 1)]

    # Calculate option value at maturity (leaf nodes)
    for j in range(n + 1):
        if option_type == "call":
            option_tree[n][j] = max(stock_tree[n][j] - K, 0)  # Call option payoff
        elif option_type == "put":
            option_tree[n][j] = max(K - stock_tree[n][j], 0)  # Put option payoff

    # Step 3: Backward induction to calculate option value at each node
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = (
                q_up * option_tree[i + 1][min(j + 1, i + 1)] +  # Prevent out-of-bounds
                q_mid * option_tree[i + 1][j] +
                q_down * option_tree[i + 1][max(j - 1, 0)]  # Prevent out-of-bounds
            ) * discount

            if option_type == "call":
                intrinsic_value = max(stock_tree[i][j] - K, 0)
            elif option_type == "put":
                intrinsic_value = max(K - stock_tree[i][j], 0)

            if american:
                # American option: Take the maximum of immediate exercise value and continuation value
                option_tree[i][j] = max(continuation_value, intrinsic_value)
            else:
                # European option: Only continuation value matters
                option_tree[i][j] = continuation_value

    return stock_tree, option_tree

def plot_trinomial_tree_with_option(tree, option_tree, option_type):
    """Plots the trinomial tree with stock prices and option values at each node.

    Args:
        tree (list of lists): Stock price tree.
        option_tree (list of lists): Option value tree.
        option_type (str): 'call' or 'put' for labeling.
    """
    n = len(tree) - 1  # Number of steps

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, level in enumerate(tree):
        x_positions = np.arange(-i, i + 1, 2)
        y_positions = [n - i] * len(level)

        # Plot stock prices at nodes
        ax.scatter(x_positions, y_positions, color='cyan', s=100, zorder=3)

        # Label nodes with stock prices
        for j, price in enumerate(level):
            ax.text(x_positions[j], y_positions[j], f'S: {price:.2f}', ha='center', va='center', fontsize=9, color='black', zorder=4)

        # Label nodes with option prices
        for j, option_price in enumerate(option_tree[i]):
            ax.text(x_positions[j], y_positions[j] - 0.5, f'V: {option_price:.2f}', ha='center', va='center', fontsize=9, color='black', zorder=4)

        # Plot connecting lines
        if i > 0:
            prev_x_positions = np.arange(-(i-1), (i-1) + 1, 2)
            for j in range(len(level)):
                ax.plot([prev_x_positions[j // 2], x_positions[j]], [y_positions[0] + 1, y_positions[0]], color='black', zorder=1)

    # Formatting the plot
    ax.set_title(f'Trinomial Tree with Asset Prices and {option_type.capitalize()} Option Values')
    ax.set_axis_off()  # Hide axes

    # Encode the call plot as a base64 string
    img_buffer_call = BytesIO()
    plt.savefig(img_buffer_call, format='png')
    img_buffer_call.seek(0)
    trinomial_plot = base64.b64encode(img_buffer_call.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the call figure to free memory

    return trinomial_plot


def generate_and_plot_trees(S0, K, T, r, n, option_type, american):
    """Generates both binomial and trinomial trees, plots them, and returns JSON with paths to saved plot images."""

    # Generate binomial tree
    binomial_stock_tree, binomial_option_tree = generate_binomial_tree(S0, K, T, r, n, option_type, american)

    # Generate trinomial tree
    trinomial_stock_tree, trinomial_option_tree = generate_trinomial_tree(S0, K, T, r, n, option_type, american)

    # Plot both trees and get file paths for saved images
    #  plot_trinomial_tree_with_option(stock_tree, option_tree, option_type="call")

    binomial_plot_path = plot_binomial_tree_with_option(binomial_stock_tree, binomial_option_tree, option_type)
    trinomial_plot_path = plot_trinomial_tree_with_option(trinomial_stock_tree, trinomial_option_tree, option_type)

    return {
        'binomial_tree_plot': binomial_plot_path,
        'trinomial_tree_plot': trinomial_plot_path
    }




# Define the Vasicek simulation function
def Vasicek_simulation(data):
    try:
        r0 = float(data.get('initial_short_rate', 0.02))  # Default initial short rate
        a = float(data.get('mean_reversion_speed', 0.5))  # Default mean reversion speed
        b = float(data.get('long_term_mean', 0.03))  # Default long-term mean
        sigma = float(data.get('vasicek_volatility', 0.01))  # Default volatility
        T = int(data.get('time_horizon', 10))  # Default time horizon
        num_steps = int(data.get('number_of_steps', 1000))  # Default number of steps
        num_paths = int(data.get('number_of_paths', 20))  # Default number of paths

        # Perform Vasicek simulation
        try:
            print("starting vasicek simulation")
            simulated_rates = simulate_vasicek(r0, a, b, sigma, T, num_steps, num_paths)
            MLE_Estimate = Vasicek_MLE(simulated_rates, T / num_steps, a, b)
            print("calculation done ... proceeding to plotting")
            # Generate a plot
            fig, ax = plt.subplots()
            ax.plot(simulated_rates)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Interest Rate')
            ax.set_title('Simulated Interest Rate Paths')

            # Encode the plot as a base64 string
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plot_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            # Time axis
            time_axis = np.linspace(0, T, num_steps + 1)

            # Average value
            average_rates = [r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t)) for t in time_axis]

            # Standard deviation
            std_dev = [(sigma * 2 / (2 * a) * (1 - np.exp(-2 * a * t))) * .5 for t in time_axis]

            # Calculate upper and lower bounds (±2 sigma)
            upper_bound = [average_rates[i] + 2 * std_dev[i] for i in range(len(time_axis))]
            lower_bound = [average_rates[i] - 2 * std_dev[i] for i in range(len(time_axis))]

            # Simulate interest rate paths (num_paths x num_steps)
            simulated_rates = simulate_vasicek(r0, a, b, sigma, T, num_steps, num_paths)

            # Plotting multiple paths with time on x-axis
            plt.figure(figsize=(10, 6))
            plt.title('Vasicek Model - Simulated Interest Rate Paths')
            plt.xlabel('Time (years)')
            plt.ylabel('Interest Rate')
            for i in range(num_paths):
                plt.plot(time_axis, simulated_rates[:, i])

            plt.plot(time_axis, average_rates, color='black', linestyle='--', label='Average', linewidth=3)
            plt.plot(time_axis, upper_bound, color='grey', linestyle='--', label='Upper Bound (2Σ)', linewidth=3)
            plt.plot(time_axis, lower_bound, color='grey', linestyle='--', label='Lower Bound (2Σ)', linewidth=3)
            plt.legend()

            # Encode the plot as a base64 string
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plot_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            print("plotting ")

            result = {
                'a_est': round(MLE_Estimate[0], 3),
                'b_est': round(MLE_Estimate[1], 3),
                'sigma_est': round(MLE_Estimate[2], 3),
                'plot': plot_data
            }
            return {'result': result}, 200
        except Exception as ex:
            return {'error': str(ex)}, 500

    except Exception as ex:
        return {'error': str(ex)}, 500


def simulate_vasicek(r0, a, b, sigma, T, num_steps, num_paths):
    dt = T / num_steps
    rates = np.zeros((num_steps + 1, num_paths))
    rates[0] = r0

    for t in range(1, num_steps + 1):
        dW = np.random.normal(0, 1, num_paths)
        rates[t] = rates[t - 1] + a * (b - rates[t - 1]) * dt + sigma * np.sqrt(dt) * dW

    return rates


def Vasicek_MLE(r, dt, a, b):
    r = r[:, 0]
    n = len(r)
    # Estimation a and b
    S0 = 0
    S1 = 0
    S00 = 0
    S01 = 0
    for i in range(n - 1):
        S0 = S0 + r[i]
        S1 = S1 + r[i + 1]
        S00 = S00 + r[i] * r[i]
        S01 = S01 + r[i] * r[i + 1]
    S0 = S0 / (n - 1)
    S1 = S1 / (n - 1)
    S00 = S00 / (n - 1)
    S01 = S01 / (n - 1)
    b_MLE = (S1 * S00 - S0 * S01) / (S0 * S1 - S0 ** 2 - S01 + S00)
    a_MLE = 1 / dt * np.log((S0 - b_MLE) / (S1 - b_MLE))

    # Estimation sigma
    beta = 1 / a * (1 - np.exp(-a * dt))
    temp = 0
    for i in range(n - 1):
        mi = b * a * beta + r[i] * (1 - a * beta)
        temp = temp + (r[i + 1] - mi) ** 2
    sigma_MLE = (1 / ((n - 1) * beta * (1 - .5 * a * beta)) * temp) ** 0.5
    return a_MLE, b_MLE, sigma_MLE


# MLE_Estimate = Vasicek_MLE(simulated_rates, T / num_steps)
# print("a_est: " + str(np.round(MLE_Estimate[0], 3)))
# print("b_est: " + str(np.round(MLE_Estimate[1], 3)))
# print("sigma_est: " + str(np.round(MLE_Estimate[2], 3)))


def import_stock_data(ticker, start, end):
    ticker_symbol = ticker.lower()

    df = yf.download(ticker_symbol, start, end)

    prices = df['Close']

    last_price = prices.iloc[-1]

    return prices


def log_returns(prices):
    log_returns = np.log(1 + prices.pct_change())
    return log_returns

    # simple ret


def simple_returns(prices):
    return (prices / prices.shift(1)) - 1


def market_data_combination(prices2, mark_ticker=ticker.lower(), start='2010-1-1'):
    market_data = import_stock_data(mark_ticker, start)

    market_rets = log_returns(market_data).dropna()

    ann_return = np.exp(market_rets.mean() * 252) - 1

    data = prices2.merge(market_data, left_index=True, right_index=True)

    return data, ann_return

    # ann_return = np.exp(market_rets.mean() * 252).values - 1

    # data = prices.merge(market_data, left_index=True, right_index=True)

    # return data, ann_return


sp500 = yf.Ticker('^GSPC')
sp500 = sp500.history(period="10y")
market_prices = sp500['Close']

# Check if prices2 is defined
if 'prices2' in locals() and isinstance(prices, pd.Series):
    prices2 = prices.to_frame()  # Convert to DataFrame
    prices2.index = prices2.index.tz_localize(None)  # Remove timezone
else:
    # Handle the case where prices2 is not defined
    # You can define prices2 or take any other necessary actions
    prices2 = pd.DataFrame()  # For example, create an empty DataFrame

market_prices.index = pd.to_datetime(market_prices.index)
market_prices.index = market_prices.index.tz_localize(None)

print(type(market_prices.index))  # Should print: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>


def capm(prices2, market_prices, riskfree_rate=0.025):
    """

    Calculates CAPM metrics for a stock.



    Args:

        stock_prices (pd.DataFrame): Stock prices with adjusted closing prices.

        market_prices (pd.DataFrame): Market prices (e.g., S&P 500) with closing prices.

        riskfree_rate (float): Risk-free rate (default: 0.025).



    Returns:

        pd.DataFrame: DataFrame with CAPM metrics: beta, expected return,

        return's standard deviation ,and Sharpe ratio.

    """
    # Handle potential Series input

    if isinstance(prices2, pd.Series):
        prices2 = prices2.to_frame()

    if isinstance(market_prices, pd.Series):
        market_prices = market_prices.to_frame()

    # Combine data

    data = prices2.merge(market_prices, left_index=True, right_index=True)

    # Calculate log returns

    log_returns = np.log(data / data.shift(1)).dropna()

    # Calculate covariance and market variance

    covariance = log_returns.cov() * 252  # Annualize

    market_variance = log_returns.iloc[:, -1].var() * 252

    # Calculate beta

    beta = covariance.iloc[:-1, -1] / market_variance

    # Calculate market excess return

    market_excess_return = log_returns.iloc[:, -1].mean() * 252 - riskfree_rate

    # Calculate expected return

    expected_return = riskfree_rate + beta * market_excess_return

    # Calculate standard deviation

    standard_deviation = log_returns.iloc[:, :-1].std() * (252 ** 0.5)

    # Calculate Sharpe ratio

    sharpe_ratio = (expected_return - riskfree_rate) / standard_deviation

    # Create result DataFrame

    result = pd.DataFrame({

        'Beta': beta,

        'Expected Return': expected_return,

        'Sharpe Ratio': sharpe_ratio,

        'returns STD': standard_deviation

    })

    return result


def drift_calc(prices, return_type='log'):
    if return_type == 'log':

        lr = log_returns(prices)

    elif return_type == 'simple':

        lr = simple_returns(prices)

    u = lr.mean()

    var = lr.var()

    drift = u - (0.5 * var)

    try:

        return drift.values

    except:

        return drift


# drift = drift_calc(prices)


def daily_returns(prices, days, iterations, return_type='log'):
    ft = drift_calc(prices, return_type)

    if return_type == 'log':

        try:

            stv = log_returns(prices).std().values

        except:

            stv = log_returns(prices).std()

    elif return_type == 'simple':

        try:

            stv = simple_returns(prices).std().values

        except:

            stv = simple_returns(prices).std()

    # Oftentimes, we find that the distribution of returns is a variation

    # of the normal distribution where it has a fat tail

    # This distribution is called cauchy distribution

    dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))

    return dr


def probs_find(predicted, higherthan, on='value'):
    """

    This function calculated the probability of a stock being above a certain threshhold, which can be defined as a value

    (final stock price) or return rate (percentage change)

    Input:

    1. predicted: dataframe with all the predicted prices (days and simulations)

    2. higherthan: specified threshhold to which compute the probability

    (ex. 0 on return will compute the  probability of at least breakeven)

    3. on: 'return' or 'value', the return of the stock or the final value of stock for every

    simulation over the time specified

    """

    if on == 'return':

        predicted0 = predicted.iloc[0, 0]

        predicted = predicted.iloc[-1]

        predList = list(predicted)

        over = [(i * 100) / predicted0 for i in predList if ((i - predicted0) * 100) / predicted0 >= higherthan]

        less = [(i * 100) / predicted0 for i in predList if ((i - predicted0) * 100) / predicted0 < higherthan]

    elif on == 'value':

        predicted = predicted.iloc[-1]

        predList = list(predicted)

        over = [i for i in predList if i >= higherthan]

        less = [i for i in predList if i < higherthan]

    else:

        print("'on' must be either value or return")

    return len(over) / (len(over) + len(less))


def simulate_MonteCarlo(prices, days, iterations, return_type='log', plot=True):
    # Generate daily returns

    returns = daily_returns(prices, days, iterations, return_type)

    # Create empty matrix

    price_list = np.zeros_like(returns)

    # Put the last actual price in the first row of matrix.

    price_list[0] = prices.iloc[-1]

    # Calculate the price of each day

    for t in range(1, days):
        price_list[t] = price_list[t - 1] * returns[t]

    # Plot Option

    if plot:
        x = pd.DataFrame(price_list).iloc[-1]
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(x, ax=ax[0], color='#6666ff')
        sns.histplot(x, cumulative=True, kde=True, ax=ax[1], color='#6666ff')
        plt.xlabel("Stock Price")

        img_buf = BytesIO()
        plt.savefig(img_buf, format='jpeg')
        img_buf.seek(0)

        path1 = base64.b64encode(img_buf.getvalue()).decode('utf-8')

        path2 = pred_plot(prices, days)

        # Clear the existing plot
        plt.close(fig)

    # plt.show()

    # CAPM and Sharpe Ratio

    # Printing information about stock

    try:

        [print(nam) for nam in prices.columns]

    except:

        print(prices.name)

    # print(f"Days: {days - 1}")

    # print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(), 2)}")

    # print(f"Return: {round(100 * (pd.DataFrame(price_list).iloc[-1].mean() - price_list[0, 1]) / pd.DataFrame(price_list).iloc[-1].mean(), 2)}%")

    # print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list), 0, on='return')}")

    result = {
        "Days": days - 1,
        "ExpectedValue": round(pd.DataFrame(price_list).iloc[-1].mean(), 2),
        "Return": round(
            100 * (pd.DataFrame(price_list).iloc[-1].mean() - price_list[0, 1])
            / pd.DataFrame(price_list).iloc[-1].mean(), 2
        ),
        "ProbabilityOfBreakeven": probs_find(pd.DataFrame(price_list), 0, on='return'),
        "path1": path1,
        "path2": path2
    }

    return result


def pred_plot(prices, days, iterations=1000):
    style.use('ggplot')

    last_price = prices.iloc[-1]
    returns = prices.pct_change()

    # Create a list to hold price_series
    price_series_list = []

    for _ in range(iterations):
        count = 0
        daily_vol = returns.std()

        price_series = [last_price * (1 + np.random.normal(0, daily_vol))]

        for _ in range(days):
            if count == 251:
                break
            price_series.append(price_series[count] * (1 + np.random.normal(0, daily_vol)))
            count += 1

        # Append the price_series list to the main list
        price_series_list.append(price_series)

    # Create the DataFrame in one go
    simulation_df = pd.DataFrame(price_series_list).T

    fig = plt.figure()
    fig.suptitle('Monte Carlo Simulation: AAPL')
    plt.plot(simulation_df)
    plt.axhline(y=last_price, color='r', linestyle='-')
    plt.xlabel('Day')
    plt.ylabel('Price')

    img_buf2 = BytesIO()
    plt.savefig(img_buf2, format='jpeg')
    img_buf2.seek(0)
    path = base64.b64encode(img_buf2.getvalue()).decode('utf-8')

    return path


# simudata = simulate_MonteCarlo(prices, 252, 1000, 'log')
# print(simudata)


"""European option pricing """


def execution_decision(strike_price, stock_price, option_type):
    call_decision, put_decision = "", ""
    if option_type == 'Call':
        strike_relation = "in-the-money" if stock_price > strike_price else "out-of-the-money"
        call_decision = f'The call option is {strike_relation}.'
    elif option_type == 'Put':
        strike_relation = "in-the-money" if stock_price < strike_price else "out-of-the-money"
        put_decision = f'The put option is {strike_relation}.'
    else:
        put_decision = "Invalid option type"
        call_decision = "Invalid option type"

    return call_decision, put_decision


# Cox-Ross-Rubinstein Binomial Model
# Derivatives Analytics with Python
#

# Model Parameters
#
S0 = 100.0  # index level
K = 100.0  # option strike
T = 1.0  # maturity date
r = 0.05  # risk-less short rate
sigma = 0.2  # volatility


# Valuation Function
def CRR_option_value(S0, K, T, r, sigma, otype, M=4):
    ''' Cox-Ross-Rubinstein European option valuation.

    Parameters
    ==========
    S0 : float
        stock/index level at time 0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    otype : string
        either 'call' or 'put'
    M : int
        number of time intervals
    '''
    # Time Parameters
    dt = T / M  # length of time interval
    df = math.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = math.exp(sigma * math.sqrt(dt))  # up movement
    d = 1 / u  # down movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability

    # Array Initialization for Index Levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Inner Values
    if otype == 'call':
        V = np.maximum(S - K, 0)  # inner values for European call option
    else:
        V = np.maximum(K - S, 0)  # inner values for European put option

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (q * V[0:M - z, t + 1]
                         + (1 - q) * V[1:M - z + 1, t + 1]) * df
        z += 1
    return V[0, 0]


def black_scholes_merton(stock_price, strike_price, rate, time, volatility, dividend, crr):
    try:
        # Calculate d1 and d2
        d1 = (log(stock_price / strike_price) + (rate - dividend + volatility ** 2 / 2) * time) / (
                volatility * time ** 0.5)
        d2 = d1 - volatility * time ** 0.5

        # Calculate call and put prices using BSM formula
        call = round(
            stats.norm.cdf(d1) * stock_price * e ** (-dividend * time) - stats.norm.cdf(d2) * strike_price * e ** (
                    -rate * time), 2)
        put = round(
            stats.norm.cdf(-d2) * strike_price * e ** (-rate * time) - stats.norm.cdf(-d1) * stock_price * e ** (
                    -dividend * time), 2)

        call_decision = execution_decision(strike_price, stock_price, 'Call')
        put_decision = execution_decision(strike_price, stock_price, 'Put')

        plot = None

        # If crr is True, call plot_convergence for both call and put
        if crr:
            callplot, putplot = plot_convergence(10, 200, 10, stock_price, call, put, strike_price, rate, time, volatility)

        return {
            'strike_price': strike_price,
            'call': call,
            'put': put,
            'call_decision': call_decision,
            'put_decision': put_decision,
            'crr_call_plot_data': callplot,
            'crr_put_plot_data': putplot,

        }

    except Exception as ex:
        print(f"Error in black_scholes_merton: {str(ex)}")
        return {
            'strike_price': 0,
            'call': 0,
            'put': 0,
            'call_decision': 'Error',
            'put_decision': 'Error',
            'crr_plot_data': None
        }


import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import jsonify


import matplotlib.pyplot as plt
import base64
from io import BytesIO

def plot_convergence(mmin, mmax, step_size, stock_price, call, put, strike_price, rate, time, volatility):
    ''' Plots the CRR option values for increasing number of time intervals M against the Black-Scholes-Merton benchmark value.'''

    BSM_benchmark_call = call
    BSM_benchmark_put = put

    m = range(mmin, mmax, step_size)
    CRR_call_values = [CRR_option_value(stock_price, strike_price, time, rate, volatility, 'call', M) for M in m]
    CRR_put_values = [CRR_option_value(stock_price, strike_price, time, rate, volatility, 'put', M) for M in m]

    # Create a figure for the Call Option plot
    fig_call, ax_call = plt.subplots(figsize=(9, 5))
    ax_call.plot(m, CRR_call_values, label='CRR Call Values')
    ax_call.axhline(BSM_benchmark_call, color='r', ls='dashed', lw=1.5, label='BSM Call Benchmark')
    ax_call.set_xlabel('# of binomial steps $M$')
    ax_call.set_ylabel('European Call Option Value')
    ax_call.set_title('CRR Call vs BSM Call')
    ax_call.legend()
    ax_call.grid()

    # Encode the call plot as a base64 string
    img_buffer_call = BytesIO()
    plt.savefig(img_buffer_call, format='png')
    img_buffer_call.seek(0)
    call_plot_data = base64.b64encode(img_buffer_call.getvalue()).decode('utf-8')
    plt.close(fig_call)  # Close the call figure to free memory

    # Create a figure for the Put Option plot
    fig_put, ax_put = plt.subplots(figsize=(9, 5))
    ax_put.plot(m, CRR_put_values, label='CRR Put Values')
    ax_put.axhline(BSM_benchmark_put, color='b', ls='dashed', lw=1.5, label='BSM Put Benchmark')
    ax_put.set_xlabel('# of binomial steps $M$')
    ax_put.set_ylabel('European Put Option Value')
    ax_put.set_title('CRR Put vs BSM Put')
    ax_put.legend()
    ax_put.grid()

    # Encode the put plot as a base64 string
    img_buffer_put = BytesIO()
    plt.savefig(img_buffer_put, format='png')
    img_buffer_put.seek(0)
    put_plot_data = base64.b64encode(img_buffer_put.getvalue()).decode('utf-8')
    plt.close(fig_put)  # Close the put figure to free memory

    return call_plot_data, put_plot_data


#black_scholes_merton(100, 100, 0.01, 30, 0.2, 0, "true")

# ==========================================================================================
# Yield curve
# ==========================================================================================
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import curve_fit


# Function to calculate the Nelson-Siegel yield curve
def nelson_siegel(tau, beta0, beta1, beta2, tau1):
    return beta0 + beta1 * (1 - np.exp(-tau / tau1)) / (tau / tau1) + beta2 * ((1 - np.exp(-tau / tau1)) / (tau / tau1) - np.exp(-tau / tau1))

import numpy as np
import plotly.graph_objects as go


def cir_model_plot(r0=0.05, theta=0.03, kappa=0.1, sigma=0.02, T=10, num_paths=10):
    """
    Generate a CIR model simulation plot using Plotly.
    """
    # Define the CIR model simulation
    def cir_model(r0, theta, kappa, sigma, T, num_paths=1000):
        dt = 1 / 252  # Trading days
        steps = int(T * 252)

        rates = np.zeros((num_paths, steps + 1))
        rates[:, 0] = r0

        for t in range(1, steps + 1):
            dr = (kappa * (theta - rates[:, t - 1]) * dt +
                  sigma * np.sqrt(np.maximum(rates[:, t - 1], 0)) * np.sqrt(dt) *
                  np.random.normal(0, 1, num_paths))
            rates[:, t] = np.maximum(rates[:, t - 1] + dr, 0)

        return rates

    # Generate rates
    rates = cir_model(r0, theta, kappa, sigma, T, num_paths)

    # Create Plotly figure
    fig = go.Figure()

    for i in range(num_paths):
        fig.add_trace(go.Scatter(x=np.linspace(0, T, int(T * 252) + 1),
                                 y=rates[i, :],
                                 mode='lines',
                                 name=f'Path {i + 1}',
                                 line=dict(width=1)))

    # Update layout
    fig.update_layout(title='CIR Model Simulation',
                      xaxis_title='Time (Years)',
                      yaxis_title='Interest Rate',
                      hovermode="x unified",
                      template="plotly_white")

    # Return the figure as a JSON string
    return fig.to_json()


import numpy as np
import plotly.graph_objects as go


def hull_white_model_plot(r0=0.05, theta_func=lambda t: 0.03 + 0.01 * np.sin(2 * np.pi * t),
                          kappa=0.1, sigma=0.02, T=10, num_paths=10):
    """
    Generate a Hull-White model simulation plot using Plotly.
    """

    def hull_white_model(r0, theta_func, kappa, sigma, T, num_paths=1000):
        dt = 1 / 252  # Trading days
        steps = int(T * 252)

        rates = np.zeros((num_paths, steps + 1))
        rates[:, 0] = r0

        for t in range(1, steps + 1):
            time = t * dt
            theta = theta_func(time)
            dr = (theta - kappa * rates[:, t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, num_paths)
            rates[:, t] = rates[:, t - 1] + dr

        return rates

    # Simulate rates
    rates = hull_white_model(r0, theta_func, kappa, sigma, T, num_paths)

    # Create Plotly figure
    fig = go.Figure()

    for i in range(num_paths):
        fig.add_trace(go.Scatter(x=np.linspace(0, T, int(T * 252) + 1),
                                 y=rates[i, :],
                                 mode='lines',
                                 name=f'Path {i + 1}',
                                 line=dict(width=1)))

    # Update layout
    fig.update_layout(title='Hull-White Model Simulation',
                      xaxis_title='Time (Years)',
                      yaxis_title='Interest Rate',
                      hovermode="x unified",
                      template="plotly_white")

    # Return the figure as a JSON string
    return fig.to_json()



def all_plots():
    try:
        # Fetch U.S. Treasury yield data
        tickers = ['^IRX', '^FVX', '^TNX', '^TYX']  # 3-month, 5-year, 10-year, 30-year Treasury yields
        data = yf.download(tickers, period="5y")['Adj Close']
        data.columns = ['3M', '5Y', '10Y', '30Y']

        # Reset index to use dates as a column
        data.reset_index(inplace=True)

        # Extract maturities and yields
        maturities = np.array([3 / 12, 5, 10, 30])  # Maturities in years
        latest_yields = data.iloc[-1][1:].values  # Get the latest yields

        # Interpolation using linear method
        interp_func = interp1d(maturities, latest_yields, kind='linear', fill_value='extrapolate')
        maturity_range = np.linspace(0.0833, 30, num=100)  # 1 month to 30 years
        interpolated_yields = interp_func(maturity_range)

        # Cubic Spline Interpolation
        cs = CubicSpline(maturities, latest_yields)
        cubic_spline_yields = cs(maturity_range)

        # Bootstrapping method
        def bootstrap_yield_curve(maturities, yields):
            """
            Bootstraps the yield curve from given maturities and yields.

            Parameters:
            maturities (array): An array of maturities in years.
            yields (array): An array of corresponding yields.

            Returns:
            DataFrame: A DataFrame containing 'Maturity' and 'Yield' columns.
            """
            yield_curve = []

            for i in range(len(maturities)):
                if i == 0:  # For the first maturity
                    yield_curve.append(yields[i])
                else:
                    # Calculate the yield for the current maturity
                    previous_maturities = maturities[:i]
                    previous_yields = yield_curve[:i]

                    # Check for NaN values
                    if np.any(np.isnan(previous_maturities)) or np.any(np.isnan(previous_yields)):
                        yield_curve.append(np.nan)
                    else:
                        cash_flows = np.array(
                            [100 * np.exp(-previous_yields[j] * previous_maturities[j]) for j in range(i)])
                        yield_value = (100 - np.sum(cash_flows)) / 100  # Simplified yield calculation
                        yield_curve.append(yield_value)

            return pd.DataFrame({'Maturity': maturities, 'Yield': yield_curve})

        # Create a DataFrame for the bootstrapped yield curve
        bootstrapped_yield_curve = bootstrap_yield_curve(maturities, latest_yields)

        # Interpolate the bootstrapped yields to match the maturity range
        bootstrapped_interp_func = interp1d(bootstrapped_yield_curve['Maturity'], bootstrapped_yield_curve['Yield'],
                                            kind='linear', fill_value='extrapolate')
        bootstrapped_yields = bootstrapped_interp_func(maturity_range)

        # Create a DataFrame for the yield curve containing both interpolated and cubic spline yields
        yield_curve_df = pd.DataFrame({
            'Maturity (Years)': maturity_range,
            'Interpolated Yield (%)': interpolated_yields,
            'Cubic Spline Yield (%)': cubic_spline_yields
        })

        # Add bootstrapped yields to the DataFrame
        yield_curve_df = yield_curve_df.merge(bootstrapped_yield_curve, how='left', left_on='Maturity (Years)',
                                              right_on='Maturity')
        yield_curve_df.rename(columns={'Yield': 'Bootstrapped Yield (%)'}, inplace=True)

        # Plot the yield curves
        fig = px.line(
            yield_curve_df,
            x='Maturity (Years)',
            y=['Interpolated Yield (%)', 'Cubic Spline Yield (%)', 'Bootstrapped Yield (%)'],
            title='Constructed U.S. Treasury Yield Curves: Interpolated, Cubic Spline, and Bootstrapped',
            labels={'value': 'Yield (%)', 'Maturity (Years)': 'Maturity (Years)'},
            markers=True
        )

        # Update layout for better aesthetics
        fig.update_layout(
            xaxis_title='Maturity (Years)',
            yaxis_title='Yield (%)',
            template='plotly_white'  # Use a clean white background
        )

        # Convert the plot to JSON
        yield_curve_plot = fig.to_json()

        # Example plots for Nelson-Siegel, CIR, and Hull-White (replace with actual code for generating these plots)
        ns_plot = px.scatter(title="Nelson-Siegel Example").to_json()
        cir_plot = px.scatter(title="CIR Example").to_json()
        hull_white_plot = px.scatter(title="Hull-White Example").to_json()

        # Nelson-Siegel curve
        tau = np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30])
        yields_for_ns = interp_func(tau)  # Interpolate yields to match tau values
        popt, _ = curve_fit(nelson_siegel, tau, yields_for_ns, p0=[0.05, -0.02, 0.02, 2])
        ns_yield_curve = nelson_siegel(tau, *popt)

        # Nelson-Siegel Plot
        ns_plot = go.Figure()
        ns_plot.add_trace(go.Scatter(x=tau, y=ns_yield_curve, mode='lines+markers', name='Nelson-Siegel'))
        ns_plot.update_layout(
            title="Nelson-Siegel Yield Curve",
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield",
            xaxis=dict(tickvals=[1 / 12, 1, 2, 5, 10, 30], ticktext=['1M', '1Y', '2Y', '5Y', '10Y', '30Y']),
            template="plotly_white"
        )

        # CIR model plot
        cir_plot = cir_model_plot()

        # Hull-White model plot
        hull_white_plot = hull_white_model_plot()

        # Combine all plots in a dictionary
        result = {
            "yield_curve_plot": yield_curve_plot,

            "nelson_siegel_plot": ns_plot.to_json(),

            "cir_plot": cir_plot,

            "hull_white_plot": hull_white_plot,
        }

        return result, 200

    except Exception as ex:
        return {"error": str(ex)}, 500



# ================================== END - YIELD CURVE ========================================================



# ================================== RUN MAIN APP ========================================================

#if __name__ == '__main__':
#    try:
 #       app.run(port=8000)
  #  except Exception as e:
   #     print(f"Error: {str(e)}")
    #    time.sleep(5)  # Wait for a while before restarting the app
