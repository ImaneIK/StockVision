import base64
import datetime as dt
import math
import time
from io import BytesIO
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            return jsonify({'error': 'Ticker or model is empty'}), 400

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
                result = black_scholes_merton(price, strike, interest_rate, period, volatility,divedend )
                print(result)
                return jsonify({'result': result})
            except Exception as ex:
                print(f"Error in black_scholes_merton: {str(ex)}")
                return jsonify({'error': 'An error occurred in black_scholes_merton'}), 500

        elif model == 'vasicek':
            # Handle Vasicek simulation
            result, status_code = Vasicek_simulation(data)
            return jsonify(result), status_code

        else:
            return jsonify({'error': 'Invalid model specified'}), 400

    except Exception as ex:
        return jsonify({'error': str(ex)}), 500


# Define the Vasicek simulation function
def Vasicek_simulation(data):
    try:
        r0 = float(data.get('initial_short_rate', 0.02))  # Default initial short rate
        a = float(data.get('mean_reversion_speed', 0.5)  )    # Default mean reversion speed
        b = float(data.get('long_term_mean', 0.03) )    # Default long-term mean
        sigma = float(data.get('vasicek_volatility', 0.01) ) # Default volatility
        T = int(data.get('time_horizon', 10) )      # Default time horizon
        num_steps = int(data.get('number_of_steps', 1000))  # Default number of steps
        num_paths = int(data.get('number_of_paths', 20))      # Default number of paths

        # Perform Vasicek simulation
        try:
            simulated_rates = simulate_vasicek(r0, a, b, sigma, T, num_steps, num_paths)
            MLE_Estimate = Vasicek_MLE(simulated_rates, T / num_steps, a, b)

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
    for i in range(n-1):
        S0 = S0 + r[i]
        S1 = S1 + r[i + 1]
        S00 = S00 + r[i] * r[i]
        S01 = S01 + r[i] * r[i + 1]
    S0 = S0 / (n-1)
    S1 = S1 / (n-1)
    S00 = S00 / (n-1)
    S01 = S01 / (n-1)
    b_MLE = (S1 * S00 - S0 * S01) / (S0 * S1 - S0**2 - S01 + S00)
    a_MLE = 1 / dt * np.log((S0 - b_MLE) / (S1 - b_MLE))

    # Estimation sigma
    beta = 1 / a * (1 - np.exp(-a * dt))
    temp = 0
    for i in range(n-1):
        mi = b * a * beta + r[i] * (1 - a * beta)
        temp = temp + (r[i+1] - mi)**2
    sigma_MLE = (1 / ((n - 1) * beta * (1 - .5 * a * beta)) * temp)**0.5
    return a_MLE, b_MLE, sigma_MLE

#MLE_Estimate = Vasicek_MLE(simulated_rates, T / num_steps)
#print("a_est: " + str(np.round(MLE_Estimate[0], 3)))
#print("b_est: " + str(np.round(MLE_Estimate[1], 3)))
#print("sigma_est: " + str(np.round(MLE_Estimate[2], 3)))


def import_stock_data(ticker, start, end):
    ticker_symbol = ticker.lower()

    df = yf.download(ticker_symbol, start, end)

    prices = df['Close']

    last_price = prices[-1]

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
sp500 = sp500.history(period="14y")
market_prices = sp500['Close']

# Check if prices2 is defined
if 'prices2' in locals() and isinstance(prices, pd.Series):
    prices2 = prices.to_frame()  # Convert to DataFrame
    prices2.index = prices2.index.tz_localize(None)  # Remove timezone
else:
    # Handle the case where prices2 is not defined
    # You can define prices2 or take any other necessary actions
    prices2 = pd.DataFrame()  # For example, create an empty DataFrame

market_prices.index = market_prices.index.tz_localize(None)


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
        sns.distplot(x, ax=ax[0], color='#6666ff')
        sns.distplot(x, hist_kws={'cumulative': True}, kde_kws={'cumulative': True}, ax=ax[1], color='#6666ff')
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


def black_scholes_merton(stock_price, strike_price, rate, time, volatility, divedend):
    try:

        d1 = (log(stock_price / strike_price) + (rate - divedend + volatility ** 2 / 2) * time) / (
                volatility * time ** 0.5)
        d2 = d1 - volatility * time ** 0.5

        call = round(
            stats.norm.cdf(d1) * stock_price * e ** (-divedend * time) - stats.norm.cdf(d2) * strike_price * e ** (
                    -rate * time), 2)
        put = round(
            stats.norm.cdf(-d2) * strike_price * e ** (-rate * time) - stats.norm.cdf(-d1) * stock_price * e ** (
                    -divedend * time), 2)

        call_decision, put_decision = execution_decision(strike_price, stock_price, 'Call'), execution_decision(
            strike_price, stock_price, 'Put')

        return {'strike_price': strike_price, 'call': call, 'put': put, 'call_decision': call_decision,
                'put_decision': put_decision}

    except Exception as ex:
        print(f"Error in black_scholes_merton: {str(ex)}")
        return {'strike_price': 0, 'call': 0, 'put': 0, 'call_decision': 'Error', 'put_decision': 'Error'}


if __name__ == '__main__':
    try:

        # Start the Flask app
        app.run(port=8000)
    except Exception as e:
        print(f"Error: {str(e)}")
        time.sleep(5)  # Wait for a while before restarting the app
