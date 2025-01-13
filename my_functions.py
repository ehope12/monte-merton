import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
from matplotlib.widgets import Slider, TextBox
from quant_functions import MJD_process
# from concurrent.futures import ProcessPoolExecutor

""" My functions: """
# -----------------------------------------------------------------------------------------------------------------------------------#
def get_stock_data(ticker, start_date='1900-01-01', end_date=None):
    """ Retrieves stock data from Yahoo Finance """
    if end_date is None:
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    S0 = data['Close'].iloc[len(data) - 1]
    """
    # Uncomment to save data to csv:
    output_path = os.path.join(os.getcwd(), f'{ticker}.csv')
    data.to_csv(output_path)
    """
    return data, S0

def get_epsilon(S, target_jump_proportion=0.05):
    """ Optimizes epsilon threshold for MJD model """
    if len(S) == 0:
        print('No data for this ticker')
        return
    R = np.log(S / S.shift(1)).dropna()

    sigma_annual = np.std(R) * np.sqrt(252)
    sigma_daily = sigma_annual / np.sqrt(252)
    epsilon = 0.5 * sigma_daily
    tolerance = 0.001
    max_iterations = 100
    iterations = 0

    while iterations < max_iterations:
        jumps = np.abs(R) > epsilon
        jump_proportion = np.sum(jumps) / len(R)
        if jump_proportion > target_jump_proportion:
            epsilon *= 1.05
        elif jump_proportion < target_jump_proportion:
            epsilon *= 0.95
        else:
            break
        if abs(jump_proportion - target_jump_proportion) < tolerance:
            break
        iterations += 1

    k = epsilon / sigma_daily
    return epsilon, jump_proportion, sigma_daily, k, R

def show_epsilon(ticker, start_date='1900-01-01', end_date=None, target_jump_proportion=0.05):
    """ Displays epsilon threshold and jump proportion """
    S, _ = get_stock_data(ticker, start_date, end_date)
    S = S['Close'].dropna()
    epsilon, jump_proportion, sigma_daily, k, R = get_epsilon(S, target_jump_proportion)
    print(f"Final Epsilon: {epsilon:.5f}")
    print(f"Proportion of returns classified as jumps: {jump_proportion:.2%}")
    print(f"Daily Volatility: {sigma_daily:.4f}, Epsilon Multiplier (k): {k:.2f}")
    
    sns.histplot(R, kde=True, bins=100, color='blue', label='Log Returns')
    plt.axvline(epsilon, color='red', linestyle='--', label=r'$\epsilon$ Threshold')
    plt.axvline(-epsilon, color='red', linestyle='--')
    plt.legend()
    plt.title('Log Returns with Optimized Jump Threshold')
    plt.show()

    return epsilon, jump_proportion, sigma_daily, k

def get_initial_parameters(S):
    """ Modeled after code from https://lnu.diva-portal.org/smash/get/diva2:1257256/FULLTEXT01.pdf """
    dt = 1 / 252  # Assume ~252 trading days in a year
    S = S['Close'].dropna()
    epsilon = get_epsilon(S)[0]
    if (len(S) == 0):
        print('No data for this ticker')
        return
    R = np.log(S / S.shift(1))
    jump_indices = np.where(np.abs(R) > epsilon)[0]
    diffusion_indices = np.where(np.abs(R) <= epsilon)[0]
    R_jumps = R[jump_indices]
    R_diffusion = R[diffusion_indices]
    lambda_val = len(jump_indices) / ((len(S) - 1) * dt)
    sigma = np.std(R_diffusion) / np.sqrt(dt)
    mu = (2 * np.mean(R_diffusion) + sigma**2 * dt) / (2 * dt)
    delta = np.sqrt(np.var(R_jumps) - sigma**2 * dt)
    m = np.mean(R_jumps) - (mu - sigma**2 / 2) * dt
    return mu, sigma, lambda_val, m, delta, R, dt

def mjd_pdf(x, mu, sigma, lambda_, m, delta, dt):
    """ PDF for MLE optimization based on https://quant.stackexchange.com/questions/60260/efficient-way-to-perform-mle-on-merton-jump-diffusion-model-parameters """
    k_max = 100  # Truncate infinite sum
    pdf = 0
    for k in range(k_max):
        p_k = (lambda_ * dt)**k / math.factorial(k) * np.exp(-lambda_ * dt)
        mean_k = (mu - sigma**2 / 2) * dt + m * k
        var_k = sigma**2 * dt + delta**2 * k
        pdf += p_k * norm.pdf(x, mean_k, np.sqrt(var_k))
    return pdf

def neg_log_likelihood(params, R, dt):
    """ Negative log likelihood for MLE optimization """
    mu, sigma, lambda_val, m, delta = params
    nll = -np.sum(np.log(mjd_pdf(R, mu, sigma, lambda_val, m, delta, dt)))
    return nll

def optimize_parameters(ticker, start_date='1900-01-01', end_date=None):
    """ Optimizes parameters for MJD model """
    S, S0 = get_stock_data(ticker, start_date, end_date)
    mu, sigma, lambda_val, m, delta, R, dt = get_initial_parameters(S)
    initial_guess = [mu, sigma, lambda_val, m, delta]
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
    result = minimize(neg_log_likelihood, initial_guess, args=(R, dt), bounds=bounds, method='L-BFGS-B')
    return result.x, S0

def run_simulation(args):
    S0, mu, sigma, lambda_val, T, m, delta, num_steps = args
    return MJD_process(S0, mu, sigma, lambda_val, T, m, delta, num_steps)

def monte_carlo_merton(S0, mu, sigma, lambda_val, m, delta, num_steps, T, sims=20):
    """ Monte Carlo simulation for Merton Jump Diffusion model """
    prices = np.zeros((sims, num_steps + 1))
    for i in range(sims):
        prices[i] = MJD_process(S0, mu, sigma, lambda_val, T, m, delta, num_steps)
    return prices
    # Attempt at parallelization (didn't make it faster):
    # args = (S0, mu, sigma, lambda_val, T, m, delta, num_steps)
    # with ProcessPoolExecutor() as executor:
    #     prices = list(executor.map(run_simulation, [args] * sims))
    # return np.array(prices)

def path_average(prices):
    """ Calculate average path for Monte Carlo simulation """
    return np.mean(prices, axis=0)

def plot_stock_comparison_data(ticker, S0, mu, sigma, lambda_val, m, delta, sims, start_date='1900-01-01', end_date=None):
    """ Plots actual stock data against simulated MJD process """
    if end_date is None:
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    data, _ = get_stock_data(ticker, start_date, end_date)
    actual_prices = data['Close']
    if len(actual_prices) == 0:
        print('No data for this ticker', ticker)
        return
    actual_dates = pd.to_datetime(data.index)
    T = len(actual_prices) / 252  # Assume ~252 trading days in a year

    num_steps = len(actual_prices) - 1
    monte = monte_carlo_merton(S0, mu, sigma, lambda_val, m, delta, num_steps, T, sims=sims)
    monte_avg = path_average(monte)

    dates = actual_dates
    plt.figure(figsize=(12, 6))
    for i in range(len(monte)):
        plt.plot(dates, monte[i], color='orange', linestyle='dashed')
    plt.plot(dates, monte_avg, label='Simulated Average', color='red')
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Data: Actual vs Simulated')
    plt.legend()
    plt.grid(True)
    plt.show()

def optimize_and_plot(ticker, start_date='1900-01-01', end_date=None, sims=20):
    """ Optimizes parameters and plots actual stock data against simulated MJD process """
    result, S0 = optimize_parameters(ticker, start_date, end_date)
    mu, sigma, lambda_val, m, delta = result
    print("Parameters: ", mu, sigma, lambda_val, m, delta)
    plot_stock_comparison_data(ticker, S0, mu, sigma, lambda_val, m, delta, sims, end_date, None)

def interactive_stock_plot(ticker='SPY', start_date='1900-01-01', end_date=None, sims=20):
    if end_date is None:
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    result, S0 = optimize_parameters(ticker, start_date, end_date)
    mu, sigma, lambda_val, m, delta = result
    data, _ = get_stock_data(ticker, end_date, None)
    actual_prices = data['Close']
    if len(actual_prices) == 0:
        print('No data for this ticker', ticker)
        return
    actual_dates = pd.to_datetime(data.index)
    T = len(actual_prices) / 252  # Assume ~252 trading days in a year
    num_steps = len(actual_prices) - 1

    monte = monte_carlo_merton(S0, mu, sigma, lambda_val, m, delta, num_steps, T, sims)
    monte_avg = path_average(monte)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.35)

    for i in range(len(monte)):
        ax.plot(actual_dates, monte[i], color='orange', linestyle='dashed')
    ax.plot(actual_dates, monte_avg, label='Simulated Average', color='red')
    ax.plot(actual_dates, actual_prices, label='Actual Prices', color='blue')

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f'{ticker} Stock Data: Actual vs Simulated')
    ax.legend()
    ax.grid(True)

    sims_slider_ax = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor='lightgrey')
    sims_slider = Slider(sims_slider_ax, 'Simulations', 1, 100, valinit=sims, valstep=1)

    axbox_mu = plt.axes([0.1, 0.2, 0.3, 0.05])  # x, y, width, height
    axbox_sigma = plt.axes([0.5, 0.2, 0.3, 0.05])
    axbox_lambda = plt.axes([0.1, 0.15, 0.3, 0.05])
    axbox_m = plt.axes([0.5, 0.15, 0.3, 0.05])
    axbox_delta = plt.axes([0.1, 0.1, 0.3, 0.05])

    mu_box = TextBox(axbox_mu, 'mu:', initial=f"{mu:.5f}")
    sigma_box = TextBox(axbox_sigma, 'mu:', initial=f"{sigma:.5f}")
    lambda_box = TextBox(axbox_lambda, 'lambda:', initial=f"{lambda_val:.5f}")
    m_box = TextBox(axbox_m, 'm:', initial=f"{m:.5f}")
    delta_box = TextBox(axbox_delta, 'delta:', initial=f"{delta:.5f}")

    def update(val):
        mu = float(mu_box.text)
        sigma = float(sigma_box.text)
        lambda_val = float(lambda_box.text)
        m = float(m_box.text)
        delta = float(delta_box.text)
        num_sims = int(sims_slider.val)
        monte_new = monte_carlo_merton(S0, mu, sigma, lambda_val, m, delta, num_steps, T, num_sims)
        monte_avg_new = path_average(monte_new)
       
        for art in list(ax.lines):
            art.remove()
        for i in range(len(monte_new)):
            ax.plot(actual_dates, monte_new[i], color='orange', linestyle='dashed')
        ax.plot(actual_dates, monte_avg_new, label='Simulated Average', color='red')
        ax.plot(actual_dates, actual_prices, label='Actual Prices', color='blue')
        fig.canvas.draw_idle()

    mu_box.on_submit(update)
    sigma_box.on_submit(update)
    lambda_box.on_submit(update)
    m_box.on_submit(update)
    delta_box.on_submit(update)
    sims_slider.on_changed(update)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------#