import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

""" Credit to https://quant-next.com/the-merton-jump-diffusion-model/ for the following functions: """
# -----------------------------------------------------------------------------------------------------------------------------------#

# Simulate Normal Compound Poisson Process
def normal_compound_poisson_process(lambda_, T, m, delta):

    t = 0
    jumps = 0
    event_values = []
    event_times = []
    event_values.append(0)
    event_times.append(0)
    while t < T:
        t = t + np.random.exponential(1 / lambda_)
        jumps = jumps + np.random.normal(m, delta)
        if t < T:
            event_values.append(jumps)
            event_times.append(t)

    return event_values, event_times

# Simulate Merton Jump Diffusion (MJD) Process
def MJD_process(S0, mu, sigma, lambda_, T, m, delta, num_steps):

    dT = T / num_steps
    price = np.zeros(num_steps + 1)
    price[0] = S0
    k = np.exp(m + .5 *delta**2) - 1
    
    for t in range(1, num_steps + 1):
        # Brownian motion
        Z = np.random.normal(0, 1)
        dW = np.sqrt(dT) * Z
        # jumps between t and t + dt
        jumps = np.sum(normal_compound_poisson_process(lambda_, dT, m, delta)[0])
        price[t] = price[t - 1] * np.exp((mu - .5 * sigma**2 - lambda_ * k) * dT + sigma * dW + jumps)
        
    return price

# -----------------------------------------------------------------------------------------------------------------------------------#