import csv

import matplotlib.pyplot as plt
import numpy as np
import padasip as pa

from FilterUtils import Filter

def get_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader.__next__()
        for row in reader:
            data.append(float(row[1]))
    return np.array(data)

def plot_filter_result(pred: np.ndarray, actual: np.ndarray, error: np.ndarray, n: int,
                       mu_val: str='0.05'):
    _avg_error = round(np.mean(10*np.log10(error[: (len(error) - n)]**2)), 2)
    plt.figure(figsize=(15, 20))
    plt.subplot(411)
    plt.title(f'NLMS (mu={mu_val})')
    plt.xlabel('No of iteration [-]')
    plt.plot(actual[: (len(actual)-n)], 'b', label='target')
    plt.plot(pred[: (len(pred)-n)], 'g', label='predict / output')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.title(f'Filter error (avg: {_avg_error})')
    plt.plot(10*np.log10(error[: (len(error) - n)]**2), 'r', label='Error')
    plt.axhline(y=_avg_error, color='black', lw=1, linestyle='--')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return None

def create_filter(data, n, filter_type='NLMS'):
    x = np.squeeze(pa.input_from_history(data, n)) # Input matrix
    N = len(x)
    d = np.zeros(N) # Intialize target with empty values

    for i, k in enumerate(range((n-1), N)):
        # Fill target with our desired values
        d[i] = data[k+1]

    f = Filter(n=n)

    errors_e, mu_range = f.explore_learning(d, x,
                mu_start=0.01,
                mu_end=1.,
                steps=100, ntrain=0.5, epochs=1,
                criteria="MSE")

    best_mu = mu_range[errors_e.argmin()]

    f = pa.filters.AdaptiveFilter(model=filter_type, n=n, mu=best_mu, w="random")
    y, e, w = f.run(d, x)

    plot_filter_result(pred=y, actual=d, error=e, mu_val=best_mu, n=n)

    return  y, e, w 


if __name__ == '__main__':
    n=5
    data =  get_data('data/series/CAT_dividends.csv')
    y, e, w = create_filter(data, n, filter_type='NLMS')
    print(f'Prediction:\t{y[-1]}\nAvg. Error:\t{round(np.mean(10*np.log10(e[: (len(e) - n)]**2)), 2)}')