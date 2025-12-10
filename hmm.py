


!pip install yfinance hmmlearn

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

class MarketRegimeDetector:
    def __init__(self, ticker, start_date):
        self.ticker = ticker
        self.start_date = start_date
        self.data = None
        self.model = None
        self.best_n_states = None

    def fetch_and_process_data(self):

        print(f"Fetching data for {self.ticker}...")
        raw_df = yf.download(self.ticker, start=self.start_date, progress=False, auto_adjust=False)


        if isinstance(raw_df.columns, pd.MultiIndex):
            try:

                series = raw_df[('Adj Close', self.ticker)]
            except KeyError:
                series = raw_df[('Close', self.ticker)]
        else:

            series = raw_df['Adj Close'] if 'Adj Close' in raw_df else raw_df['Close']


        log_returns = np.log(series / series.shift(1))
        self.data = log_returns.dropna().to_frame(name='Log_Returns')
        print(f"Data ready: {len(self.data)} trading days.")
        return self.data

    def _get_param_count(self, n_states, n_features=1):

        mean_cov = n_states * (2 * n_features)
        trans = n_states * (n_states - 1)
        start = n_states - 1
        return mean_cov + trans + start

    def find_optimal_states(self, max_states=5):

        X = self.data.values.reshape(-1, 1)
        results = []

        print("\nOptimizing Model (BIC Score):")
        print("-" * 30)

        for n in range(2, max_states + 1):
            # Train HMM
            _model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=42)
            _model.fit(X)


            log_likelihood = _model.score(X)
            k = self._get_param_count(n)
            N = len(X)


            bic = (np.log(N) * k) - (2 * log_likelihood)

            results.append({'States': n, 'BIC': bic, 'Model': _model})
            print(f"States: {n} | BIC: {bic:.2f}")


        best_result = min(results, key=lambda x: x['BIC'])
        self.best_n_states = best_result['States']
        self.model = best_result['Model']
        print("-" * 30)
        print(f"Selected Optimal States: {self.best_n_states}")

    def analyze_regimes(self):

        if not self.model:
            raise ValueError("Run find_optimal_states() first.")

        means = self.model.means_.flatten()
        vols = np.sqrt(self.model.covars_.flatten())


        stats_df = pd.DataFrame({
            'Regime': range(self.best_n_states),
            'Mean_Return': means,
            'Volatility': vols
        })
        print("\nRegime Characteristics:")
        print(stats_df.round(5))


        last_posterior = self.model.predict_proba(self.data.values.reshape(-1, 1))[-1]
        next_prob = np.dot(last_posterior, self.model.transmat_)
        print(f"\nPrediction for next trading day (Probabilities per Regime):")
        print(np.round(next_prob, 4))

    def visualize(self):

        X = self.data.values.reshape(-1, 1)
        hidden_states = self.model.predict(X)


        plot_df = self.data.copy()
        plot_df['Regime'] = hidden_states
        plot_df['Price_Proxy'] = np.exp(plot_df['Log_Returns'].cumsum())

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[1, 1.5])


        colors = sns.color_palette("deep", self.best_n_states)
        for i in range(self.best_n_states):
            mask = plot_df['Regime'] == i
            ax1.scatter(plot_df.index[mask], plot_df.loc[mask, 'Log_Returns'],
                       s=5, c=[colors[i]], label=f'Regime {i}')
        ax1.set_title(f"Daily Log Returns - {self.ticker}", fontsize=12)
        ax1.legend(loc='upper right', markerscale=2)

        ax2.plot(plot_df.index, plot_df['Price_Proxy'], color='black', lw=1, alpha=0.6)

        changes = np.concatenate(([0], np.diff(hidden_states) != 0, [0]))

        for i in range(self.best_n_states):
             ax2.fill_between(plot_df.index, 0, 1, where=(plot_df['Regime'] == i),
                              transform=ax2.get_xaxis_transform(),
                              color=colors[i], alpha=0.2)

        ax2.set_title("Cumulative Performance (Regime Shaded)", fontsize=12)
        ax2.set_ylabel("Normalized Value")
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    detector = MarketRegimeDetector("AAPL", start_date="2015-01-01")


    detector.fetch_and_process_data()


    detector.find_optimal_states(max_states=5)


    detector.analyze_regimes()


    detector.visualize()
