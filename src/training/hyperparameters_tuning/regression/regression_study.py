import os
import json
import seaborn as sns

from abc import ABC
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.training.hyperparameters_tuning.study import Study


class RegressionStudy(Study, ABC):

    def save_results(self, true, results, title):
        preds = results[0]
        plots_dir = os.path.join(self.working_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # scores
        mse = mean_squared_error(true, preds)
        mae = mean_absolute_error(true, preds)
        r2 = r2_score(true, preds)

        results = {'mse': mse,
                   'mae': mae,
                   'r2': r2}

        with open(os.path.join(self.working_dir, f'{title}_scores.json'), "w") as f:
            json.dump(results, f)

        # plots

        f = plt.figure()
        sns.scatterplot(x=true, y=preds)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("True vs Predictions")
        plt.savefig(os.path.join(plots_dir, f'{title}_true_vs_preds.png'))

        f = plt.figure()
        residuals = true - preds
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals")
        plt.title("Residuals Distribution")
        plt.savefig(os.path.join(plots_dir, f'{title}_residuals_distribution.png'))