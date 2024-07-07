import os
from abc import ABC

import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, auc, \
    f1_score

from src.training.hyperparamets_tuning.study import Study


class classification_study(Study, ABC):

    def save_results(self, true, preds, prob, title):
        plots_dir = os.path.join(self.working_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # scores
        precision_curve, recall_curve, _ = precision_recall_curve(true, prob[:, 1])
        fpr, tpr, thresholds = roc_curve(true, prob[:, 1])

        results = {'auprc': auc(recall_curve, precision_curve),
                   'auroc': auc(fpr, tpr),
                   'precision': precision_score(true, preds),
                   'recall': recall_score(true, preds),
                   'f1': f1_score(y_true=true, y_pred=preds),
                   'f1_weighted': f1_score(y_true=true, y_pred=preds, average='weighted')}

        with open(os.path.join(self.working_dir, f'{title}_scores.json'), "w") as f:
            json.dump(results, f)

        # plots

        conf_matrix = confusion_matrix(true, preds)
        f = plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted Negative", "Predicted Positive"],
                    yticklabels=["Actual Negative", "Actual Positive"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(plots_dir, f'{title}_confusion_matrix.png'))

        f = plt.figure()
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(os.path.join(plots_dir, f'{title}_auroc.png'))
