import os

import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt

sys.path.append("dpm")
from dpm import config_
from dpm.utils import common, log

logger = log.logger()


def loss_plot(train_losses: list, valid_losses: list) -> None:
    """

    :param train_losses: list, list of train loss value
    :param valid_losses: list, list of evaluate loss value
    :return:
    """
    # visualize the loss as the network trained
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    ax.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_losses.index(min(valid_losses)) + 1
    ax.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    ax.axhline(min(valid_losses), linestyle='--', color='b', label='minimum validation point')
    ax.set_yticks([0, min(valid_losses), 1.1])
    ax.set_yticklabels([0, "{:.3f}".format(min(valid_losses)), 1.1])

    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_ylim(0, 1.1)  # consistent scale
    ax.set_xlim(0, len(train_losses) + 1)  # consistent scale
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('./output/nn/loss_plot.png', bbox_inches='tight')
    plt.close()


# draw loss value plot
# file_path = "dpm/nn/save/epoch50_early_stopping/nn_metric.json"
# nn_metric = common.read_json(file_path)
# train_loss = [subdict["train_average_loss"] for subdict in nn_metric]
# valid_loss = [subdict["val_avg_loss"] for subdict in nn_metric]
# loss_plot(train_loss, valid_loss)

# test set MSE RMSE
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error, \
    median_absolute_error, PredictionErrorDisplay
from scipy.stats import pearsonr


def prediction_display(model, y_test, y_pred, random_state, num_sample, save_path):
    display1 = PredictionErrorDisplay.from_predictions(y_test, y_pred, kind="actual_vs_predicted",
                                                       subsample=num_sample,
                                                       random_state=random_state)  # 默认抽取1000个点显示
    plt.savefig(os.path.join(save_path, f"{model}_preds_error_seed{random_state}_subset{num_sample}.png"), dpi=300)
    plt.close()
    # display2 = PredictionErrorDisplay.from_predictions(y_test, y_pred, kind="actual_vs_predicted", subsample=None, )
    # plt.savefig(f"./output/{model}_prediction_error_iter{random_state}_full.png", dpi=300)
    # plt.close()


def metric(y_test, y_pred):
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MedAE = median_absolute_error(y_test, y_pred)
    personr, _ = pearsonr(y_test, y_pred)
    return {"mse": MSE, 'rmse': RMSE, 'r2': R2, 'MAE': MAE, 'medae': MedAE, 'personr': personr}




pathconfig=config_.PathConfig(mode="test")
base_path=pathconfig.base_path
version=pathconfig.version
preds_path = os.path.join(base_path["nn_output_path"],"test_preds.csv")
preds_file = common.read_csv(preds_path, delimiter_=",")
y_rts = preds_file["y_rt"]
y_preds = preds_file["y_preds"]
save_path=f"dpm/nn/save/{version}/"
os.makedirs(save_path, exist_ok=True)
prediction_display(version, y_rts, y_preds, random_state=42, num_sample=10000, save_path=save_path)
metric_dict = metric(y_rts, y_preds)
print(metric_dict)