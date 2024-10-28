import os

import torch
import torch.nn as nn

from dpm.nn import data_generator, train_singleGPU, preds_singleGPU, parser
from dpm.nn.data_generator import ScalerHandler
from dpm import config_
from dpm.utils import log, common
from dpm.nn import fiveCNN, fiveCNN_with_dropout, fiveCNN_with_dropout_v2,twoCNN_with_dropout
from torch.utils.data import DataLoader, Dataset

logger = log.logger()

# config_train_path = config_.PathConfig(mode="train", approach="DL").base_path
# config_test_path = config_.PathConfig(mode="test", approach="DL").base_path
configPrep = config_.PrepConfig()
configDL = config_.DLParamConfig()
config_cnn = configDL.cnn_params
num_descriptor_features = config_cnn["num_descriptor_features"]
num_metadata_x_features = config_cnn["num_metadata_x_features"]
num_metadata_y_features = config_cnn["num_metadata_y_features"]
num_rt_x_features = config_cnn["num_rt_x_features"]


def split(config_base_path: dict, save_dataset: bool, small_set_flag: bool) -> tuple[Dataset, Dataset]:
    """load dataset for training and split it into train and evaluate subset"""
    dataset_ = data_generator.DatasetTensor(config_base_path, configPrep, configDL,
                                            feat_filename=config_base_path["inputs_fpath"])
    train_dataset, eval_dataset = dataset_.random_split()
    if small_set_flag is True:
        logger.debug("small set were chosen")
        train_dataset, eval_dataset = dataset_.random_sampler_sd(train_size=config_cnn["small_train_size"],
                                                                 eval_size=config_cnn["small_train_size"],
                                                                 train_seed=config_cnn[
                                                                     "small_train_set_seed"],
                                                                 eval_seed=config_cnn[
                                                                     "small_eval_set_seed"])

    # train_loader, eval_loader = dataset_.dataloader(train_dataset, eval_dataset, batch_size=batch_size)
    if save_dataset:
        common.save_json(train_dataset.indices,
                         f"./collate_data/split_idx/train_dataset_indices_small_set_{small_set_flag}.json")
        common.save_json(eval_dataset.indices,
                         f"./collate_data/split_idx/eval_dataset_indices_small_set_{small_set_flag}.json")

    return train_dataset, eval_dataset


def chose_model(architecture, **kwargs):
    match architecture:
        case "fiveCNN":
            logger.debug("Model Architecture->fiveCNN")
            archi = fiveCNN.fiveCNN(configDL, num_descriptor_features, num_metadata_x_features, num_metadata_y_features,
                                    num_rt_x_features)
            fiveCNN.output_model_params(archi)
            return archi
        case "fiveCNN_with_dropout":
            logger.debug("Model Architecture->fiveCNN_with_dropout")
            archi = fiveCNN_with_dropout.fiveCNNwithDropout(configDL, num_descriptor_features, num_metadata_x_features,
                                                            num_metadata_y_features,
                                                            num_rt_x_features, dropout=0.3)
            fiveCNN_with_dropout.output_model_params(archi)
            return archi
        case "fiveCNN_with_dropout_v2":
            logger.debug("Model Architecture->fiveCNN_with_dropout_v2")
            archi = fiveCNN_with_dropout_v2.fiveCNNwithDropout2(configDL, num_descriptor_features,
                                                                num_metadata_x_features,
                                                                num_metadata_y_features,
                                                                num_rt_x_features, dropout=0.2)
            fiveCNN_with_dropout_v2.output_model_params(archi)
            return archi
        case "twoCNN_with_dropout":
            logger.debug("Model Architecture->twoCNN_with_dropout")
            archi = twoCNN_with_dropout.twoCNNwithDropout(configDL, num_descriptor_features,
                                                                num_metadata_x_features,
                                                                num_metadata_y_features,
                                                                num_rt_x_features, dropout=0.2)
            twoCNN_with_dropout.output_model_params(archi)
            return archi

def load_train_obs(architecture: str, batch_size: int, mode: str, save_dataset: bool, small_set_flag: bool,
                   scaling: bool):
    config_base_path = config_.PathConfig(mode=mode, approach="DL").base_path
    train_dataset, eval_dataset = split(config_base_path=config_base_path, save_dataset=save_dataset,
                                        small_set_flag=small_set_flag)
    if scaling:
        scaler_handler = ScalerHandler(train_dataset, scaler_save_path=config_base_path["scaler_path"])
        train_dataset = scaler_handler.fit_scalers()
        eval_dataset = scaler_handler.transform_dataset(eval_dataset)
        logger.debug("........train, evaluate dataset feature scaled")
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              )
    eval_loader = DataLoader(eval_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             )

    # Create the model
    model = chose_model(architecture)
    criterion = nn.MSELoss()
    # criterion = nnm.rmse_loss
    # if mode == "train":
    #     lr = config_cnn["learning_rate"]
    # elif mode == "lrfinder":
    #     lr = config_cnn["learning_date_finder"]
    # else:
    #     raise ValueError("mode not suitable for optimizer learning rate setting")
    lr=config_cnn["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return train_loader, eval_loader, model, optimizer, criterion


def load_test_obs(architecture: str, batch_size: int, mode: str, device, scaling: bool):
    config_base_path = config_.PathConfig(mode=mode, approach="DL").base_path
    test_dataset = data_generator.DatasetTensor(config_base_path, configPrep, configDL,
                                                feat_filename=config_base_path["inputs_fpath"])
    if scaling:
        scaler_handler = ScalerHandler(train_dataset=None, scaler_save_path=config_base_path["scaler_path"])
        test_dataset = scaler_handler.transform_dataset(test_dataset)
        logger.debug("........test dataset feature scaled")
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             )
    print("load_test_obs device", device)
    checkpoint = torch.load(config_base_path["nn_model_fpath"], map_location=device)
    model = chose_model(architecture)
    model = model.to(device)
    # check device
    device = next(model.parameters()).device
    print(f"Model is loaded on: {device}")

    model.load_state_dict(checkpoint["model_state_dict"])

    return test_loader, model


# %%
def main(device: str,
         save_every: int,
         total_epochs: int,
         batch_size: int,
         mode: str,
         model_path: str,
         metric_path: str,
         # test_flag: bool,
         save_dataset_flag: bool,
         small_set_flag: bool,
         run_from_ckp: bool,
         early_stopping: bool,
         patience: int,
         architecture: str,
         scaling_flag: bool
         ) -> None:
    match mode:
        case "train":
            train_loader, eval_loader, model, optimizer, criterion = load_train_obs(architecture=architecture,
                                                                                    batch_size=batch_size,
                                                                                    mode=mode,
                                                                                    save_dataset=save_dataset_flag,
                                                                                    small_set_flag=small_set_flag,
                                                                                    scaling=scaling_flag)

            trainer = train_singleGPU.Trainer(model, train_loader, eval_loader,
                                              optimizer, criterion, device,
                                              total_epochs, save_every, model_path, metric_path,
                                              run_from_ckp, patience)
            if early_stopping:  # if conduct early stopping
                trainer.train_with_earlystop()
            else:
                trainer.train()
        case "test":
            test_loader, model = load_test_obs(architecture, batch_size, mode, device, scaling_flag)
            labels = preds_singleGPU.Prediction.nn_preds(model, test_loader, device)
            # save preds labels into new files
            config_base_path = config_.PathConfig(mode=mode, approach="DL").base_path
            test_df = common.read_csv(config_base_path["pool_fpath"], delimiter_=",")
            test_df["y_preds"] = labels
            common.save_csv(test_df, path=os.path.join(config_base_path["nn_output_path"], "test_preds.csv"))
            return labels
        case "lrfinder":
            import ignite
            from ignite.engine import create_supervised_trainer, create_supervised_evaluator
            config_base_path = config_.PathConfig(mode="train", approach="DL").base_path
            train_loader, eval_loader, model, optimizer, criterion = load_train_obs(architecture=architecture,
                                                                                    batch_size=batch_size,
                                                                                    mode="train",
                                                                                    save_dataset=save_dataset_flag,
                                                                                    small_set_flag=small_set_flag,
                                                                                    scaling=scaling_flag)
            # trainer = train_singleGPU.Trainer(model, train_loader, eval_loader,
            #                                   optimizer, criterion, device,
            #                                   total_epochs, save_every, model_path, metric_path,
            #                                   run_from_ckp, patience)
            model = model.to(device)
            train_singleGPU.lr_finder(model, criterion, optimizer, train_loader,eval_loader, device,
            start_lr=1e-5, end_lr= 10.0, num_iter= 100, diverge_th=5,
            save_path=config_base_path["nn_lrfinder_plotpath"])

if __name__ == "__main__":
    logger.debug("process start~~~~~~~~~")
    args = parser.get_args()

    # world_size = torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(device,
         args.save_every_n_epoch,
         args.num_epochs,
         args.batch_size,
         args.mode,
         args.model_path,
         args.metric_path,
         # args.test,
         args.save_dataset_idx,
         args.run_small_set,
         args.run_from_ckp,
         args.early_stopping,
         args.patience,
         args.architecture,
         args.scaling
         )

    logger.debug("process finish~~~~~~~~~")
