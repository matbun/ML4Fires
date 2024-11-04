from Fires._utilities.utils_trainer import get_callbacks, get_loggers
from itwinai.loggers import MLFlowLogger, Prov4MLLogger, LoggersCollection
from Fires.trainer import FabricTrainer
from Fires._utilities.utils_mlflow import setup_mlflow_experiment
from Fires._utilities.utils_general import check_backend
from Fires._utilities.metrics import TverskyLoss, FocalLoss
from Fires._utilities.logger import Logger as logger
from Fires._utilities.decorators import debug
from Fires._utilities.configuration import load_global_config
from Fires._utilities.cli_args_parser import CLIParser
from Fires._utilities.cli_args_checker import checker
from Fires._scalers.minmax import MinMaxScaler
from Fires._scalers.standard import StandardScaler
from Fires._scalers.scaling_maps import StandardMapsPointWise, MinMaxMapsPointWise
import Fires._models.unetpp
from Fires._models.unetpp import UnetPlusPlus
from Fires._models.unet import Unet
import Fires._models
from Fires._macros.macros import (
    CONFIG,
    DRIVERS as drivers,
    SEED,
    TARGETS as targets,
    TRN_YEARS as trn_years,
    VAL_YEARS as val_years,
    DATA_PATH_100KM,
    TORCH_CFG,
    CREDENTIALS_CFG,
    DATA_DIR,
    LOGS_DIR,
    NEW_DS_PATH,
    RUN_DIR,
    SCALER_DIR,
)
from Fires._datasets.torch_dataset import FireDataset
import Fires
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.plugins.environments import MPIEnvironment
from lightning.fabric.strategies.fsdp import FSDPStrategy
import lightning.pytorch as lp
import lightning as L
from torchmetrics.classification import Precision, Recall, F1Score, FBetaScore, MatthewsCorrCoef, ConfusionMatrix
from torchmetrics.regression import MeanSquaredError, ConcordanceCorrCoef
from torchmetrics import F1Score, FBetaScore, MatthewsCorrCoef, Precision, Recall
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
import torch
from itwinai.loggers import MLFlowLogger as Itwinai_MLFLogger, Prov4MLLogger
import mlflow
import urllib3
import os
from typing import Dict, Optional, Tuple
import toml
import xarray as xr
import numpy as np
from datetime import datetime as dt

# Settings the warnings to be ignored
import warnings
warnings.filterwarnings('ignore')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# MLFlow imports

# Itwinai imports

# Pytorch imports

# Lightning imports


# ML4Fires imports

# define logger
_log = logger(log_dir=LOGS_DIR).get_logger("Training_on_100km")


@debug(log=_log)
def init_fabric():
    # check backend if MPS, CUDA or CPU
    backend = check_backend()

    # get loggers for Fabric Trainer
    # _loggers = get_loggers()

    # get callbacks for Fabric Trainer
    _callbacks = get_callbacks()

    # fabric args
    fabric_args = dict(
        accelerator='gpu' if backend in ['mps', 'cuda'] else 'cpu',
        strategy=eval(TORCH_CFG.model.strategy) if backend in ['mps', 'cuda'] else 'auto',
        devices=TORCH_CFG.trainer.devices,
        num_nodes=TORCH_CFG.trainer.num_nodes,
        precision=TORCH_CFG.trainer.precision,
        plugins=eval(TORCH_CFG.trainer.plugins),
        callbacks=_callbacks,
        # loggers=_loggers,
    )

    # # init fabric accelerator
    # fabric = L.Fabric(
    # 	accelerator=backend,
    # 	strategy=eval(TORCH_CFG.model.strategy) if backend in ['mps', 'cuda'] else 'auto',
    # 	devices=TORCH_CFG.trainer.devices,
    # 	num_nodes=TORCH_CFG.trainer.num_nodes,
    # 	precision=TORCH_CFG.trainer.precision,
    # 	plugins=eval(TORCH_CFG.trainer.plugins),
    # 	callbacks=_callbacks,
    # 	loggers=_loggers,
    # )

    return fabric_args


@debug(log=_log)
def get_trainer(fabric_args: Dict) -> FabricTrainer:
    """
    Creates and configures a FabricTrainer instance for model training.

    This function checks the backend, sets up various training parameters 
    including callbacks, devices, logging, epochs, and more, and then 
    initializes the FabricTrainer.

    Returns
    -------
    FabricTrainer
            An instance of the FabricTrainer class configured with the specified training parameters.
    """

    # define trainer args
    trainer_args = {
        # fabric args
        'fabric_args': fabric_args,
        # define number of epochs
        'max_epochs': 5,  # TORCH_CFG.trainer.epochs,
        # define trainer accumulation steps
        'grad_accum_steps': TORCH_CFG.trainer.accumulation_steps,
        # set distributed sampler
        'use_distributed_sampler': eval(TORCH_CFG.trainer.use_distributed_sampler)
    }

    for k in trainer_args.keys():
        _log.info(f" | {k}:{trainer_args[k]}")

    # initialize trainer and its arguments
    trainer = FabricTrainer(**trainer_args)

    return trainer


@debug(log=_log)
def create_torch_datasets(data_source_path: str) -> Tuple[FireDataset, FireDataset]:
    """
    Creates PyTorch datasets for training and validation from the provided data source.

    This function checks if the data source path exists, loads the training data, 
    applies standard scaling, and creates PyTorch datasets for training and validation.

    Parameters
    ----------
    data_source_path : str
            The path to the data source for the 100km dataset.

    Returns
    -------
    Tuple[FireDataset, FireDataset]
            A tuple containing the training and validation datasets as PyTorch FireDataset objects.

    Raises
    ------
    ValueError
            Check if the data source path exists, if it doesn't exist raise an error.
    """

    if not os.path.exists(data_source_path):
        raise ValueError(f"Path to 100km dataset doesn't exists: {data_source_path}")

    # load training data
    data = xr.open_zarr(data_source_path)[drivers+targets]
    train_data = data.sel(time=data.time.dt.year.isin(trn_years)).load()

    # create standard scaler
    mean_std_args = dict(dim=['time', 'latitude', 'longitude'], skipna=True)
    mean_ds = train_data.mean(**mean_std_args)
    stdv_ds = train_data.std(**mean_std_args)
    x_scaler = StandardScaler(mean_ds=mean_ds, stdv_ds=stdv_ds, features=drivers)

    # define pytorch datasets for training and validation
    fire_ds_args = dict(src=data_source_path, drivers=drivers, targets=targets)
    trn_torch_ds = FireDataset(**fire_ds_args, years=trn_years, scalers=[x_scaler, None])
    val_torch_ds = FireDataset(**fire_ds_args, years=val_years, scalers=[x_scaler, None])

    return trn_torch_ds, val_torch_ds


@debug(log=_log)
def setup_model() -> Optional[Unet | UnetPlusPlus]:
    """
    Initializes and configures the UNet model for training.

    This function sets up the model configuration, including input shape, 
    number of classes, depth, and activation function, and then creates an 
    instance of the UNet model. It also sets the loss function and initializes 
    the metrics for the model.

    Returns
    -------
    Unet
            An instance of the Unet class configured for training.
    """

    # define model loss
    model.loss = torch.nn.modules.loss.BCELoss()
    # model.loss = TverskyLoss(alpha=0.5, beta=0.5)

    # define metrics list
    _metrics = []

    # precision
    precision = Precision(task='binary')
    precision.name = "precision"
    _metrics.append(precision)

    # recall
    recall = Recall(task='binary')
    recall.name = "recall"
    _metrics.append(recall)

    # f1 score
    f1_score = F1Score(task='binary')
    f1_score.name = "f1_score"
    _metrics.append(f1_score)

    # f2 score
    f2_score = FBetaScore(task='binary', beta=float(2))
    f2_score.name = "f2_score"
    _metrics.append(f2_score)

    # mcc
    mcc = MatthewsCorrCoef(task='binary')
    mcc.name = "mcc"
    _metrics.append(mcc)

    all_metrics = False

    # define model metrics
    model.metrics = _metrics if all_metrics else []

    _log.info(f" | Model: \n\n {model}")

    return model


@debug(log=_log)
def get_lightning_trainer():

    # check backend
    backend = check_backend()

    # get loggers for Fabric Trainer
    # _loggers = get_loggers(run_name=run_name)
    itwinai_logger = LoggersCollection([
        MLFlowLogger(
            experiment_name=run_name,
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
            log_freq=10
        ),
        Prov4MLLogger(
            savedir=os.path.join(LOGS_DIR, "ITWINAI", "provenance"),
            experiment_name=run_name,
            save_after_n_logs=1
        )
    ])

    # get callbacks for Fabric Trainer
    _callbacks = get_callbacks()

    # seed everything for reproducibility
    lp.seed_everything(seed=SEED, workers=True)

    # define lightining.pytorch.Trainer object
    pl_trainer = lp.Trainer(
        accelerator='gpu' if backend in ['mps', 'cuda'] else 'cpu',
        # eval(TORCH_CFG.model.strategy) if backend in ['mps', 'cuda'] else 'auto',
        strategy='ddp' if backend in ['mps', 'cuda'] else 'auto',
        devices=1,  # TORCH_CFG.trainer.devices,
        num_nodes=1,  # TORCH_CFG.trainer.num_nodes,
        precision=TORCH_CFG.trainer.precision,
        # logger=_loggers,
        callbacks=_callbacks,
        max_epochs=5,  # TORCH_CFG.trainer.epochs,
    )
    pl_trainer.itwinai_logger = itwinai_logger

    return pl_trainer


@debug(log=_log)
def main():
    """
    Main function to execute the model training pipeline.

    This function orchestrates the entire training process by creating datasets, 
    initializing the trainer and model, setting up data loaders, and starting the 
    training process. It also logs the training progress and saves the final model 
    to disk.
    """

    # create pytorch datasets for training and validation
    trn_torch_ds, val_torch_ds = create_torch_datasets(data_source_path=DATA_PATH_100KM)

    # define model
    model = setup_model()

    # load dataloader
    dloader_args = dict(batch_size=TORCH_CFG.trainer.batch_size, shuffle=True,
                        drop_last=TORCH_CFG.trainer.drop_reminder, num_workers=2)
    train_loader = DataLoader(trn_torch_ds,	**dloader_args)
    valid_loader = DataLoader(val_torch_ds, **dloader_args)

    # get fabric
    # fabric = init_fabric()
    # fabric_args = init_fabric()

    # define trainer
    # trainer = get_trainer(fabric_args=fabric_args)

    # setup the model and the optimizer
    # trainer.setup(
    # 	model=model,
    # 	optimizer_cls=eval(TORCH_CFG.trainer.optim.cls),
    # 	optimizer_args=eval(TORCH_CFG.trainer.optim.args),
    # 	scheduler_cls=eval(TORCH_CFG.trainer.scheduler.cls),
    # 	scheduler_args=eval(TORCH_CFG.trainer.scheduler.args),
    # 	checkpoint=eval(TORCH_CFG.trainer.checkpoint.ckpt)
    # )

    # get instance of Pytorch Lightning Trainer
    trainer = get_lightning_trainer()
    with trainer.itwinai_logger.start_logging(rank=trainer.global_rank):

        # get global rank
        global_rank = trainer.global_rank
        print(f" | Global rank {global_rank}")

        # Automatically log params, metrics, and model
        # mlflow.pytorch.autolog()

        # # Initialize MLflow run using the setup_mlflow_run function
        # if global_rank == 0:
        # 	mlflow.start_run(run_name=run_name)

        # fit the model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader
        )

        # save the model to disk
        last_model = os.path.join(RUN_DIR, 'last_model.pt')
        trainer.save_checkpoint(filepath=last_model)

        # trainer.fabric.save(
        # 	path=last_model,
        # 	state={
        # 		'model':trainer.model,
        # 		'optimizer':trainer.optimizer,
        # 		'scheduler': trainer.scheduler_cfg
        # 	}
        # )

        # # log weights
        # if global_rank == 0:
        # 	mlflow.log_artifact(last_model, artifact_path="model_weights")
        trainer.itwinai_logger.log(
            item=last_model,
            identifier="model_weights",
            kind='artifact'
        )

        # log model
        original_model = trainer.model  # trainer.model.module

        # Remove non-serializable attributes
        # if hasattr(original_model, '_fabric'):
        # 	del original_model._fabric
        # if hasattr(original_model, 'comm'):
        # 	del original_model.comm

        original_model.cpu()

        trainer.itwinai_logger.log(
            item=original_model,
            identifier="last_model",
            kind='model'
        )

        # # log model
        # if global_rank == 0:
        # 	mlflow.pytorch.log_model(original_model, "last_model")

        # # end MLFlow run
        # if global_rank == 0:
        # 	mlflow.end_run()


@debug(log=_log)
def check_cli_args():
    """
    Parses and validates command-line arguments for configuring a UNet model for training.

    This function displays the program name and description, then parses command-line 
    arguments related to the UNet model's configuration, such as the base filter dimension 
    and the activation function for the last layer. It ensures that all required arguments 
    are provided and sets default values if necessary. Based on the parsed arguments, it 
    constructs and returns a dictionary containing the model configuration.

    Returns
    -------
    dict
            A dictionary containing the UNet model configuration with the following keys:
            - 'input_shape': The shape of the input data (fixed as (180, 360, 7)).
            - 'base_filter_dim': The base filter dimension for the UNet model, as specified by the user.
            - 'activation': The activation function for the last layer, either Sigmoid or ReLU, 
              based on the user's choice.
    """

    PROGRAM_NAME = ""
    PROGRAM_DESCRIPTION = "The following script is designed to perform the training of a ML model that must predict Wildfires Burned Areas on global scale."

    options = [
        [('-bfd', '--base_filter_dim'), dict(type=int, default=32,
                                             help='Base filter dimension for Unet (default: 32)')],
        [('-afn', '--activation'), dict(type=str, choices=['S', 'R'], default='S',
                                        help='Activation function for the last layer: S - Sigmoid (default) | R - ReLU')],
        [('-mdl', '--model'), dict(type=str, choices=['unet', 'unetpp'], default='unet',
                                   help='Name of the model that must be trained: unet (default) | unetpp - UNet++')],
    ]
    cli_parser = CLIParser(program_name=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
    cli_parser.add_arguments(parser=None, options=options)
    cli_args = cli_parser.parse_args()

    activation_fn = torch.nn.Sigmoid() if cli_args.activation == 'S' else torch.nn.ReLU()

    cli_base_filter_dim = cli_args.base_filter_dim

    model_config = {
        'input_shape': (180, 360, 7),
        'base_filter_dim': cli_base_filter_dim,
        'activation': activation_fn
    }

    cli_model_name = cli_args.model

    if cli_model_name == 'unet':
        model_class = Fires._models.unet.Unet
    elif cli_model_name == 'unetpp':
        model_class = Fires._models.unetpp.UnetPlusPlus
    else:
        raise ValueError(f"Model not supported: {cli_args.model}")

    global run_name
    run_name = f"LOCAL_{cli_model_name.upper()}_BCE_{cli_base_filter_dim}"

    return model_class, model_config


if __name__ == '__main__':

    # setup MLFlow experiment
    setup_mlflow_experiment()

    # check cli args
    model_class, model_config = check_cli_args()
    for k in model_config.keys():
        print(f"{k}: {model_config[k]}")
    print("\n\n")

    # get model class and configuration
    global model
    model = model_class(**model_config)
    print(f"Model: {model}")

    main()
