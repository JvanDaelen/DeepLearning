import logging
import os

import hydra
import pytorch_lightning as pl
import torch

from utils.callbacks import IncreaseSequenceLengthCallback
from utils.utils import *

logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


@hydra.main(config_path="configs", config_name="train_defaults", version_base='1.1')
def train(cfg):
    pl.seed_everything(1234)

    # Update configuration dicts with common keys
    propagate_keys(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Instantiate model and dataloaders
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )
    print("Type of model:", type(model)) # Note 9-4-24 2: models.correlation3_unscaled.TrackerNetC
    print("Model conv_0 size:", model.target_encoder.conv_bottom_0.model)
    print("Train.py checkpoint path:", cfg.checkpoint_path.lower()) # Note 9-4-24 1: path == none so if statement is false
    if cfg.checkpoint_path.lower() != "none":
        # Load weights
        model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)

        # Override stuff for fine-tuning
        model.hparams.optimizer.lr = cfg.model.optimizer.lr
        model.hparams.optimizer._target_ = cfg.model.optimizer._target_
        model.debug = True
        model.unrolls = cfg.init_unrolls
        model.max_unrolls = cfg.max_unrolls
        model.pose_mode = cfg.model.pose_mode

    # Note 32: Investigating this next
    print("cfg.data: ", cfg.data)
    data_module = hydra.utils.instantiate(cfg.data) 

    # Logging
    if cfg.logging:
        training_logger = pl.loggers.TensorBoardLogger(
            ".", "", "", log_graph=False, default_hp_metric=False # Change log_graph to False due to "input_array was not given" error
        )
    else:
        training_logger = None

    # Training schedule
    callbacks = [
        IncreaseSequenceLengthCallback(
            unroll_factor=cfg.unroll_factor, schedule=cfg.unroll_schedule
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Note 25: pl.Trainer is getting unexpected keyword argument 'num_processes'
    # Note 26: Commenting out line 27 num_processes: 1 in train_defaults.yaml allows it to run further now: `Trainer.fit` stopped: No training batches.
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        devices=[0],
        accelerator="gpu",
        callbacks=callbacks,
        logger=training_logger
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
