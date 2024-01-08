import sys
import yaml
import argparse
import pytorch_lightning as pl
sys.path.append("./src/SLiMe")
from src.slime import (
    Slime,
)
from src.dataset import DataModule
from src.arguments import init_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    SLiMe_config = init_args()
    SLiMe_config = convert_yaml_to_args(config, SLiMe_config)
    # import pdb; pdb.set_trace()
    dm = DataModule(
        train_data_dir=SLiMe_config.train_data_dir,
        val_data_dir=SLiMe_config.val_data_dir,
        test_data_dir=SLiMe_config.test_data_dir,
        batch_size=SLiMe_config.batch_size,
        train_mask_size=SLiMe_config.train_mask_size,
        test_mask_size=SLiMe_config.test_mask_size,
        num_parts=1,
        min_crop_ratio=SLiMe_config.min_crop_ratio,
        dataset_name=SLiMe_config.dataset_name,
    )
    model = Slime(config=SLiMe_config)
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        default_root_dir=config["output_dir"],
        max_epochs=SLiMe_config.epochs,
        devices=config["devices"],
        log_every_n_steps=1,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    # trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


def convert_yaml_to_args(yaml_dict, args):
    args.dataset_name = "sample"
    args.part_names = yaml_dict["part_names"]
    args.checkpoint_dir = yaml_dict["checkpoint_dir"]
    args.test_data_dir = yaml_dict["test_data_dir"]
    args.save_mask = yaml_dict["save_mask"]
    return args


if __name__ == "__main__":
    main()
