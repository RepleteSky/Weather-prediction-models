# Standard library
from argparse import ArgumentParser
from copy import deepcopy

# Local application
import climate_learn as cl
from climate_learn.data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from climate_learn.data.processing.era5_constants import (
    CONSTANTS,
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)

# Third party
import pandas as pd
import torch
from scipy.stats import rankdata
from tqdm import tqdm



class ChannelWiseExplainAi:
    def __init__(self, model, data_module):
        self.model = model
        self.data_module = data_module
        # test_loss does not contain MSE.
        self.test_losses = model.val_loss
        self.transforms = model.val_target_transforms

    def replace_constant(self, y, yhat, out_variables):
        for i in range(yhat.shape[1]):
            # if constant replace with ground-truth value
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = y[:, i]
        return yhat

    def masking_channel(self, x, channel_index):
        x[:, :, channel_index] = 0
        return x

    def get_loss(self, model, x, y, out_variables):
        yhat = model.forward(x)
        yhat = self.replace_constant(y, yhat, out_variables)
        loss_dict = {}
        for i, lf in enumerate(self.test_losses):
            loss_name = getattr(lf, "name", f"loss_{i}")
            # Calculate normal data loss
            if self.transforms is not None and self.transforms[i] is not None:
                yhat_ = self.transforms[i](yhat)
                y_ = self.transforms[i](y)
            else:
                yhat_ = yhat
                y_ = y
            losses = lf(yhat_, y_)
            losses = losses.tolist()
            for var_name, loss in zip(out_variables, losses):
                name = f"{loss_name}:{var_name}"
                loss_dict[name] = loss
            loss_dict[f"{loss_name}:aggregate"] = losses[-1]
        return loss_dict

    def sum_loss_dict(self, all_loss_dict, masked_diff_loss):
        for key in all_loss_dict.keys():
            all_loss_dict[key] += masked_diff_loss[key]
        return all_loss_dict

    def average_loss_dict(self, dict, divisor):
        for key1 in dict.keys():
            for key2 in dict[key1].keys():
                dict[key1][key2] /= divisor
        return dict

    def get_var_ids(self, in_vars, search_vars):
        ids_list = []
        for var in search_vars:
            ids_list.append(in_vars.index(var))
        return ids_list

    def test_step(self):
        num_batch = 0
        all_loss_dict = {}
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data_module.test_dataloader()):
                num_batch += 1
                x, y, in_variables, out_variables = batch
                x, y = x.to(self.model.device), y.to(self.model.device)
                loss_dict = self.get_loss(self.model, x, y, out_variables)
                if "all_channel" in all_loss_dict:
                    all_loss_dict["all_channel"] = self.sum_loss_dict(all_loss_dict["all_channel"], loss_dict)
                else:
                    all_loss_dict["all_channel"] = loss_dict
                for i in range(x.shape[2]):
                    copied_x = deepcopy(x)
                    # if in_variables[i] == "2m_temperature" or in_variables[i] == "specific_humidity_925" or in_variables[i] == "temperature_850" or in_variables[i] == "temperature_925":
                    #     continue
                    # var_ids_list = self.get_var_ids(in_variables, ["2m_temperature", "specific_humidity_925", "temperature_850", "temperature_925"])
                    # for ids in var_ids_list:
                    #     masked_x = self.masking_channel(copied_x, ids)
                    masked_x = self.masking_channel(copied_x, i)
                    masked_loss_dict = self.get_loss(self.model, masked_x, y, out_variables)
                    diff_loss_dict = {key: masked_loss_dict[key] - loss_dict.get(key) for key in loss_dict}
                    if in_variables[i] in all_loss_dict:
                        all_loss_dict[in_variables[i]] = self.sum_loss_dict(all_loss_dict[in_variables[i]], masked_loss_dict)
                        all_loss_dict["diff_"+str(in_variables[i])] = self.sum_loss_dict(all_loss_dict["diff_"+str(in_variables[i])], diff_loss_dict)
                    else:
                        all_loss_dict[in_variables[i]] = masked_loss_dict
                        all_loss_dict["diff_"+str(in_variables[i])] = diff_loss_dict
            average_loss_dict = self.average_loss_dict(all_loss_dict, num_batch)
        return average_loss_dict


def main():
    parser = ArgumentParser()

    parser.add_argument("--checkpoint", default=None)

    subparsers = parser.add_subparsers(
        help="Whether to perform direct, iterative, or continuous forecasting.",
        dest="forecast_type",
    )
    direct = subparsers.add_parser("direct")

    direct.add_argument("--era5_dir")
    direct.add_argument("--model", choices=["vit"])
    direct.add_argument("--pred_range", type=int, choices=[6, 24, 48, 72, 120, 240])

    args = parser.parse_args()

    # Set up data
    variables = [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "relative_humidity",
        "specific_humidity",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "toa_incident_solar_radiation",
        "total_precipitation",
        "land_sea_mask",
        "orography",
        "lattitude",
    ]
    history = 1
    window = 1
    subsample = 6
    batch_size = 128
    num_workers = 1
    patch_size = 2
    in_channels = 50
    out_channels = 4

    in_vars = []

    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    if args.forecast_type in ("direct", "continuous"):
        out_variables = ["2m_temperature", "geopotential_500", "temperature_850", "total_precipitation"]
    elif args.forecast_type == "iterative":
        out_variables = variables
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)

    dm = cl.data.IterDataModule(
        f"{args.forecast_type}-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=history,
        window=window,
        pred_range=args.pred_range,
        subsample=subsample,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.setup()

    # Set up deep learning model
    if args.model == "vit":
        model_kwargs = {  # override some of the defaults
            "img_size": (32, 64),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "history": history,
            "patch_size": patch_size,
            "embed_dim": 128,
            "depth": 8,
            "decoder_depth": 2,
            "learn_pos_emb": True,
            "num_heads": 4,
        }
    else:
        raise RuntimeError("Please specify 'architecture' or 'model'")

    optim_kwargs = {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 5,
        "max_epochs": 50,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_forecasting_module(
        data_module=dm,
        model=args.model,
        model_kwargs=model_kwargs,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
    )
    if args.checkpoint is not None:
        loaded_vit = cl.LitModule.load_from_checkpoint(
            args.checkpoint,
            net=model.net,
            optimizer=model.optimizer,
            lr_scheduler=None,
            train_loss=model.train_loss,
            val_loss=model.val_loss,
            test_loss=model.test_loss,
            val_target_transforms=model.val_target_transforms,
            test_target_transforms=model.test_target_transforms,
        )

    # model explain
    xai = ChannelWiseExplainAi(loaded_vit, dm)
    loss = xai.test_step()
    df = pd.DataFrame(loss, columns = list(loss.keys()))
    df.to_csv('loss_diff.csv', encoding='utf-8-sig')

if __name__ == "__main__":
    main()
