import torch

from torch import optim, nn
import torch.nn.utils.prune as prune
import numpy as np
import scipy
import pandas as pd

def create_model(model_cls, device, *args, filename=None, lr=1e-3,model_class='BatchVAE', **kwargs):
    """
    Simple model serialization to resume training from given epoch.

    :param model_cls: Model definition
    :param device: Device (cpu or gpu)
    :param args: arguments to be passed to the model constructor
    :param filename: filename if the model is to be loaded
    :param lr: learning rate to be used by the model optimizer
    :param kwargs: keyword arguments to be used by the model constructor
    :return:
    """
    model = model_cls(*args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if filename is not None:
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Loaded model epoch: {checkpoint['epoch']}, loss {checkpoint['loss']}")
    else:
        if model_class=='BatchVAE':
            # adding MASK to layer 1
            S = pd.read_csv('mask_1.csv',index_col=0).values
            mask_1 = torch.Tensor(S.T).to("cpu")
            print(mask_1)
            prune.custom_from_mask(model.encoder.fc1, 'weight', mask=mask_1)

            # adding MASK to layer 2
            S = pd.read_csv('mask_2.csv',index_col=0).values

            mask_2 = torch.Tensor(S.T).to("cpu")
            print(mask_2)
            prune.custom_from_mask(model.encoder.fc2, 'weight', mask=mask_2)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Loading model on ", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    return model.to(device), optimizer


def save_model(model, optimizer, epoch, loss, filename, device):
    """
    Save the model to a file.

    :param model: model to be saved
    :param optimizer: model optimizer
    :param epoch: number of epoch, only for information
    :param loss: loss, only for information
    :param filename: where to save the model
    :param device: device of the model
    """
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, filename)
