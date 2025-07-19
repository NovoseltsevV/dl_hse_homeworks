import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel
from pathlib import Path


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    _, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    perplexity = lambda x: np.exp(np.array(x))
    train_perplexities, val_perplexities = perplexity(train_losses), perplexity(val_losses)

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        max_length = lengths.max()
        crop_indices = indices[:, :max_length].to(device) # обрезаем лишние паддинги

        optimizer.zero_grad()
        output = model(crop_indices[:, :-1], lengths - 1)
        loss = criterion(output.transpose(1, 2), crop_indices[:, 1:]) # подходящий размер для crossentropy
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * crop_indices.shape[0]

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):

        max_length = lengths.max()
        crop_indices = indices[:, :max_length].to(device)

        output = model(crop_indices[:, :-1], lengths - 1)
        loss = criterion(output.transpose(1, 2), crop_indices[:, 1:])
        val_loss += loss.item() * crop_indices.shape[0]

    val_loss /= len(loader.dataset)
    return val_loss

def save_checkpoint(
        model, optimizer, epoch, 
        val_loss, best_val_loss, 
        name, scheduler=None,
        save_dir=Path('./')
):
    if scheduler is not None:
        torch.save({
            'epoch': epoch,
            'best_loss': best_val_loss,
            'loss': val_loss,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }, save_dir / name)
    else:
        torch.save({
            'epoch': epoch,
            'best_loss': best_val_loss,
            'loss': val_loss,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, save_dir / name)
    


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5,
          save_dir=Path('./'), prev_best_val_loss=None, train_name=''
          ):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    if prev_best_val_loss is None:
        best_val_loss = torch.inf
    else:
        best_val_loss = prev_best_val_loss

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(model, optimizer, epoch, val_loss,
                            best_val_loss, f'{train_name}_best_checkpoint.pth',
                            scheduler, save_dir)

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())

        save_checkpoint(model, optimizer, epoch, val_loss,
                best_val_loss, f'{train_name}_last_checkpoint.pth',
                scheduler, save_dir)
