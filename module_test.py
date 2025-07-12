import torch.nn as nn
from tqdm import tqdm
from utils import MetricsTop
import torch
import pandas as pd
import os


def do_test(args, model_name, dataset_name, model, dataloader):
    criterion = nn.L1Loss()
    metrics = MetricsTop(args.test_mode).getMetrics(args.dataset_name)
    model.eval()
    y_pred, y_true = [], []
    eval_loss = 0.0

    with torch.no_grad():
        with tqdm(dataloader) as td:
            for batch_data in td:
                vision = batch_data['vision'].to(args.device)
                audio = batch_data['audio'].to(args.device)
                text = batch_data['text'].to(args.device)
                labels = batch_data['labels'].to(args.device)
                labels = labels.view(-1, 1)
                outputs = model(text, audio, vision)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                y_pred.append(outputs.cpu())
                y_true.append(labels.cpu())

            train_loss = eval_loss / len(dataloader)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            y_pred.append(pred.cpu().numpy())
            y_true.append(true.cpu().numpy())

    eval_results = metrics(pred, true)
    eval_results["Loss"] = round(train_loss, 4)
    print(eval_results)


