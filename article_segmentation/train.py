from model_components.sep_dataloader import get_dataloader
import argparse
from tqdm import tqdm
from model_config.sep_model_linear import SepLinear
from model_config.sep_model_gnn import SepGnn
import torch
import os
import pickle
from model_components.validator import validate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_components.gnn_dataloader import get_dataloader_gnn

torch.manual_seed(3407)

def train(kwargs):
    epoch_num = 1000
    if kwargs['type'] == 'linear':
        train_dataloader = get_dataloader(kwargs)
        model = SepLinear(kwargs, weight=train_dataloader.dataset.weight)
    else:
        train_dataloader_temp = get_dataloader(kwargs)
        train_dataloader = get_dataloader_gnn(kwargs)['net_list']
        test_dataloader = get_dataloader_gnn(kwargs, goal='test')
        model = SepGnn(kwargs, weight=train_dataloader_temp.dataset.weight)

    model.to(kwargs['device'])
    for param in model.model.parameters():
        param.requires_grad = kwargs['retrain_backbone']

    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=2,
                                  verbose=True)
    loss_all = [0]
    for epoch_index in range(epoch_num):
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            break
            output = model(data)
            loss = output['loss']
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('train loss: ', sum(loss_all) / (len(loss_all)))
        validate_result = validate(config, model, test_dataloader)
        scheduler.step(validate_result)
        print(epoch_index)
        print('val loss: ', validate_result)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='fi', choices=['fr', 'fi'])
    parser.add_argument("--model_name", default='dbmdz/bert-base-historic-multilingual-64k-td-cased')
    parser.add_argument("--path", default='../../seperator_detection/result/')
    parser.add_argument("--max_token_num", default=512, type=int)
    parser.add_argument("--device", default='cuda:2')
    parser.add_argument("--type", default='gnn', choices=['gnn', 'linear'])
    parser.add_argument("--store_path", default='log/')
    parser.add_argument("--center_flag", action='store_true', default=False)
    parser.add_argument("--encoder_flag", action='store_true', default=True)
    parser.add_argument("--weight", action='store_true', default=True)
    parser.add_argument("--region_flag", action='store_true', default=False)
    parser.add_argument("--retrain_backbone", action='store_true', default=False)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float)
    args = parser.parse_args()
    config = vars(args)
    train(config)




