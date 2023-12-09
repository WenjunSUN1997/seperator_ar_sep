from model_components.sep_dataloader import get_dataloader
import argparse
from tqdm import tqdm
from model_config.sep_model_linear import SepLinear
import torch
from article_segmentation.model_components.validator import validate
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(kwargs):
    epoch_num = 1000
    train_dataloader = get_dataloader(kwargs)
    model = SepLinear(kwargs)
    model.to(kwargs['device'])
    for param in model.model.parameters():
        param.requires_grad = kwargs['retrain_backbone']

    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=2,
                                  verbose=True)
    loss_all = []
    for epoch_index in range(epoch_num):
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            break
            output = model(data)
            loss = output['loss']
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validate(config, model)
        print(epoch_index)
        print(sum(loss_all) / len(loss_all))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='dbmdz/bert-base-historic-multilingual-64k-td-cased')
    parser.add_argument("--path", default='../../seperator_detection/result/')
    parser.add_argument("--max_token_num", default=512, type=int)
    parser.add_argument("--device", default='cuda:1')
    parser.add_argument("--center_flag", action='store_true', default=True)
    parser.add_argument("--encoder_flag", action='store_true', default=True)
    parser.add_argument("--region_flag", action='store_true', default=True)
    parser.add_argument("--retrain_backbone", action='store_true', default=False)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float)
    args = parser.parse_args()
    config = vars(args)
    train(config)




