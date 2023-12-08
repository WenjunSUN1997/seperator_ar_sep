from model_components.sep_dataloader import get_dataloader
import argparse
from tqdm import tqdm

def train(kwargs):
    epoch_num = 1000
    train_dataloader = get_dataloader(kwargs)
    for epoch_index in range(epoch_num):
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            print()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='dbmdz/bert-base-historic-multilingual-64k-td-cased')
    parser.add_argument("--path", default='../../seperator_detection/result/')
    parser.add_argument("--max_token_num", default=512, type=int)
    parser.add_argument("--device", default='cuda:1')
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float)
    args = parser.parse_args()
    config = vars(args)
    train(config)




