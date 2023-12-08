from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pickle
import torch
import os

class SepDateset(Dataset):
    def __init__(self,
                 model_name='dbmdz/bert-base-historic-multilingual-cased',
                 device='cuda:1',
                 max_token_num=512,
                 file_path='../../seperator_detection/result/'):
        super(SepDateset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data = self.organize_data(file_path)
        self.max_token_num = max_token_num
        self.device = device

    def organize_data(self, file_path):
        result = []
        file_name_list = os.listdir(file_path)
        for file_name in file_name_list:
            with open(file_path + file_name + '/nx.nx', 'rb') as file:
                predict_net = pickle.load(file)

            edge_list = predict_net.edges()
            for edge in edge_list:
                result.append([predict_net.nodes[edge[0]], predict_net.nodes[edge[0]]])

        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        text_0 = data[0]['text']
        center_0 = [int((min([4500, x]) / 4500) * 500) for x in data[0]['center']]
        x_0 = torch.tensor(center_0[0]).to(self.device)
        y_0 = torch.tensor(center_0[1]).to(self.device)
        left_0 = torch.tensor(list(data[0]['sign'])[0]).to(self.device)
        right_0 = torch.tensor(list(data[0]['sign'])[1]).to(self.device)
        top_0 = torch.tensor(list(data[0]['sign'])[2]).to(self.device)
        bottom_0 = torch.tensor(list(data[0]['sign'])[3]).to(self.device)
        output_tokenizer_0 = self.tokenizer([text_0],
                                            return_tensors='pt',
                                            max_length=self.max_token_num,
                                            truncation=True,
                                            padding='max_length')
        for key, value in output_tokenizer_0.items():
            output_tokenizer_0[key] = value.to(self.device)

        text_1 = data[1]['text']
        center_1 = [int((min([4500, x]) / 4500) * 500) for x in data[1]['center']]
        x_1 = torch.tensor(center_1[0]).to(self.device)
        y_1 = torch.tensor(center_1[1]).to(self.device)
        left_1 = torch.tensor(list(data[1]['sign'])[0]).to(self.device)
        right_1 = torch.tensor(list(data[1]['sign'])[1]).to(self.device)
        top_1 = torch.tensor(list(data[1]['sign'])[2]).to(self.device)
        bottom_1 = torch.tensor(list(data[1]['sign'])[3]).to(self.device)
        output_tokenizer_1 = self.tokenizer([text_1],
                                            return_tensors='pt',
                                            max_length=self.max_token_num,
                                            truncation=True,
                                            padding='max_length')
        for key, value in output_tokenizer_1.items():
            output_tokenizer_1[key] = value.to(self.device)

        if data[0]['reading_order'] == data[1]['reading_order']:
            label = torch.tensor(1).to(self.device)
        else:
            label = torch.tensor(0).to(self.device)

        return {'input_ids_0': output_tokenizer_0['input_ids'],
                'attention_mask_0': output_tokenizer_0['attention_mask'],
                'x_0': x_0,
                'y_0': y_0,
                'left_0': left_0,
                'right_0': right_0,
                'top_0': top_0,
                'bottom_0': bottom_0,
                'input_ids_1': output_tokenizer_1['input_ids'],
                'attention_mask_1': output_tokenizer_1['attention_mask'],
                'x_1': x_1,
                'y_1': y_1,
                'left_1': left_1,
                'right_1': right_1,
                'top_1': top_1,
                'bottom_1': bottom_1,
                'label': label}

if __name__ == "__main__":
    dataset_obj = SepDateset()
    print()