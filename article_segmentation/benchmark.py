import networkx as nx
from as_evaluation import evaluate
import os
import pickle
from transformers import AutoTokenizer, AutoModel
import argparse
import torch

def post_process_net(predict_net: nx.Graph):
    node_list = [x[1]['index'] for x in predict_net.nodes().data()]
    prediction = []
    truth = []
    while len(node_list) > 0:
        temp = [node_list[0]]
        for index in range(1, len(node_list)):
            if nx.has_path(predict_net, node_list[0], node_list[index]):
                temp.append(node_list[index])

        prediction.append(temp)
        for value in temp:
            node_list.remove(value)

    node_list = [x[1]['index'] for x in predict_net.nodes().data()]
    while len(node_list) > 0:
        temp = [node_list[0]]
        for index in range(1, len(node_list)):
            if predict_net.nodes[node_list[index]]['reading_order'] == predict_net.nodes[node_list[0]]['reading_order']:
                temp.append(predict_net.nodes[node_list[index]]['index'])

        truth.append(temp)
        for value in temp:
            node_list.remove(value)

    error_value_list = evaluate(prediction, truth).tolist()
    return error_value_list


def benchmark(file_path, tokenizer, model, threshold, type, device):

    def analysis_single_page(predict_net):
        result = []
        edge_list = predict_net.edges()
        for edge in edge_list:
            text_1 = predict_net.nodes[edge[0]]['text']
            text_2 = predict_net.nodes[edge[1]]['text']
            output_tokenizer_1 = tokenizer([text_1],
                                           is_split_into_words=True,
                                           padding="max_length",
                                           max_length=512,
                                           truncation=True,
                                           return_tensors='pt')
            output_tokenizer_2 = tokenizer([text_2],
                                           is_split_into_words=True,
                                           padding="max_length",
                                           max_length=512,
                                           truncation=True,
                                           return_tensors='pt')
            for key, value in output_tokenizer_1.items():
                output_tokenizer_1[key] = value.to(device)

            for key, value in output_tokenizer_2.items():
                output_tokenizer_2[key] = value.to(device)
            if type == 'mean':
                semantic_1 = torch.mean(model(**output_tokenizer_1)['last_hidden_state'], dim=1)
                semantic_2 = torch.mean(model(**output_tokenizer_2)['last_hidden_state'], dim=1)
                if torch.cosine_similarity(semantic_1, semantic_2, dim=-1) < threshold:
                    predict_net.remove_edge(edge[0], edge[1])

        return post_process_net(predict_net)

    file_list = os.listdir(file_path)
    aer_list = []
    acr_list = []
    for file_name in file_list:
        with open(file_path + file_name + '/' + 'nx.nx', 'rb') as file:
            predict_net = pickle.load(file)

        aer_list = aer_list + analysis_single_page(predict_net)
        acr_list = acr_list + [1-x for x in aer_list]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda:2')
    parser.add_argument("--file_path", default=r'../seperator_detection/result/')
    parser.add_argument("--model_name", default='dbmdz/bert-base-historic-multilingual-cased')
    parser.add_argument("--threshold", default=0.0, type=float)
    parser.add_argument("--type", default='mean', choices=['mean', 'end'])
    args = parser.parse_args()
    device = args.device
    file_path = args.file_path
    model_name = args.model_name
    threshold = args.threshold
    type = args.type
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'llama' not in model_name.lower():
        model = AutoModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name).bfloat16()

    input_args = {'file_path': file_path,
                  'tokenizer': tokenizer,
                  'model': model.to(device),
                  'threshold': threshold,
                  'type': type,
                  'device': device}
    benchmark(**input_args)

