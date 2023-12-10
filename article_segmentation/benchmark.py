import networkx as nx
from as_evaluation import evaluate
import os
import pickle
from transformers import AutoTokenizer, AutoModel, LlamaModel
import argparse
import torch
from tqdm import tqdm
from sklearn import metrics

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

    node_list = [x[1]['index'] for x in predict_net.nodes().data()]
    prediction_clustering = []
    truth_clustering = []
    for node in node_list:
        for prediction_index, prediction_unit in enumerate(prediction):
            if node in prediction_unit:
                prediction_clustering.append(prediction_index)

        for truth_index, truth_unit in enumerate(truth):
            if node in truth_unit:
                truth_clustering.append(truth_index)

    ari = [metrics.adjusted_rand_score(truth_clustering, prediction_clustering)]
    nmi = [metrics.normalized_mutual_info_score(truth_clustering, prediction_clustering)]
    homo = [metrics.homogeneity_score(truth_clustering, prediction_clustering)]
    evaluation_result = evaluate(prediction, truth)
    error_value_list = evaluation_result['error_value_list'].tolist()
    ppa = evaluation_result['ppa']
    return {'ari': ari,
            'nmi': nmi,
            'homo': homo,
            'error_value_list': error_value_list,
            'ppa': ppa,
            'p': evaluation_result['p'],
            'r': evaluation_result['r'],
            'f1': evaluation_result['f1']}


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

            if isinstance(model, LlamaModel):
                real_token = output_tokenizer_1['attention_mask'] != 0
                for key, value in output_tokenizer_1.items():
                    output_tokenizer_1[key] = value[real_token]

                real_token = output_tokenizer_2['attention_mask'] != 0
                for key, value in output_tokenizer_2.items():
                    output_tokenizer_2[key] = value[real_token]

            if type == 'mean':
                semantic_1 = torch.mean(model(**output_tokenizer_1)['last_hidden_state'], dim=1)
                semantic_2 = torch.mean(model(**output_tokenizer_2)['last_hidden_state'], dim=1)
            else:
                semantic_1 = model(**output_tokenizer_1)['last_hidden_state'][:, -1, :]
                semantic_2 = model(**output_tokenizer_2)['last_hidden_state'][:, -1, :]

            if torch.cosine_similarity(semantic_1, semantic_2, dim=-1) < threshold:
                predict_net.remove_edge(edge[0], edge[1])
                # print('remove')

        return post_process_net(predict_net)

    file_list = os.listdir(file_path)
    aer_list = []
    acr_list = []
    ari = []
    nmi = []
    homo = []
    ppa = []
    p = []
    r = []
    f1 = []

    for file_name in tqdm(file_list):
        with open(file_path + file_name + '/' + 'nx.nx', 'rb') as file:
            predict_net = pickle.load(file)

        performance = analysis_single_page(predict_net)
        aer_list = aer_list + performance['error_value_list']
        acr_list = acr_list + [1-x for x in aer_list]
        ari += performance['ari']
        nmi += performance['nmi']
        homo += performance['homo']
        ppa += performance['ppa']
        p += performance['p']
        r += performance['r']
        f1 += performance['f1']

    print('macs: ', sum(acr_list) / len(acr_list))
    print('ppa: ', sum(ppa) / len(ppa))
    print('ari: ', sum(ari) / len(ari))
    print('nmi: ', sum(nmi) / len(nmi))
    print('homo: ', sum(homo) / len(homo))
    print('p: ', sum(p) / len(p))
    print('r: ', sum(r) / len(r))
    print('f1: ', sum(f1) / len(f1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda:2')
    parser.add_argument("--file_path", default=r'../seperator_detection/result/')
    parser.add_argument("--model_name", default='dbmdz/bert-base-historic-multilingual-cased')
    parser.add_argument("--threshold", default=0.5, type=float)
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
        tokenizer.pad_token = tokenizer.eos_token

    input_args = {'file_path': file_path,
                  'tokenizer': tokenizer,
                  'model': model.to(device),
                  'threshold': threshold,
                  'type': type,
                  'device': device}
    benchmark(**input_args)

