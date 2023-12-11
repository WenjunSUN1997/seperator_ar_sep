import pickle
import dgl
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import networkx as nx

def get_dataloader_gnn(config, goal='train'):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if goal == 'train':
        if config['lang'] == 'fi':
            with open('train_file', 'rb') as file:
                base_path = '../seperator_detection/result/'
                file_name_list = pickle.load(file)
        else:
            with open('train_file_fr', 'rb') as file:
                base_path = '../seperator_detection/result_fr/'
                file_name_list = pickle.load(file)

    else:
        if config['lang'] == 'fi':
            with open('test_file', 'rb') as file:
                base_path = '../seperator_detection/result/'
                file_name_list = pickle.load(file)
        else:
            with open('test_file_fr', 'rb') as file:
                base_path = '../seperator_detection/result_fr/'
                file_name_list = pickle.load(file)

    net_list = []
    netx_list = []
    print('prepare data')
    for file_name in tqdm(file_name_list):
        with open(base_path+file_name+'/nx.nx', 'rb') as file:
            net_ori = pickle.load(file)
            netx_list.append(net_ori)
            net_unit_list = list(net_ori.subgraph(c) for c in nx.connected_components(net_ori))
            if goal == 'train':
                for net_unit in net_unit_list:
                    if len(net_unit.edges()) == 0:
                        continue

                    net = process_net(net_unit, tokenizer, config['max_token_num'])
                    net_list.append(dgl.from_networkx(net, edge_attrs=None, node_attrs=['reading_order',
                                                                                        'center',
                                                                                        'index',
                                                                                        'input_ids',
                                                                                        'attention_mask',
                                                                                        'center_x',
                                                                                        'center_y',
                                                                                        'left',
                                                                                        'right',
                                                                                        'top',
                                                                                        'bottom']).to(config['device']))
            else:
                net = process_net(net_ori, tokenizer, config['max_token_num'])
                net_list.append(dgl.from_networkx(net, edge_attrs=None, node_attrs=['reading_order',
                                                                                    'center',
                                                                                    'index',
                                                                                    'input_ids',
                                                                                    'attention_mask',
                                                                                    'center_x',
                                                                                    'center_y',
                                                                                    'left',
                                                                                    'right',
                                                                                    'top',
                                                                                    'bottom']).to(config['device']))

    return {'net_list': net_list,
            'netx_list': netx_list}

def process_net(net, tokenizer, max_token_num):
    nodes_list = net.nodes()
    for node_index in nodes_list:
        node = net.nodes[node_index]
        text = node['text']
        output_tokenizer = tokenizer([text],
                                      return_tensors='pt',
                                      max_length=max_token_num,
                                      truncation=True,
                                      padding='max_length')
        node['input_ids'] = output_tokenizer['input_ids'].squeeze(0)
        node['attention_mask'] = output_tokenizer['attention_mask'].squeeze(0)
        node['reading_order'] = int(node['reading_order'].split('a')[-1])
        node['center'] = [int((min([4500, x]) / 4500) * 500) for x in node['center']]
        node['center_x'] = node['center'][0]
        node['center_y'] = node['center'][1]
        node['left'] = torch.tensor(node['sign'][0])
        node['right'] = torch.tensor(node['sign'][1])
        node['top'] = torch.tensor(node['sign'][2])
        node['bottom'] = torch.tensor(node['sign'][3])

    edge_list = net.edges()

    for edge in edge_list:
        point_0 = net.nodes[edge[0]]
        point_1 = net.nodes[edge[1]]
        if point_0['reading_order'] != point_1['reading_order']:
            net.edges[edge]['label'] = torch.tensor(0)
        else:
            net.edges[edge]['label'] = torch.tensor(1)

    return net