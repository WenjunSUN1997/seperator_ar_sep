from PIL import Image, ImageDraw
from article_segmentation.benchmark import *
from tqdm import tqdm
import os
import torch
import dgl
import pickle
from transformers import AutoTokenizer
from article_segmentation.model_components.gnn_dataloader import get_dataloader_gnn
import networkx as nx

def analysis_link(point_0, point_1, tokenizer, model, config):
    text_0 = point_0['text']
    output_tokenizer_0 = tokenizer([text_0],
                            return_tensors='pt',
                            max_length=config['max_token_num'],
                            truncation=True,
                            padding='max_length')
    for key, value in output_tokenizer_0.items():
        output_tokenizer_0[key] = value.to(config['device'])

    center_0 = [int((min([4500, x]) / 4500) * 500) for x in point_0['center']]
    x_0 = torch.tensor([center_0[0]]).to(config['device'])
    y_0 = torch.tensor([center_0[1]]).to(config['device'])
    left_0 = torch.tensor([list(point_0['sign'])[0]]).to(config['device'])
    right_0 = torch.tensor([list(point_0['sign'])[1]]).to(config['device'])
    top_0 = torch.tensor([list(point_0['sign'])[2]]).to(config['device'])
    bottom_0 = torch.tensor([list(point_0['sign'])[3]]).to(config['device'])
    text_1 = point_1['text']
    output_tokenizer_1 = tokenizer([text_1],
                                   return_tensors='pt',
                                   max_length=config['max_token_num'],
                                   truncation=True,
                                   padding='max_length')
    for key, value in output_tokenizer_1.items():
        output_tokenizer_1[key] = value.to(config['device'])

    center_1 = [int((min([4500, x]) / 4500) * 500) for x in point_1['center']]
    x_1 = torch.tensor([center_1[0]]).to(config['device'])
    y_1 = torch.tensor([center_1[1]]).to(config['device'])
    left_1 = torch.tensor([list(point_1['sign'])[0]]).to(config['device'])
    right_1 = torch.tensor([list(point_1['sign'])[1]]).to(config['device'])
    top_1 = torch.tensor([list(point_1['sign'])[2]]).to(config['device'])
    bottom_1 = torch.tensor([list(point_1['sign'])[3]]).to(config['device'])
    if point_0['reading_order'] == point_1['reading_order']:
        label = torch.tensor([1]).to(config['device'])
    else:
        label = torch.tensor([0]).to(config['device'])

    input_dict = {'input_ids_0': output_tokenizer_0['input_ids'].unsqueeze(0),
                  'attention_mask_0': output_tokenizer_0['attention_mask'].unsqueeze(0),
                  'x_0': x_0,
                  'y_0': y_0,
                  'left_0': left_0,
                  'right_0': right_0,
                  'top_0': top_0,
                  'bottom_0': bottom_0,
                  'input_ids_1': output_tokenizer_1['input_ids'].unsqueeze(0),
                  'attention_mask_1': output_tokenizer_1['attention_mask'].unsqueeze(0),
                  'x_1': x_1,
                  'y_1': y_1,
                  'left_1': left_1,
                   'right_1': right_1,
                  'top_1': top_1,
                  'bottom_1': bottom_1,
                  'label': label}
    for key, value in input_dict.items():
        input_dict[key] = value.type(torch.int64)

    output_model = model(input_dict)
    return output_model

def validate(config, model, dataloader=None):
    os.makedirs(config['store_path'], exist_ok=True)
    loss_all = [0]
    aer_list = []
    acr_list = []
    ari = []
    nmi = []
    homo = []
    ppa = []
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if config['lang'] == 'fi':
        file_path = '../seperator_detection/result/'
        with open('test_file', 'rb') as file:
            file_name_list = pickle.load(file)
            bath_path = '../article_dataset/AS_TrainingSet_NLF_NewsEye_v2/'
    else:
        file_path = '../seperator_detection/result_fr/'
        with open('test_file_fr', 'rb') as file:
            file_name_list = pickle.load(file)
            bath_path = '../article_dataset/AS_TrainingSet_BnF_NewsEye_v2/'

    def analysis_single_page(predict_net, image_to_draw, file_name):
        drawer = ImageDraw.Draw(image_to_draw, 'RGB')
        edge_list = predict_net.edges()
        for edge in edge_list:
            point_0 = predict_net.nodes[edge[0]]
            point_1 = predict_net.nodes[edge[1]]
            # if point_0['reading_order'] != point_1['reading_order']:
            #     predict_net.remove_edge(edge[0], edge[1])

            result = analysis_link(point_0, point_1, tokenizer, model, config)
            loss_all.append(result['loss'].item())
            if result['path'] == 0:
                predict_net.remove_edge(edge[0], edge[1])

        edge_list = predict_net.edges()
        for edge in edge_list:
            point_0 = predict_net.nodes[edge[0]]
            point_1 = predict_net.nodes[edge[1]]
            drawer.line(point_0['center']+point_1['center'], 'red', width=10)

        image_to_draw.save('log/'+file_name+'.png')
        return post_process_net(predict_net)

    def analysis_single_page_gnn(data, netx, model, image_to_draw=None):
        net_unit_list = list(netx.subgraph(c) for c in nx.connected_components(netx))
        for net_unit in net_unit_list:
            if len(net_unit.edges()) == 0:
                continue
            input = process_net(net_unit, tokenizer, config['max_token_num'], config['device'])
            output = model(input)
            loss_all.append(output['loss'].item())
            # drawer = ImageDraw.Draw(image_to_draw, 'RGB')
            edge_list = netx.edges()
            for edge in edge_list:
                for index in range(len(output['edge_point'])):
                    if output['edge_point'][index][0] == edge[0] \
                            and output['edge_point'][index][1] == edge[1]:
                        if output['path'][index] == 0:
                            netx.remove_edge(edge[0], edge[1])

        edge_list = netx.edges()
        for edge in edge_list:
            point_0 = netx.nodes[edge[0]]
            point_1 = netx.nodes[edge[1]]
            # drawer.line(point_0['center'] + point_1['center'], 'red', width=10)

        return post_process_net(netx)


    if config['type'] == 'linear':
        for file_name in tqdm(file_name_list):
            image_to_draw = Image.open(bath_path + file_name + '.jpg').convert('RGB')
            with open(file_path + file_name + '/' + 'nx.nx', 'rb') as file:
                predict_net = pickle.load(file)

            performance = analysis_single_page(predict_net, image_to_draw, file_name)
            aer_list = aer_list + performance['error_value_list']
            acr_list = acr_list + [1 - x for x in aer_list]
            ari += performance['ari']
            nmi += performance['nmi']
            homo += performance['homo']
            ppa += performance['ppa']

    else:
        net_list = dataloader['net_list']
        netx_list = dataloader['netx_list']
        for index, data in tqdm(enumerate(net_list), total=len(net_list)):
            # image_to_draw = Image.open(bath_path + netx_list + '.jpg').convert('RGB')
            performance = analysis_single_page_gnn(data, netx_list[index], model)
            aer_list = aer_list + performance['error_value_list']
            acr_list = acr_list + [1 - x for x in aer_list]
            ari += performance['ari']
            nmi += performance['nmi']
            homo += performance['homo']
            ppa += performance['ppa']

    print('macs: ', sum(acr_list) / len(acr_list))
    print('ppa: ', sum(ppa) / len(ppa))
    print('ari: ', sum(ari) / len(ari))
    print('nmi: ', sum(nmi) / len(nmi))
    print('homo: ', sum(homo) / len(homo))
    return sum(loss_all) / (len(loss_all)-1)

def process_net(net, tokenizer, max_token_num, device):
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
        node['reading_order'] = node['reading_order']
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

    return dgl.from_networkx(net, edge_attrs=None, node_attrs=['reading_order',
                                                                                    'center',
                                                                                    'index',
                                                                                    'input_ids',
                                                                                    'attention_mask',
                                                                                    'center_x',
                                                                                    'center_y',
                                                                                    'left',
                                                                                    'right',
                                                                                    'top',
                                                                                    'bottom']).to(device)


