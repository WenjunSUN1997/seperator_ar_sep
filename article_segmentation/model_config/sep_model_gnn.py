from dgl.nn.pytorch.conv import GraphConv
import dgl
import torch
from article_segmentation.model_config.sep_model_linear import SepLinear

class SepGnn(SepLinear):
    def __init__(self, config, weight):
        super(SepGnn, self).__init__(config, weight)
        self.conv = GraphConv(self.model.config.hidden_size,
                              int(self.model.config.hidden_size / 2),
                              norm='both', weight=True, bias=True)
        self.linear = torch.nn.Linear(in_features=int(self.model.config.hidden_size / 2),
                                      out_features=2)

    def set_node_feature(self, data):
        semantic = []
        for index in range(len(data.ndata['input_ids'])):
            semantic.append(self.model(input_ids=data.ndata['input_ids'][index].unsqueeze(0),
                                       attention_mask=data.ndata['attention_mask'][index].unsqueeze(0))['last_hidden_state'])
        data.ndata['semantic'] = torch.cat(semantic, dim=0)
        data.ndata['center_x'] = self.x_embedding(data.ndata['center_x'])
        data.ndata['center_y'] = self.x_embedding(data.ndata['center_y'])
        data.ndata['left'] = self.left_embedding(data.ndata['left'])
        data.ndata['right'] = self.left_embedding(data.ndata['right'])
        data.ndata['top'] = self.left_embedding(data.ndata['top'])
        data.ndata['bottom'] = self.left_embedding(data.ndata['bottom'])
        if self.encoder_flag:
            data.ndata['semantic'] = torch.mean(self.encoder(data.ndata['semantic']), dim=1)

        if self.center_flag:
            data.ndata['semantic'] = data.ndata['semantic'] + data.ndata['center_x'] + data.ndata['center_y']

        if self.region_flag:
            data.ndata['semantic'] = data.ndata['semantic'] + data.ndata['left']
            data.ndata['semantic'] = data.ndata['semantic'] + data.ndata['right']
            data.ndata['semantic'] = data.ndata['semantic'] + data.ndata['top']
            data.ndata['semantic'] = data.ndata['semantic'] + data.ndata['bottom']

        data.ndata['semantic'] = self.activate(self.normal(data.ndata['semantic']))
        return data

    def apply_edges(self, edges):
        h_u = edges.src['output_conv']
        h_v = edges.dst['output_conv']
        point = torch.cat((edges.src['index'].unsqueeze(1), edges.dst['index'].unsqueeze(1)), dim=1)
        output_linear = self.linear(torch.abs(h_v - h_u))
        label = (edges.src['reading_order'] == edges.dst['reading_order']).long()
        return {'output_linear': output_linear,
                'label': label,
                'point': point}

    def forward(self, data):
        net = self.set_node_feature(data)
        # net = dgl.add_self_loop(net)
        output_conv = self.conv(net, net.ndata['semantic'])
        net.ndata['output_conv'] = output_conv
        # net = dgl.remove_self_loop(net)
        net.apply_edges(self.apply_edges)
        loss = self.loss_func(net.edata['output_linear'], net.edata['label'])
        logits_edge = net.edata['output_linear']
        path = torch.max(logits_edge, dim=1).indices
        edge_point = net.edata['point']

        return {'loss': loss,
                'path': path,
                'edge_point': edge_point}
