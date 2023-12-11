import torch
from transformers import AutoModel

class SepLinear(torch.nn.Module):
    def __init__(self, config, weight):
        super(SepLinear, self).__init__()
        self.center_flag = config['center_flag']
        self.region_flag = config['region_flag']
        self.encoder_flag = config['encoder_flag']
        self.model = AutoModel.from_pretrained(config['model_name'])
        if config['weight']:
            self.loss_func = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

        self.drop_out = torch.nn.Dropout(p=config['drop_out'])
        self.activate = torch.nn.ReLU()
        self.device = config['device']
        self.normal = torch.nn.LayerNorm(self.model.config.hidden_size)
        self.linear = torch.nn.Linear(in_features=self.model.config.hidden_size,
                                      out_features=2)
        self.cnn = torch.nn.Conv2d(in_channels=1,
                                   out_channels=1,
                                   kernel_size=(2, self.model.config.hidden_size),
                                   stride=1,
                                   padding=0,)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model.config.hidden_size,
                                                              nhead=2,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)
        self.all_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model.config.hidden_size,
                                                                  nhead=2,
                                                                  batch_first=True)
        self.all_encoder = torch.nn.TransformerEncoder(self.all_encoder_layer,
                                                   num_layers=2)
        self.x_embedding = torch.nn.Embedding(num_embeddings=501,
                                              embedding_dim=self.model.config.hidden_size)
        self.y_embedding = torch.nn.Embedding(num_embeddings=501,
                                              embedding_dim=self.model.config.hidden_size)
        self.top_embedding = torch.nn.Embedding(num_embeddings=4000,
                                                embedding_dim=self.model.config.hidden_size)
        self.bottom_embedding = torch.nn.Embedding(num_embeddings=4000,
                                                   embedding_dim=self.model.config.hidden_size)
        self.left_embedding = torch.nn.Embedding(num_embeddings=4000,
                                                 embedding_dim=self.model.config.hidden_size)
        self.right_embedding = torch.nn.Embedding(num_embeddings=4000,
                                                  embedding_dim=self.model.config.hidden_size)

    def __forward(self, data, index):
        semantic = self.model(input_ids=data['input_ids_'+str(index)].squeeze(1),
                              attention_mask=data['attention_mask_'+str(index)].squeeze(1))['last_hidden_state']
        x_embedding = self.x_embedding(data['x_'+str(index)])
        y_embedding = self.y_embedding(data['y_'+str(index)])
        left_embedding = self.left_embedding(data['left_' + str(index)])
        right_embedding = self.right_embedding(data['right_' + str(index)])
        top_embedding = self.top_embedding(data['top_' + str(index)])
        bottom_embedding = self.bottom_embedding(data['bottom_' + str(index)])
        if self.encoder_flag:
            semantic = torch.mean(self.encoder(semantic), dim=1)
        else:
            semantic = torch.mean(semantic, dim=1)

        if self.region_flag:
            semantic = semantic + left_embedding + right_embedding + top_embedding + bottom_embedding

        if self.center_flag:
            semantic = semantic + x_embedding + y_embedding

        return self.normal(semantic)

    def cnn_forward(self, data):
        first_semantic = self.__forward(data, index=0)
        second_semantic = self.__forward(data, index=1)
        temp = []
        for batch_size in range(first_semantic.shape[0]):
            pair_semantic = self.all_encoder(torch.cat((first_semantic[batch_size].unsqueeze(0),
                                                        second_semantic[batch_size].unsqueeze(0)),
                                                        dim=0))
            output_cnn = self.cnn(pair_semantic.unsqueeze(0))

        return

    def encoder_linear_forward(self, data):
        first_semantic = self.__forward(data, index=0)
        second_semantic = self.__forward(data, index=1)
        temp = []
        for batch_size in range(first_semantic.shape[0]):
            pair_semantic = self.all_encoder(torch.cat((first_semantic[batch_size].unsqueeze(0),
                                                        second_semantic[batch_size].unsqueeze(0)),
                                                        dim=0))
            temp.append(torch.mean(pair_semantic, dim=0))

        all_semantic = torch.stack(temp)
        output_linear = self.linear(self.activate(self.normal(all_semantic)))
        loss = self.loss_func(output_linear, data['label'])
        path = torch.max(output_linear, dim=1).indices
        return {'loss': loss,
                'path': path}

    def linear_forward(self, data):
        first_semantic = self.__forward(data, index=0)
        second_semantic = self.__forward(data, index=1)
        output_linear = self.linear(self.activate(self.normal(torch.abs(first_semantic-second_semantic))))
        loss = self.loss_func(output_linear, data['label'])
        path = torch.max(output_linear, dim=1).indices
        return {'loss': loss,
                'path': path}

    def forward(self, data):
        return self.encoder_linear_forward(data)
        # return self.linear_forward(data)
        # self.cnn_forward(data)
        # first_semantic = self.__forward(data, index=0)
        # second_semantic = self.__forward(data, index=1)
        # temp = []
        # for batch_size in range(first_semantic.shape[0]):
        #     pair_semantic = self.all_encoder(torch.cat((first_semantic[batch_size].unsqueeze(0),
        #                                                 second_semantic[batch_size].unsqueeze(0)),
        #                                                dim=0))
        #     temp.append(torch.mean(pair_semantic, dim=0))
        #
        # all_semantic = torch.stack(temp)
        # output_linear = self.linear(self.activate(all_semantic))
        # data['label'][data['label'] == 0] = -1
        # loss = self.loss_func(first_semantic, second_semantic, data['label'])
        # path = []
        # cos_sim = torch.cosine_similarity(first_semantic, second_semantic, dim=-1)
        # for x in cos_sim:
        #     if x > 0:
        #         path.append(1)
        #     else:
        #         path.append(0)
        # # path = torch.max(output_linear, dim=1).indices
        # return {'loss': loss,
        #         'path': path}
