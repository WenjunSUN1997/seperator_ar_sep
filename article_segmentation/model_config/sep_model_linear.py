import torch
from transformers import AutoModel

class SepLinear(torch.nn.Module):
    def __init__(self, config):
        super(SepLinear, self).__init__()
        self.center_flag = config['center_flag']
        self.region_flag = config['region_flag']
        self.encoder_flag = config['encoder_flag']
        self.model = AutoModel.from_pretrained(config['model_name'])
        self.loss_func = torch.nn.CosineEmbeddingLoss()
        self.drop_out = torch.nn.Dropout(p=config['drop_out'])
        self.activate = torch.nn.ReLU()
        self.normal = torch.nn.LayerNorm(self.model.config.hidden_size)
        self.linear = torch.nn.Linear(in_features=self.model.config.hidden_size*2,
                                      out_features=2)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model.config.hidden_size,
                                                              nhead=2,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)
        self.x_embedding = torch.nn.Embedding(num_embeddings=501,
                                              embedding_dim=self.model.config.hidden_size)
        self.y_embedding = torch.nn.Embedding(num_embeddings=501,
                                              embedding_dim=self.model.config.hidden_size)
        self.top_embedding = torch.nn.Embedding(num_embeddings=10,
                                                embedding_dim=self.model.config.hidden_size)
        self.bottom_embedding = torch.nn.Embedding(num_embeddings=10,
                                                   embedding_dim=self.model.config.hidden_size)
        self.left_embedding = torch.nn.Embedding(num_embeddings=10,
                                                 embedding_dim=self.model.config.hidden_size)
        self.right_embedding = torch.nn.Embedding(num_embeddings=10,
                                                  embedding_dim=self.model.config.hidden_size)

    def __forward(self, data, index):
        semantic = self.model(input_ids=data['input_ids_'+str(index)].squeeze(1),
                              attention_mask=data['attention_mask_'+str(index)].squeeze(1))['last_hidden_state']
        semantic = self.normal(semantic)
        x_embedding = self.normal(self.x_embedding(data['x_'+str(index)]))
        y_embedding = self.normal(self.y_embedding(data['y_'+str(index)]))
        left_embedding = self.normal(self.left_embedding(data['left_' + str(index)]))
        right_embedding = self.normal(self.right_embedding(data['right_' + str(index)]))
        top_embedding = self.normal(self.top_embedding(data['top_' + str(index)]))
        bottom_embedding = self.normal(self.bottom_embedding(data['bottom_' + str(index)]))
        if self.encoder_flag:
            semantic = self.normal(torch.mean(self.encoder(semantic), dim=1))



    def forward(self, data):
        first_semantic = self.__forward(data, index=0)
        second_semantic = self.__forward(data, index=1)