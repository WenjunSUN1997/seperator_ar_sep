import torch
from transformers import AutoModel

class SepLinear(torch.nn.Module):
    def __init__(self, config):
        super(SepLinear, self).__init__()
        self.center_flag = config['flag_center']
        self.region_flag = config['region_flag']
        self.encoder_flag = config['encoder_flag']
        self.model = AutoModel.from_pretrained(config['model_name'])
        self.loss_func = torch.nn.CosineEmbeddingLoss()
        self.linear = torch.nn.Linear(in_features=self.model.config.hidden_size,
                                      out_features=2)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model.config.hidden_size,
                                                              nhead=2,
                                                              batch_first=True)
        self.encoder_layer = torch.nn.TransformerEncoder(self.all_encoder_layer,
                                                         num_layers=2)
        self.x_embedding = torch.nn.Embedding(num_embeddings=500,
                                              embedding_dim=self.model.config.hidden_size)
        self.y_embedding = torch.nn.Embedding(num_embeddings=500,
                                              embedding_dim=self.model.config.hidden_size)
        self.top_embedding = torch.nn.Embedding(num_embeddings=10,
                                                embedding_dim=self.model.config.hidden_size)
        self.bottom_embedding = torch.nn.Embedding(num_embeddings=10,
                                                   embedding_dim=self.model.config.hidden_size)
        self.left_embedding = torch.nn.Embedding(num_embeddings=10,
                                                 embedding_dim=self.model.config.hidden_size)
        self.right_embedding = torch.nn.Embedding(num_embeddings=10,
                                                  embedding_dim=self.model.config.hidden_size)

    def get_semantic_embedding(self, data):
        pass

    def forward(self, data):
        pass