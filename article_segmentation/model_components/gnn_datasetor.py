import dgl
import pickle
from dgl.data import DGLDataset

class gnn_dataset(DGLDataset):
    def __init__(self,
                 model_name='dbmdz/bert-base-historic-multilingual-cased',
                 device='cuda:1',
                 max_token_num=512,
                 file_path='../../seperator_detection/result/',
                 lang='fi'):
        super(gnn_dataset, self).__init__()

        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass