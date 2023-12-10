from torch.utils.data.dataloader import DataLoader
from article_segmentation.model_components.datasetor import SepDateset
import pickle

def get_dataloader(input):
    train_path = '../seperator_detection/result/'
    dev_path = '../seperator_detection/result/'
    test_path = '../seperator_detection/result/'
    train_dataset = SepDateset(model_name=input['model_name'],
                               device=input['device'],
                               max_token_num=input['max_token_num'],
                               file_path=train_path)
    # kwargs['file_path'] = dev_path
    # dev_dataset = SepDateset(kwargs)
    # kwargs['file_path'] = test_path
    # test_dataset = SepDateset(kwargs)
    dataloader_obj = DataLoader(train_dataset,
                                batch_size=input['batch_size'],
                                shuffle=False)
    return dataloader_obj

