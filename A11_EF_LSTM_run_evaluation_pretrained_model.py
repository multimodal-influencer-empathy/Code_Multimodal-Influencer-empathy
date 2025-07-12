import torch
from utils import  *
from config.config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from module_test import do_test

def test(model_name, dataset_name, featurePath):
    # Set a fixed seed
    set_seed(42)
    # load args
    config_file = r"config/config_pretrained.json"
    args = get_config_regression(model_name, dataset_name, config_file)
    args["featurePath"] = featurePath
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    dataloader = MMDataLoader(args, 1)
    model = AMIO(args).to(args['device'])

    model_path = r"pretrained_model/A11_EF_LSTM_pretrained_model.pth"
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(args['device'])

    do_test(args, model_name, dataset_name, model, dataloader)


if __name__ == '__main__':
    dataset_name = "Empathy"
    featurePath = r"data/empathy_test_features.pkl"

    model_name = 'A11_EF_LSTM'
    test(model_name, dataset_name, featurePath)


