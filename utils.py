import json
import torch
import numpy as np

column_names = ["HeadPitch","HeadYaw","HipPitch","HipRoll","KneePitch","LElbowRoll","LElbowYaw","LHand","LShoulderPitch","LShoulderRoll","LWristYaw","RElbowRoll",
                "RElbowYaw","RHand","RShoulderPitch","RShoulderRoll","RWristYaw"]

def load_params(json_filename):
    with open(json_filename, 'r') as f:
        params = json.load(f)
    return params

def save_model(params, model, optimizer, epoch, loss):
    filename_model = './result/model.tar'
    torch.save({
        'params': params,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename_model)




def load_model(model, optimizer):
    filename_model = './result/model.tar'
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(filename_model, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    params = checkpoint['params']
    return model, optimizer, epoch, loss, params


def set_random_seed(params):
    random_seed = params.get("random_seed")
    if(random_seed != -1):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

