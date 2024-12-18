import myDataloader
import models
import utils
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.load_params("./params.json")

    model = models.StochasticLSTMPB(params, device).to(device)
    model, _, _, _, _ = utils.load_model(model, None)


    # ==========================================================================
    if(True):  # Sample from learned mu and logvar
        path_dataset = params["path_dataset_train"]
        dataset_train, dataset_filename = myDataloader.load_dataset(path_dataset, device)
        data_loader = DataLoader(dataset=dataset_train, batch_size=1,
                                 shuffle=False, collate_fn=myDataloader.pad_collate)

        model.eval()
        with torch.no_grad():
            for index, (sequence_data, sequence_label, sequence_length) in enumerate(data_loader):
                forward_computation_length = sequence_data.shape[0]
                outputs, labels, pbs, mus, logvars = model.forward(sequence_label, forward_computation_length)
                df_output = pd.DataFrame(outputs.squeeze(1).cpu(), columns=utils.column_names)
                df_target = pd.DataFrame(sequence_data.squeeze(1).cpu(), columns=utils.column_names)
                rmse = np.sqrt(((df_target - df_output) ** 2).sum())
                print(rmse)



    # ==========================================================================
    if(True):  # Sample from the given mu and logvar
        model.eval()
        with torch.no_grad():
            forward_computation_length = 100
            mu_pb = torch.zeros(params["model"]["pb_size"], dtype=torch.float32).to(device)
            logvar_pb = torch.zeros(params["model"]["pb_size"], dtype=torch.float32).to(device)
            outputs, pbs = model.forward_given_pb(mu_pb.to(device), logvar_pb.to(device), forward_computation_length)
            df_output = pd.DataFrame(outputs.squeeze(1).cpu(), columns=utils.column_names)
            print(df_output)
