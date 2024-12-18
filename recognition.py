import myDataloader
import models
import utils
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)


def get_dataset(params, device):
    path_dataset = params["path_dataset_recognition"]
    dataset_train, dataset_filename = myDataloader.load_dataset(path_dataset, device)
    data_loader = DataLoader(dataset=dataset_train, batch_size=1,
                             shuffle=False, collate_fn=myDataloader.pad_collate)

    return data_loader, dataset_filename, params


def run_recognition(params, model, device, target_sequence, presearch="zero",
                    number_of_iteration=100, observed_ratio=0.8, lr_mu=0.1, lr_logvar=0.1):



    # During recognition, we only update mu & logvar while other learnable parameters are fixed.
    for name, param in model.named_parameters():
        if name in ['mu_pb', 'logvar_pb']:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Set the learning rate for mu and logvar.
    # Empirically, higher learning rate showed the better performance in recognition.
    optimizer = Adam([
        {'params': model.mu_pb, 'lr': lr_mu},
        {'params': model.logvar_pb, 'lr': lr_logvar}
        ])

    # Assume the model only observed the first (observed_ratio)% of the sequence.
    # This is to test both (1) reconstructing (observed_ratio)% and (2) forecasting remaining (1-observed_ratio)%
    target_sequence = target_sequence.to(device)
    length_observed = int(target_sequence.shape[0] * observed_ratio)
    observed_sequence = target_sequence[0:length_observed, :, :].to(device)

    # ===================================================================
    # Pre-search (warm start). finding the initial mu values
    # zero: Initialize PB as Unit Gaussian (mu=0, logvar=0)
    # learned: mu = choose one of learned mu values (logvar = 0)
    # randN: mu = choose one of N random mu values (logvar = 0)
    # ===================================================================
    mse_loss = nn.MSELoss()
    if(presearch == "learned"):
        lowest_loss = float('inf')
        most_similar_pb_mu = None
        N = int(model.mu_pb.shape[0])
        D_PB = int(model.mu_pb.shape[1])
        list_logvar = -10000.0 * torch.ones(N, D_PB)  # to make variance nearly zero
        for mu_pb, logvar_pb in zip(model.mu_pb, list_logvar):
            output_sequence, _ = model.forward_given_pb(mu_pb, logvar_pb.to(device), length_observed)
            loss = mse_loss(output_sequence[0:length_observed, :, :], observed_sequence)
            if loss < lowest_loss:
                lowest_loss = loss
                most_similar_pb_mu = mu_pb
        pb_init_mu = most_similar_pb_mu
    elif ("rand" in presearch):
        lowest_loss = float('inf')
        most_similar_pb_mu = None
        N = int(presearch[4:])
        D_PB = int(model.mu_pb.shape[1])
        list_mu = 3.0 * torch.randn(N, D_PB)
        list_logvar = -10000.0 * torch.ones(N, D_PB)  # to make variance nearly zero
        for mu_pb, logvar_pb in zip(list_mu, list_logvar):
            output_sequence, _ = model.forward_given_pb(mu_pb.to(device), logvar_pb.to(device), length_observed)
            loss = mse_loss(output_sequence[0:length_observed, :, :], observed_sequence)
            if loss < lowest_loss:
                lowest_loss = loss
                most_similar_pb_mu = mu_pb
        pb_init_mu = most_similar_pb_mu
    elif (presearch == "zero"):
        pb_init_mu = torch.zeros(1, params["model"]["pb_size"]).to(device)
    else:
        print("Error. Specify the warm start method.")
        exit()

    
    # unit variance assumption
    pb_init_logvar = torch.zeros(1, params["model"]["pb_size"]).to(device)

    # ====================================================================================
    sequence_label = 0 # Use the first one as a placeholder during recognition
    model.mu_pb.data[sequence_label] = pb_init_mu
    model.logvar_pb.data[sequence_label] = pb_init_logvar

    min_recons_loss = 0.1
    for _ in tqdm(range(number_of_iteration)):
        forward_computation_length = target_sequence.shape[0]
        outputs, labels, pbs, mus, logvars = model.forward([sequence_label], forward_computation_length)
        optimizer.zero_grad()
        reconstruction_loss = mse_loss(outputs[0:length_observed, :, :], observed_sequence)
        # prediction error is not used during iteration.
        #prediction_error = mse_loss(outputs[length_observed:, :, :], target_sequence[length_observed:, :, :])

        # an implementation of an early update for the reconstruction_loss less than the threshold
        # this early update manually set the mu_pb to the current value
        # empirically, we found this early update effectively leverages the advantages of the stochastic model
        if(reconstruction_loss.detach().item() < min_recons_loss):
            min_recons_loss = reconstruction_loss.detach().item()
            new_mu = torch.tensor(pbs.detach().tolist()[0], device=device)
            model.mu_pb.data[sequence_label] = new_mu
        else:
            reconstruction_loss.backward()
            optimizer.step()


    df_output = pd.DataFrame(outputs.squeeze(1).cpu().detach(), columns=utils.column_names)
    df_target = pd.DataFrame(target_sequence.squeeze(1).cpu().detach(), columns=utils.column_names)

    return df_target, df_output



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.load_params("./params.json")

    model = models.StochasticRNNPB(params, device).to(device)
    optimizer = Adam(params=model.parameters(), lr=params["train"]["learning_rate"])
    model, _, _, _, _ = utils.load_model(model, None)
    data_loader, dataset_filename, params = get_dataset(params, device)

    # ==========================================================================

    presearch = "rand100" # "zero", "rand10", "rand100", "learned"
    number_of_iteration = 100
    observed_ratio = 0.8 # 80% of the target sequence will be used for computing loss
    lr_mu = 0.1 # learning rate for mu
    lr_logvar = 0.1 # learning rate for logvar

    for indexSeq, (sequence_data, sequence_label, sequence_length) in enumerate(data_loader):
        # for each novel sequence, start the optimization with the newly loaded model
        model, _, _, _, _ = utils.load_model(model, optimizer)

        df_target, df_output = run_recognition(params, model, device,
                                                       sequence_data,
                                                       presearch=presearch,
                                                       number_of_iteration=number_of_iteration,
                                                       observed_ratio=observed_ratio,
                                                       lr_mu=lr_mu,
                                                       lr_logvar=lr_logvar)

        length_observed = int(len(df_target) * observed_ratio)
        output_reconstruction = df_output.iloc[:length_observed]
        target_reconstruction = df_target.iloc[:length_observed]

        output_forecast = df_output.iloc[length_observed:]
        target_forecast = df_target.iloc[length_observed:]

        reconstruction_loss = np.mean((target_reconstruction.values - output_reconstruction.values) ** 2)
        prediction_error = np.mean((target_forecast.values - output_forecast.values) ** 2)

        print("idx seq: ", indexSeq,
              "reconstruction loss: %.6f" % reconstruction_loss,
              "prediction error: %.6f" % prediction_error)
