import myDataloader
import models
import utils
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
torch.set_printoptions(sci_mode=False)


def get_dataset(params, device):
    path_dataset = params["path_dataset_recognition"]
    dataset_train, dataset_filename = myDataloader.load_dataset(path_dataset, device)
    data_loader = DataLoader(dataset=dataset_train, batch_size=1,
                             shuffle=False, collate_fn=myDataloader.pad_collate)

    return data_loader, dataset_filename, params


def run_recognition(params, model, device, target_sequence, presearch="zero",
                    number_of_iteration=100, observed_ratio=0.8, lr_mu=0.1, lr_logvar=0.1):

    length_observed = int(target_sequence.shape[0] * observed_ratio)
    for name, param in model.named_parameters():
        if name in ['mu_pb', 'logvar_pb']:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = Adam([
        {'params': model.mu_pb, 'lr': lr_mu},
        {'params': model.logvar_pb, 'lr': lr_logvar}
        ])



    # ===================================================================
    # Pre-search (warm start). finding initial mu values
    target_sequence = target_sequence.to(device)
    observed_sequence = target_sequence[0:length_observed, :, :].to(device)
    mse_loss = nn.MSELoss()
    if(presearch == "learned"):
        lowest_loss = float('inf')
        most_similar_pb_mu = None
        for mu_pb, logvar_pb in zip(model.mu_pb, model.logvar_pb):
            output_sequence, _ = model.forward_given_pb(mu_pb, logvar_pb, length_observed)
            loss = mse_loss(output_sequence[0:length_observed, :, :], observed_sequence)
            if loss < lowest_loss:
                lowest_loss = loss
                most_similar_pb_mu = mu_pb
        pb_init_mu = most_similar_pb_mu
    elif ("rand" in presearch):
        lowest_loss = float('inf')
        most_similar_pb_mu = None
        N = int(presearch[4:])
        D = len(model.mu_pb[0])
        list_mu = 3.0 * torch.randn(N, D)
        list_logvar = -10000.0 * torch.ones(N, D)  # to make variance zero
        for mu_pb, logvar_pb in zip(list_mu, list_logvar):
            output_sequence, _ = model.forward_given_pb(mu_pb.to(device), logvar_pb.to(device), length_observed)
            loss = mse_loss(output_sequence[0:length_observed, :, :], observed_sequence)
            if loss < lowest_loss:
                lowest_loss = loss
                most_similar_pb_mu = mu_pb
        pb_init_mu = most_similar_pb_mu
    elif (presearch == "zero"):
        pb_init_mu = torch.zeros(1, params["model"]["pb_size"]).to(device)


    # unit variance assumption
    pb_init_logvar = torch.zeros(1, params["model"]["pb_size"]).to(device)

    # ====================================================================================
    sequence_label = 0 # Use the first one during recognition
    model.mu_pb.data[sequence_label] = pb_init_mu
    model.logvar_pb.data[sequence_label] = pb_init_logvar

    min_recons_loss = 0.1
    for epoch in range(number_of_iteration):
        forward_computation_length = target_sequence.shape[0]
        outputs, labels, pbs, mus, logvars = model.forward([sequence_label], forward_computation_length)
        optimizer.zero_grad()
        loss_observation = mse_loss(outputs[0:length_observed, :, :], observed_sequence)
        loss_prediction = mse_loss(outputs[length_observed:, :, :], target_sequence[length_observed:, :, :])

        if(loss_observation.detach().item() < min_recons_loss):
            min_recons_loss = loss_observation.detach().item()
            new_mu = torch.tensor(pbs.detach().tolist()[0], device=device)
            model.mu_pb.data[sequence_label] = new_mu
        else:
            loss_observation.backward()
            optimizer.step()


    df_output = pd.DataFrame(outputs.squeeze(1).cpu().detach(), columns=utils.column_names)
    df_target = pd.DataFrame(target_sequence.squeeze(1).cpu().detach(), columns=utils.column_names)

    return df_target, df_output



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.load_params("./params.json")

    model = models.StochasticLSTMPB(params, device).to(device)
    optimizer = Adam(params=model.parameters(), lr=params["train"]["learning_rate"])
    model, _, _, _, _ = utils.load_model(model, None)
    data_loader, dataset_filename, params = get_dataset(params, device)

    # ==========================================================================
    presearch = "learned" #"zero", "rand10", "rand100", "learned"
    number_of_iteration = 100
    observed_ratio = 0.8
    lr_mu = 0.1
    lr_logvar = 0.1

    for indexSeq, (sequence_data, sequence_label, sequence_length) in enumerate(data_loader):
        model, _, _, _, _ = utils.load_model(model, optimizer)

        df_target, df_output = run_recognition(params, model, device,
                                                       sequence_data,
                                                       presearch=presearch,
                                                       number_of_iteration=number_of_iteration,
                                                       observed_ratio=observed_ratio,
                                                       lr_mu=lr_mu,
                                                       lr_logvar=lr_logvar)
