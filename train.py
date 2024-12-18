import myDataloader
import models
import utils
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)

def train_model(params, device, data_loader):
    model = models.StochasticRNNPB(params, device).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = Adam(params=model.parameters(), lr=params["train"]["learning_rate"])
    beta = params["model"]["beta"]

    for epoch in tqdm(range(params["train"]["num_epochs"])):
        for index, (sequence_data, sequence_label, sequence_length) in enumerate(data_loader):
            forward_computation_length = sequence_data.shape[0] # the longest one in the batch
            outputs, labels, pbs, mus, logvars = model.forward(sequence_label, forward_computation_length)

            loss = criterion(outputs, sequence_data)
            loss = loss.sum(-1)
            mask = (torch.arange(outputs.size(0)).unsqueeze(1) < sequence_length.unsqueeze(0)).to(device)
            masked_loss = loss * mask.float()
            loss_reconstruction = masked_loss.sum() / mask.float().sum()

            loss_kld = model.compute_kld(model.mu_pb, model.logvar_pb)
            loss_total = loss_reconstruction + beta * loss_kld
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        if (epoch % 10 == 0):
            loss_total = loss_total.detach().cpu().numpy()
            loss_reconstruction = loss_reconstruction.detach().cpu().numpy()
            loss_kld = loss_kld.detach().cpu().numpy()

            print("Epoch: %d\tTotal Loss: %.6f\tReconstruction: %.6f\tKLD: %.6f" % (epoch, loss_total, loss_reconstruction, loss_kld))


    # END of training
    utils.save_model(params, model, optimizer, epoch, loss_total)
    return model, optimizer, params


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.load_params("./params.json")
    utils.set_random_seed(params)

    dataset_train, dataset_filename = myDataloader.load_dataset(params["path_dataset_train"], device)
    data_loader = DataLoader(dataset=dataset_train, batch_size=params["train"]["batch_size"],
                             shuffle=False, collate_fn=myDataloader.pad_collate)

    model, optimizer, params = train_model(params, device, data_loader)
