import myDataloader
import models
import utils
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

def plot_pca_on_pb(df_result):
    color_discrete_sequence = (px.colors.qualitative.Alphabet
                               + px.colors.qualitative.Dark24
                               + px.colors.qualitative.Light24
                               + px.colors.qualitative.Prism
                               + px.colors.qualitative.Safe
                               + px.colors.qualitative.Vivid)

    fig = px.scatter(df_result, x="PC1", y="PC2", color="idx_seq", color_discrete_sequence=color_discrete_sequence)
    fig.update_traces(marker=dict(showscale=False))
    img_filename = "./figure_PCA_on_learned_PBs.png"
    fig.write_image(img_filename, format="png", width=1024, height=768, scale=3)



def visualize_pb(model, params, num_samples_per_sequence):
    N = model.mu_pb.shape[0]
    D_PB = model.mu_pb.shape[1]
    list_result = []
    for idx_seq in range(N):
        for idx_sample in range(num_samples_per_sequence):
            if(params["model"]["behavior"] == "stochastic"):
                pb = model.reparameterize(model.mu_pb[idx_seq], model.logvar_pb[idx_seq]).detach().cpu().tolist()
            else:
                pb = model.mu_pb[idx_seq].detach().cpu().tolist()
            row = ["SEQ_%d"%idx_seq, idx_sample] + pb
            list_result.append(row)
    columns_pb = ["PB_%x" % x for x in range(D_PB)]
    df_result = pd.DataFrame(list_result, columns=["idx_seq", "idx_sample"] + columns_pb)
    X = df_result[columns_pb].copy()
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    df_principal = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    df_result = pd.concat([df_result, df_principal], axis=1)
    plot_pca_on_pb(df_result)



def generate_sequence_given_mu_logvar(model, mu_pb, logvar_pb, forward_computation_length):
    model.eval()
    with torch.no_grad():
        outputs, pbs = model.forward_given_pb(mu_pb.to(device), logvar_pb.to(device), forward_computation_length)
        df_output = pd.DataFrame(outputs.squeeze(1).cpu(), columns=utils.column_names)
        print(df_output)
        print("Given mu: ", mu_pb.cpu().tolist())
        print("Given logvar: ", logvar_pb.cpu().tolist())
        print("Sampled PB:", pbs.cpu().tolist()[0])

def generate_training_sequences(model):
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
            mse = np.mean((df_target - df_output) ** 2)
            print("idx Seq:", labels.tolist()[0],
                  "mu: ", ["%.2f" % x for x in mus.tolist()[0]],
                  "var: ", ["%.2f" % np.exp(x) for x in logvars.tolist()[0]],
                  "mse: %.6f" % mse)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = utils.load_params("./params.json")

    model = models.StochasticRNNPB(params, device).to(device)
    model, _, _, _, params_saved = utils.load_model(model, None)

    # ==========================================================================
    # Example 0. Visualize the learned PB values using PCA
    # This conducts PCA on the PB values sampled from the learned mu and logvar.
    # ==========================================================================
    # visualize_pb(model, params_saved, num_samples_per_sequence=100)


    # ==========================================================================
    # Example 1. Generate the sequence from the specific mu and logvar
    # This generates a sequence from the PB sampled from the given mu and logvar
    # ==========================================================================
    # mu_pb = torch.zeros(params["model"]["pb_size"], dtype=torch.float32).to(device)
    # logvar_pb = torch.zeros(params["model"]["pb_size"], dtype=torch.float32).to(device) # unit variance
    # #logvar_pb = -999999999.0 * torch.ones(params["model"]["pb_size"], dtype=torch.float32).to(device) # (nearly) zero variance
    # generate_sequence_given_mu_logvar(model, mu_pb, logvar_pb, forward_computation_length=100) # specify the desired output length



    # ==========================================================================
    # Example 2. Reconstruct the training data from the learned mu and logvar
    # This generates sequences from the PB sampled from the learned mu and logvar
    # Note. Since the model is not keeping the forward_computation_length for each seq, we load the dataset to get them.
    # ==========================================================================
    generate_training_sequences(model)
