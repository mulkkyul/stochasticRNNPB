import torch
import torch.nn as nn

class StochasticRNNPB(nn.Module):
    def __init__(self, params, device):
        super(StochasticRNNPB, self).__init__()

        self.model_behavior = params["model"]["behavior"]
        self.num_layers = params["model"]["num_layers"]
        self.data_dim = params["model"]["data_dim"]
        self.pb_size = params["model"]["pb_size"]
        self.output_dim = self.data_dim
        self.hidden_size = params["model"]["hidden_size"]
        self.num_sequence = params["model"]["num_pb"]
        self.device = device

        self.lstm = nn.LSTM(self.data_dim + self.pb_size, self.hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.output_dim)
        self.mu_pb = nn.Parameter(torch.zeros(self.num_sequence, self.pb_size))
        self.logvar_pb = nn.Parameter(torch.zeros(self.num_sequence, self.pb_size))


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def compute_kld(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    def forward(self, sequence_label, forward_computation_length):
        h_t = torch.zeros(self.num_layers, len(sequence_label), self.hidden_size, device=self.device)
        c_t = torch.zeros(self.num_layers, len(sequence_label), self.hidden_size, device=self.device)


        mu_pb = self.mu_pb[sequence_label]
        logvar_pb = self.logvar_pb[sequence_label]
        if(self.model_behavior == "deterministic"):
            pb = self.mu_pb[sequence_label]
        elif(self.model_behavior == "stochastic"):
            pb = self.reparameterize(mu_pb, logvar_pb)

        pb = pb.unsqueeze(0)

        outputs = []
        for t in range(forward_computation_length):
            if(t == 0):
                input_t = torch.zeros(1, len(sequence_label), self.data_dim, device=self.device)
            else:
                input_t = outputs[-1]
            input_t = torch.cat((input_t, pb), 2)
            output, (h_t, c_t) = self.lstm(input_t, (h_t, c_t))
            output = self.linear(output).to(self.device)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return outputs, sequence_label, pb.squeeze(0), mu_pb, logvar_pb

    def forward_given_pb(self, mu_pb, logvar_pb, forward_computation_length=100):
        h_t = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        c_t = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)

        if (self.model_behavior == "deterministic"):
            pb = mu_pb
        elif (self.model_behavior == "stochastic"):
            pb = self.reparameterize(mu_pb, logvar_pb)
        pb = pb.unsqueeze(0).unsqueeze(0)

        outputs = []
        for t in range(forward_computation_length):
            if (t == 0):
                input_t = torch.zeros(1, 1, self.data_dim, device=self.device)
            else:
                input_t = outputs[-1]

            input_t = torch.cat((input_t, pb), 2)
            output, (h_t, c_t) = self.lstm(input_t, (h_t, c_t))
            output = self.linear(output).to(self.device)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return outputs, pb.squeeze(0)

