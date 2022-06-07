# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class DiGCN_SAT(nn.Module):

    def __init__(self, in_feats, out_feats, adj):
        super(DiGCN_SAT, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.adj = adj
        self.GCN_init_weights = nn.Parameter(torch.Tensor(self.in_feats, self.out_feats))

    def forward(self, x, spatial_attention):  # x: (batch_size, N, F_in, T), spatial_attention: (b, n, n)
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            adj_with_at = self.adj.mul(spatial_attention)  # (b, n, n)
            output = adj_with_at.permute(0, 2, 1).matmul(
                graph_signal.matmul(self.GCN_init_weights))  # output: (b, n, F_out)
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):

    def __init__(self, DEVICE, in_feats, num_of_vertices, num_time_steps):
        super(Temporal_Attention_layer, self).__init__()
        self.in_feats = in_feats
        self.num_of_vertices = num_of_vertices
        self.num_time_steps = num_time_steps
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices)).to(DEVICE)
        self.W_q = nn.Parameter(torch.FloatTensor(in_feats, in_feats)).to(DEVICE)
        self.W_k = nn.Parameter(torch.FloatTensor(in_feats, in_feats)).to(DEVICE)
        self.W_v = nn.Parameter(torch.FloatTensor(in_feats, in_feats)).to(DEVICE)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        x_t = torch.matmul(x.permute(0, 3, 2, 1), self.U1)  # (B, T, F_in, N)(N) -> (B,T,F_in)

        Q = torch.matmul(x_t, self.W_q)  # (b, t, F_in)
        K = torch.matmul(x_t, self.W_k)
        V = torch.matmul(x_t, self.W_v)  # (b, t, F_in)

        attention_score = torch.matmul(Q, K.permute(0, 2, 1)) / (self.num_time_steps ** 0.5)
        attention_score = torch.softmax(attention_score, dim=-1)  # (b, t, t)

        return attention_score


class ASTGCN_block(nn.Module):

    def __init__(self, DEVICE, in_feats, out_feats, adj, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.SAt = Spatial_Attention_layer(DEVICE, in_feats, num_of_vertices, num_of_timesteps)
        self.gcn_layer = DiGCN_SAT(in_feats, out_feats, adj)
        # self.gcn_layer2 = DiGCN_SAT(hid_feats, out_feats, adj)
        self.TAt = Temporal_Attention_layer(DEVICE, out_feats, num_of_vertices, num_of_timesteps)
        self.time_conv = nn.ModuleDict(
            {'filter': nn.Conv2d(out_feats, out_feats, kernel_size=(1, 3), stride=(1, 1),padding=(0, 1)),
             'gate': nn.Conv1d(out_feats, out_feats, kernel_size=(1, 3), padding=(0, 1)),
             'residual': nn.Conv1d(in_feats, out_feats, kernel_size=(1, 1), stride=1)})
        self.layer_norm = nn.LayerNorm(out_feats)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # SAt
        spatial_At = self.SAt(x)
        # gcn
        spatial_gcn = self.gcn_layer(x, spatial_At)  # (b,N,F_out,T)

        # TAt
        temporal_At = self.TAt(spatial_gcn)  # (b, T, T)
        x_TAt = torch.matmul(spatial_gcn.reshape(batch_size, -1, num_of_timesteps), temporal_At)\
            .reshape(batch_size, num_of_vertices, -1, num_of_timesteps)  # (b, n, f_out, t)

        # convolution along the time axis
        filter = self.time_conv['filter'](x_TAt.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        filter = torch.tanh(filter)
        gate = self.time_conv['gate'](x_TAt.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        gate = torch.sigmoid(gate)
        temporal_conv = filter * gate

        # residual shortcut
        x_residual = self.time_conv['residual'](x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        x_residual = self.layer_norm(F.relu(x_residual + temporal_conv).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        return x_residual


class ASTGCN_submodule(nn.Module):

    def __init__(self, DEVICE, args_layers, adj, num_for_predict, len_input, num_of_vertices):
        super(ASTGCN_submodule, self).__init__()
        self.args_layers = args_layers
        self.adj = adj
        self.BlockList = nn.ModuleList([ASTGCN_block(DEVICE, args_layers['layer1'][0], args_layers['layer1'][1], adj,
                                                     num_of_vertices, len_input)])
        self.BlockList.extend([ASTGCN_block(DEVICE, args_layers['layer2'][0], args_layers['layer2'][1], adj,
                                                     num_of_vertices, len_input)])
        self.final_conv = nn.Conv2d(len_input, num_for_predict, kernel_size=(1, args_layers['layer2'][1]))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


def make_model(DEVICE, args_layers, adj_mx, num_for_predict, len_input, num_of_vertices):

    model = ASTGCN_submodule(DEVICE, args_layers, adj_mx, num_for_predict, len_input, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model



















