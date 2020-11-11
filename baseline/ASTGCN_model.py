import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import util

class Net(torch.nn.Module):
    def __init__(self,device, in_channels,K, nb_chev_filter, num_time_filter, T, P, num_nodes, time_strides,cheb_polynomials):
        super(Net, self).__init__()
        self.BlockList1 = GAT_block(device = device, in_channels = in_channels, K = K, nb_chev_filter = nb_chev_filter, num_time_filter =  num_time_filter,T =  T, num_nodes = num_nodes, time_strides = time_strides, cheb_polynomials = cheb_polynomials)
        self.BlockList2 = GAT_block(device = device, in_channels = 64, K = K, nb_chev_filter = nb_chev_filter, num_time_filter =  num_time_filter, T =  T, num_nodes = num_nodes, time_strides = time_strides, cheb_polynomials = cheb_polynomials)

        self.final_conv = nn.Conv2d(int(T / time_strides), P, kernel_size=(1, num_time_filter))

    def forward(self, x):
        '''
        :param x: (Batch_size, Num_nodes, F_in, T_in)
        :return: (B, Num_nodes, T_out)
        '''
        list = torch.split(x, 1, dim=3)

        x = list[0].permute(0,2,3,1)

        #x = x.permute(0,2,3,1)
        x = self.BlockList1(x)
        x = self.BlockList2(x)
        x = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 1, 2)

        return x

class GAT_block(nn.Module):
    def __init__(self,device, in_channels,K,nb_chev_filter, num_time_filter, T, num_nodes, time_strides,cheb_polynomials):
        super(GAT_block, self).__init__()
        self.TAL = Temporal_Attention_Layer(device,in_channels, num_nodes,T)
        self.SAL = Spatial_Attention_Layer(device,in_channels, num_nodes,T)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, num_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, num_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(num_time_filter)

    def forward(self,x):
        '''
        :param x: (Batch_size, Num_nodes, F_in, T)
        :return: (Batch_size, Num_nodes, nb_time_filter, T)
        '''

        batch_size, num_nodes, num_features, num_timesteps = x.shape

        # temporal attention
        temporal_attention = self.TAL(x) # (Batchsize,T, T)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_timesteps), temporal_attention).reshape(batch_size,
                                                                                               num_nodes,
                                                                                               num_features,
                                                                                               num_timesteps)
        spatial_attention = self.SAL(x_TAt) # (Batchsize,Num_nodes,Num_features,T)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_attention)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)


        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class Temporal_Attention_Layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_Layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (Batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        batch_size, num_nodes, num_features, num_timesteps = x.shape
        a = torch.matmul(x.permute(0, 3, 2, 1), self.U1)
        lhs = torch.matmul(a, self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized



class Spatial_Attention_Layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_Layer, self).__init__()
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

class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

def model(args, A):
    '''
    :param args.device
    :param args.in_channels
    :param K:
    :param args.nb_block
    :param nb_time_filter:
    :param args.S:
    :param args.P:
    :param args.num_nodes:
    :param time_strides
    :param edge_index
    :param edge_weight
    :return:
    '''
    K = 3
    nb_chev_filter = 64
    L_tilde = util.scaled_Laplacian(A)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in util.cheb_polynomial(L_tilde, K)]
    model = Net(args.device, args.in_channels,K, nb_chev_filter, 64, args.T, args.P, args.num_nodes, 1, cheb_polynomials)


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model