import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os.path as osp
import torch
from scipy.sparse.linalg import eigs
import torch.nn.functional as F
from dataset import mydataset
from tqdm import tqdm

def loadData(args):

    print("loading the input file(s)")
    # pdb.set_trace()
    A = pd.read_csv(osp.join(args.root, args.adj),index_col=0)
    index = A.index
    A = A.astype(np.float32)
    A = np.array(A)

    #A = torch.from_numpy(np.array(A))

    df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col = 0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col = 0)
    num_input = df.shape[0]
    train_input = round(0.7 * num_input)
    test_input = round(0.2 * num_input)
    val_input = round(0.1 * num_input)
    time = set_time(df.index.values,df)

    dis = set_dis(df_dis)
    con = Construction_Layer(torch.from_numpy(np.array(df_con)),torch.from_numpy(np.array(dis)))
    dataframe = np.stack([np.array(df),con.numpy(),time.numpy()])

    #all = dataframe[0,:,:]
    mean, std = np.mean(df), np.std(df)
    X = normalize(dataframe, mean[0], std[0])
    train = X[:,:train_input,:]
    val = X[:,train_input: train_input + val_input,:]
    test = X[:,train_input + val_input:, :]


    train_ = data_generate(train, args, df[:train_input], 'train.pt')
    val_  = data_generate(val, args, df[train_input: train_input + val_input], 'val.pt')
    test_ = data_generate(test, args, df[train_input + val_input:], 'test.pt')

    train = mydataset(train_[0],train_[1])
    val = mydataset(val_[0], val_[1])
    test = mydataset(test_[0], test_[1])
    test_time = mydataset(test_[2], test_[3])
    return train,val,test, test_time,  A, mean, std , index

def data_generate(dataset,args, df, save_file):

    path = osp.join('data/processed',save_file)
    if osp.exists(path) is not True:
        time_interval = 300
        dataset = dataset.transpose(1,2,0)
        num_input, num_node, dims = dataset.shape
        num_sample = num_input - args.T - args.P + 1
        index = pd.DataFrame(df.index.values, columns=['time'])
        index.time = pd.to_datetime(index.time)
        # dataset = np.transpose(self.dataset, (1, 2, 0))
        n = 0
        x_set = []
        y_set = []
        x_time_set = []
        y_time_set = []
        for i in tqdm(range(num_sample)):
            flag = 0
            x = np.zeros(shape=(args.T, num_node, dims))
            y = np.zeros(shape=(args.P, num_node, dims))
            x_time = np.empty(shape=(args.T), dtype=object)
            y_time = np.empty(shape=(args.P), dtype=object)
            x[0] = dataset[i, :, :]
            x_time[0] = index['time'].iloc[i]
            for j in range(1, args.T + args.P):
                if j <= (args.T - 1):
                    if (index['time'].iloc[i + j] - index['time'].iloc[i + j - 1]).seconds == time_interval:
                        x[j] = dataset[i + j, :, :]
                        x_time[j] = index['time'].iloc[i+j]
                    else:
                        flag = 1
                        break

                else:
                    if (index['time'].iloc[i + j] - index['time'].iloc[i + j - 1]).seconds == time_interval:
                        y[j - args.T] = dataset[i + j, :, :]
                        y_time[j - args.T] = index['time'].iloc[i+j]
                    else:
                        flag = 1
                        break
            if flag == 0:
                n = n + 1
                x = x.astype(np.float32)
                y = y.astype(np.float32)
                x_set.append(x)
                y_set.append(y)
                x_time_set.append(x_time)
                y_time_set.append(y_time)

        l = (len(x_set) // args.batch_size) * args.batch_size
        input = x_set[0:l]
        output = y_set[0:l]
        x_t = x_time_set[0:l]
        y_t = y_time_set[0:l]
        torch.save((input, output,x_t,y_t), path)
        data = torch.load(path)
    else:
        data = torch.load(path)

    return data


def set_dis(df):
    x = 1 - (df / 3)**2
    x[x == 1] = 0
    x[x < 0] = 0
    x.values[tuple([np.arange(x.shape[0])[:, np.newaxis]]) * 2] = 1
    return x

def Construction_Layer(x, adj):
    lens, num_of_vertices = x.shape
    set = torch.zeros(size = (lens, num_of_vertices), dtype=torch.float64)
    for i in range(lens):
        a = (x[i] == 1).nonzero()
        for j in range(len(a)):
            e = adj[a[j]]
            set[i] = torch.max(set[i],e)
    return set

def set_time(x,df):
    time_pd = pd.DataFrame(x, columns = ['Times'])
    time_pd['stamp'] = [pd.Timestamp(time) for time in time_pd['Times'].values]
    time_pd['year'] = time_pd['stamp'].apply(lambda x: x.year)
    time_pd['month'] = time_pd['stamp'].apply(lambda x: x.month)
    time_pd['day'] = time_pd['stamp'].apply(lambda x: x.day)
    time_pd['DayOfWeek'] = time_pd['stamp'].apply(lambda d: d.dayofweek)
    time_pd['DayOfYear'] = time_pd['stamp'].apply(lambda d: d.dayofyear)
    time_pd['WeekOfYear'] = time_pd['stamp'].apply(lambda d: d.weekofyear)
    time_pd['Quarter'] = time_pd['stamp'].apply(lambda d: d.quarter)
    time_pd['Hour'] = time_pd['stamp'].apply(lambda d: d.hour)
    time_pd['Minute'] = time_pd['stamp'].apply(lambda d: d.minute)
    cut_hour = [-1, 5, 11, 14, 17, 21,  23]
    cut_labels = ['last night', 'morning', 'Noon', 'afternoon','evening', 'Night']
    time_pd['Hour_cut'] = pd.cut(time_pd['Hour'], bins=cut_hour, labels=cut_labels)
    time = LabelEncoder().fit_transform(time_pd['Hour_cut'])
    set = torch.zeros(size = df.shape, dtype=torch.long)
    for i in range(len(time)):
        set[i] = time[i]
    return set

def normalize(x,mean,std):
    x[0,:,:] = (x[0, :, :] - mean) / std
    return x

def re_normalization(x, mean, std):
    x = x * std + mean
    return x

def RMSE(input, target, min, max, m_factor):
    rmse = torch.sqrt(F.mse_loss(input, target)) * (max - min) / 2. * m_factor
    return rmse

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        # mask = np.not_equal(label, 0)
        # mask = mask.astype(np.float32)
        # mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse)
        result = np.mean(rmse, axis=0)
        b = np.sqrt(np.mean(result, axis = 0))
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape)
        mape = np.mean(mape)

    return mae, rmse, mape, b

def scaled_Laplacian_1(A):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert A.shape[0] == A.shape[1]

    D = np.diag(np.sum(A, axis=1))

    return D, A, np.identity(A.shape[0])

def A_wave(args, A):
    assert A.shape[0] == A.shape[1]
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return torch.from_numpy(A_wave).type(torch.FloatTensor).to(args.device)

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials