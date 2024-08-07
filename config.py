import torch

model_list=['CNN_emnist','CNN_mnist','CNN_fashion_mnist','CNN_cifar10','ResNet18','ShuffleNet','MobileNet','LSTM']
dataset_list=['emnist','mnist','fashion_mnist','cifar10','cifar100','IMDB']
method_list=['Fedavg','moon','FedUV','FedDistill+','Fedprox','Fedproto','Fedrep','FedBABU','FedDW','FedDW(pFL)','Fedper','Per-Fedavg','FedALA','local_only']



# Fedavy   : 0
# moon     : 1
# FedUV    : 2
# FedDistill+ : 3
# Fedprox  : 4
# Fedproto : 5
# Fedrep   : 6
# FedBABU  : 7
# FedDW    : 8
# FedDW(pFL) : 9
# Fedper   : 10
# Per-Fedavg(FO) : 11
# FedALA   : 12
# loacl_only : 13



# local_only
class Configs:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 0.001
        self.server_epoch =52
        self.client_epoch=5
        self.n_client=10
        self.batch_size = 128
        self.participation_rate=0.5
        self.idd=0
        self.dirichlet_alpha = 0.1



        self.id_model=3
        self.id_dataset=3
        self.id_method=3

        self.name_model=model_list[self.id_model]
        self.name_dataset=dataset_list[self.id_dataset]
        self.name_method=method_list[self.id_method]


        #by moon
        self.max_pre_net_num=1
        self.moon_coefficient=5
        self.temperature=0.5
        #by Fedprox
        self.Fedprox_coefficient=0.01
        #by Fedproto
        self.Fedproto_coefficient=1
        #by Fedrep  only fc3
        self.head_ep=1
        #by FedDW
        self.FedDW_coefficient=0.1
        self.init_align=0
        self.coefficient_change=True
        #by Per-Fedavg
        self.alpha=1e-2
        self.beta=1e-3
        #by FedALA
        self.layer_idx=2
        #by FedUV
        self.std_coeff=2.5
        self.unif_coeff=0.5




        self.n_class=0
        if self.id_dataset==0:
            self.n_class=62
            pass
        elif self.id_dataset==1:
            self.n_class=10
            pass
        elif self.id_dataset==2:
            self.n_class=10
            pass
        elif self.id_dataset==3:
            self.n_class=10
            pass
        elif self.id_dataset==4:
            self.n_class=100
            pass
        elif self.id_dataset==5:
            self.n_class=2
            pass
        pass


        self.hidden_size=0
        self.out_size=0
        if self.name_model == "CNN_emnist":
            self.hidden_size=7*7*32
            self.out_size=128
            pass
        elif self.name_model == 'CNN_mnist':
            self.hidden_size=320
            self.out_size=128
            pass
        elif self.name_model == 'CNN_fashion_mnist':
            self.hidden_size=7*7*32
            self.out_size=128
            pass
        elif self.name_model == 'CNN_cifar10':
            self.hidden_size=16 * 5 * 5
            self.out_size=128
            pass
        elif self.name_model == 'ResNet18':
            self.hidden_size=512
            self.out_size=128
            pass
        elif self.name_model == 'ShuffleNet':
            self.out_size = 128
            self.hidden_size=800
            pass
        elif self.name_model == 'MobileNet':
            self.out_size = 128
            self.hidden_size=640
            pass
        elif self.name_model == 'LSTM':
            self.out_size = 128
            self.hidden_size=128
            pass
    pass



