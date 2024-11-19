import torch
from torch import nn
from torch.utils.data import Dataset
import copy
import h5py
import os


class Norm(nn.Module):
    def __init__(self, dataset):
        super(Norm, self).__init__()
        if dataset == "mnist":
            self.mean = [0.1307]
            self.std = [0.3081]
        elif dataset == "cifar10":
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2470, 0.2435, 0.2616]
        elif dataset == "miniimagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            raise ValueError("数据集名字只支持mnist ,cifar, imagenet")

    def forward(self, x):
        n_channels = len(self.mean)
        mean = torch.tensor(self.mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(self.std).reshape(1, n_channels, 1, 1)
        mean = mean.to(x.device)
        std = std.to(x.device)
        return (x - mean) / std


class Disnorm(nn.Module):
    def __init__(self, dataset):
        super(Disnorm, self).__init__()
        if dataset == "mnist":
            self.mean = [0.1307]
            self.std = [0.3081]
        elif dataset == "cifar10":
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2470, 0.2435, 0.2616]
        elif dataset == "miniimagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            raise ValueError("数据集名字只支持mnist ,cifar, imagenet")

    def forward(self, x):
        n_channels = len(self.mean)
        mean = torch.tensor(self.mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(self.std).reshape(1, n_channels, 1, 1)
        mean = mean.to(x.device)
        std = std.to(x.device)
        return x * std + mean


class Monitor_stop(nn.Module):
    def __init__(self, threshold):
        super(Monitor_stop, self).__init__()
        self.threshold = threshold
        self.max_acc = 0.
        self.model = None
        self.times = 0

    def forward(self, acc, model, root):
        if acc >= self.max_acc:
            self.model = copy.deepcopy(model)
            self.times = 0
            self.max_acc = acc
        else:
            self.times += 1

        if self.times > self.threshold:
            print("----开始保存模型-----")
            torch.save(self.model.state_dict(), root)
            print("----保存模型成功-----")
            return True
        else:
            return False


class h5py_dataset(Dataset):
    """
    进行h5py数据集文件的读取，返回的是一个Dataset需要使用Dataloader进行包装，保存H5PY文件的时候要求两个字典名称分别是image与target
    """

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['image'][index], int(f['target'][index])

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])


class h5py_dataset_z_robust(Dataset):
    """
    进行h5py数据集文件的读取，返回的是一个Dataset需要使用Dataloader进行包装，保存H5PY文件的时候要求两个字典名称分别是image,target,z_robust
    """

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['image'][index], int(f['target'][index]), f['z_robust'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])


class h5py_dataset_zandRx(Dataset):
    """
    进行h5py数据集文件的读取，返回的是一个Dataset需要使用Dataloader进行包装，保存H5PY文件的时候要求两个字典名称分别是image与target
    """

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['image'][index], int(f['target'][index]), f['latent_z'][index], f['Rx_image'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])


class h5py_dataset_adv(Dataset):
    """
    进行h5py数据集文件的读取，返回的是一个Dataset需要使用Dataloader进行包装，保存H5PY文件的时候要求两个字典名称分别是image与target和adv
    """

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['image'][index], int(f['target'][index]), f['adv'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])

class h5py_dataset_adv_distan(Dataset):
    """
    进行h5py数据集文件的读取，返回的是一个Dataset需要使用Dataloader进行包装，保存H5PY文件的时候要求两个字典名称分别是image与target和adv
    以及继续adv解耦的latent_z和Rx_image
    """

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['image'][index], int(f['target'][index]), f['adv'][index], f['latent_z'][index], f['Rx_image'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])


def load_part_model_parameters(model, parameters_root):
    """
    之前一个模型的所有参数被保存下来，需要对子模型加载其部分参数
    """
    model_dict = model.state_dict()
    save_model = torch.load(parameters_root)
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


def load_checkpoint(checkpoint_folder):

    # read what the latest model file is:
    filename = os.path.join(checkpoint_folder, "checkpoint_file")
    if not os.path.exists(filename):
        return None

    # load and return the checkpoint:
    else:
        print("load checkpoint")
        return torch.load(filename)


# function that saves checkpoint:
def save_checkpoint(checkpoint_folder, checkpoint):

    # make sure that we have a checkpoint folder:
    if not os.path.isdir(checkpoint_folder):
        try:
            os.makedirs(checkpoint_folder)
        except BaseException:
            print('| WARNING: could not create directory %s' % checkpoint_folder)
    if not os.path.isdir(checkpoint_folder):
        return False

    # write checkpoint atomically:
    try:
        torch.save(checkpoint, checkpoint_folder + "/checkpoint_file")
        print("save checkpoint to temp_data/checkpoint_file")
        return True
    except BaseException:
        print('| WARNING: could not write checkpoint to %s.' % checkpoint_folder)
        return False


def main():
    a = torch.randn(2, 3, 2, 2)
    print(a)
    norm = Norm("cifar")
    a = norm(a)
    print(a)


if __name__ == '__main__':
    main()
