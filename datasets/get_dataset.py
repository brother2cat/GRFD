import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F


class transforms_all:
    mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    imagenet_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
    ])
    imagenet_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
    ])


def get_dataset(dataset_name, batch_size, train_transforms=None, test_transforms=None, root=''):
    """

    :param dataset_name: 数据集名字，mnist ,cifar10, miniimagenet
    :param batch_size:
    :param train_transforms:
    :param test_transforms:
    :param root: 数据集路径，默认为当前文件夹
    :return: 训练集， 测试集
    """
    if dataset_name == 'mnist':
        if train_transforms is None:
            train_transforms = transforms_all.mnist_train
        if test_transforms is None:
            test_transforms = transforms_all.mnist_test
        mnist_train = datasets.MNIST('datasets/mnist', train=True, transform=train_transforms, download=True)
        mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        mnist_test = datasets.MNIST('datasets/mnist', train=False, transform=test_transforms, download=True)
        total_test = len(mnist_test)
        mnist_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
        return mnist_train, mnist_test, total_test

    elif dataset_name == 'cifar10':
        if train_transforms is None:
            train_transforms = transforms_all.cifar_train
        if test_transforms is None:
            test_transforms = transforms_all.cifar_test
        cifar_train = datasets.CIFAR10('datasets/cifar', train=True, transform=train_transforms, download=True)
        cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
        cifar_test = datasets.CIFAR10('datasets/cifar', train=False, transform=test_transforms, download=True)
        total_test = len(cifar_test)
        # print(total_test)
        cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
        label_class = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        return cifar_train, cifar_test, total_test, label_class
    
    elif dataset_name == 'cifar100':
        if train_transforms is None:
            train_transforms = transforms_all.cifar_train
        if test_transforms is None:
            test_transforms = transforms_all.cifar_test
        cifar_train = datasets.CIFAR100('datasets/cifar100', train=True, transform=train_transforms, download=True)
        cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
        cifar_test = datasets.CIFAR100('datasets/cifar100', train=False, transform=test_transforms, download=True)
        total_test = len(cifar_test)
        # print(total_test)
        cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
        return cifar_train, cifar_test, total_test

    elif dataset_name == 'miniimagenet':
        if train_transforms is None:
            train_transforms = transforms_all.imagenet_train
        if test_transforms is None:
            test_transforms = transforms_all.imagenet_test
        train_set = datasets.ImageFolder("datasets/MiniImageNet/train", transform=train_transforms)
        test_set = datasets.ImageFolder("datasets/MiniImageNet/val", transform=test_transforms)
        test_num = len(test_set)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        label_class = {'n01532829': 'house_finch',
                       'n01558993': 'robin',
                       'n01704323': 'triceratops',
                       'n01749939': 'green_mamba',
                       'n01770081': 'harvestman',
                       'n01843383': 'toucan',
                       'n01855672': 'goose',
                       'n01910747': 'jellyfish',
                       'n01930112': 'nematode',
                       'n01981276': 'king_crab',
                       'n02074367': 'dugong',
                       'n02089867': 'Walker_hound',
                       'n02091244': 'Ibizan_hound',
                       'n02091831': 'Saluki',
                       'n02099601': 'golden_retriever',
                       'n02101006': 'Gordon_setter',
                       'n02105505': 'komondor',
                       'n02108089': 'boxer',
                       'n02108551': 'Tibetan_mastiff',
                       'n02108915': 'French_bulldog',
                       'n02110063': 'malamute',
                       'n02110341': 'dalmatian',
                       'n02111277': 'Newfoundland',
                       'n02113712': 'miniature_poodle',
                       'n02114548': 'white_wolf',
                       'n02116738': 'African_hunting_dog',
                       'n02120079': 'Arctic_fox',
                       'n02129165': 'lion',
                       'n02138441': 'meerkat',
                       'n02165456': 'ladybug',
                       'n02174001': 'rhinoceros_beetle',
                       'n02219486': 'ant',
                       'n02443484': 'black-footed_ferret',
                       'n02457408': 'three-toed_sloth',
                       'n02606052': 'rock_beauty',
                       'n02687172': 'aircraft_carrier',
                       'n02747177': 'ashcan',
                       'n02795169': 'barrel',
                       'n02823428': 'beer_bottle',
                       'n02871525': 'bookshop',
                       'n02950826': 'cannon',
                       'n02966193': 'carousel',
                       'n02971356': 'carton',
                       'n02981792': 'catamaran',
                       'n03017168': 'chime',
                       'n03047690': 'clog',
                       'n03062245': 'cocktail_shaker',
                       'n03075370': 'combination_lock',
                       'n03127925': 'crate',
                       'n03146219': 'cuirass',
                       'n03207743': 'dishrag',
                       'n03220513': 'dome',
                       'n03272010': 'electric_guitar',
                       'n03337140': 'file',
                       'n03347037': 'fire_screen',
                       'n03400231': 'frying_pan',
                       'n03417042': 'garbage_truck',
                       'n03476684': 'hair_slide',
                       'n03527444': 'holster',
                       'n03535780': 'horizontal_bar',
                       'n03544143': 'hourglass',
                       'n03584254': 'iPod',
                       'n03676483': 'lipstick',
                       'n03770439': 'miniskirt',
                       'n03773504': 'missile',
                       'n03775546': 'mixing_bowl',
                       'n03838899': 'oboe',
                       'n03854065': 'organ',
                       'n03888605': 'parallel_bars',
                       'n03908618': 'pencil_box',
                       'n03924679': 'photocopier',
                       'n03980874': 'poncho',
                       'n03998194': 'prayer_rug',
                       'n04067472': 'reel',
                       'n04146614': 'school_bus',
                       'n04149813': 'scoreboard',
                       'n04243546': 'slot',
                       'n04251144': 'snorkel',
                       'n04258138': 'solar_dish',
                       'n04275548': 'spider_web',
                       'n04296562': 'stage',
                       'n04389033': 'tank',
                       'n04418357': 'theater_curtain',
                       'n04435653': 'tile_roof',
                       'n04443257': 'tobacco_shop',
                       'n04509417': 'unicycle',
                       'n04515003': 'upright',
                       'n04522168': 'vase',
                       'n04596742': 'wok',
                       'n04604644': 'worm_fence',
                       'n04612504': 'yawl',
                       'n06794110': 'street_sign',
                       'n07584110': 'consomme',
                       'n07613480': 'trifle',
                       'n07697537': 'hotdog',
                       'n07747607': 'orange',
                       'n09246464': 'cliff',
                       'n09256479': 'coral_reef',
                       'n13054560': 'bolete',
                       'n13133613': 'ear'}
        return train_loader, test_loader, test_num, train_set.class_to_idx, label_class

    else:
        raise ValueError("数据集名字只支持mnist ,cifar10, miniimagenet")


def main():
    # a, b, c = get_dataset("mnist", 128)
    # d, e, f, g = get_dataset("cifar10", 128)
    h, i, j, k, l = get_dataset("miniimagenet", 128)
    for batch_idx, (x, label) in enumerate(h):
        print(x.shape)
        print(label.shape)
        break
    for batch_idx, (x, label) in enumerate(i):
        print(x.shape)
        print(label.shape)
        break


if __name__ == '__main__':
    main()
