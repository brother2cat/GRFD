import torch
from torch.nn import functional as F
import torch.nn as nn


def cal_input_loss_grad(model, inputs, label, f_loss, backprop=True, cvae=False):
    """

    :param cvae:如果是CVAE训练则为True，因为不能直接对输入进行扰动求梯度相似度，
                如果直接求梯度相似度，那么训练出来的是加上VAE之后的
                求的仅仅是CVAE中classifier的扰动梯度相似度
    :param model:
    :param inputs:
    :param label:
    :param f_loss:
    :param backprop: 需要求二次导则为True
    :return:
    """
    if not cvae:
        out = model(inputs)
        loss = f_loss(out, label)
        grad = torch.autograd.grad(loss, inputs, create_graph=True if backprop else False)[0]
        if not backprop:
            return grad.detach()
        else:
            return grad
    else:
        out = model(inputs, True)
        loss = f_loss(out, label)
        grad = torch.autograd.grad(loss, inputs, create_graph=True if backprop else False)[0]
        if not backprop:
            return grad.detach()
        else:
            return grad


def cal_simOllg(model, input, label, eps, norm, disnorm, device,
                delta_init='gaussian', sim_method="cosine", backprop=False, cvae=False):
    """
    计算正常样本loss的梯度以及加了扰动的样本梯度的相似性，来衡量模型的鲁棒性，相似性越高鲁棒性越强
    :param cvae: 如果是CVAE训练则为True，因为不能直接对输入进行扰动求梯度相似度，如果直接求梯度相似度，那么训练出来的是加上VAE之后的
                求的仅仅是CVAE中classifier的扰动梯度相似度
    :param model:
    :param input:
    :param label:
    :param eps: 随机扰动的幅度
    :param norm: 对图片归一化
    :param disnorm: 消除对图片的归一化，反归一化
    :param device:
    :param delta_init:扰动的产生方式
    :param sim_method:衡量相似度的方法 一个是余弦相似度，另一种可以对于各种范数比较差异性
    :param backprop: 需要求二次导则为True
    :return:返回的是计算的相似度
    """
    f_loss = torch.nn.CrossEntropyLoss().to(device)
    batch_size = input.size(0)
    if sim_method == "cosine":
        criterion = torch.nn.CosineSimilarity(dim=1, eps=0.)
    elif sim_method == "euclidean":
        criterion = torch.nn.PairwiseDistance(p=2)
    else:
        criterion = None
        assert True, "cosine"

    if delta_init == 'none':
        delta = torch.zeros_like(input)
    elif delta_init == 'ones':
        delta = torch.ones_like(input) * eps
    elif delta_init == "gaussian":
        delta = torch.randn_like(input) * eps
    elif delta_init == 'uniform':
        delta = (torch.rand_like(input) - 0.5) * 2 * eps
    else:
        raise ValueError('Invalid delta init')

    if not cvae:
        x_pert = disnorm(input)
        x_pert = x_pert + delta
        x_pert = x_pert.clone().detach().requires_grad_(True).to(device)
        x_pert = norm(x_pert)

        input.requires_grad = True

        clean_grad = cal_input_loss_grad(model, input, label, f_loss, backprop=backprop)
        pert_grad = cal_input_loss_grad(model, x_pert, label, f_loss, backprop=backprop)
    else:
        _, _, _, rencon_image, _, _ = model(input, False)
        useful_image = input - rencon_image
        pert_image = useful_image + delta
        pert_image = pert_image.clone().detach().requires_grad_(True).to(device)

        clean_grad = cal_input_loss_grad(model, rencon_image, label, f_loss, backprop=backprop, cvae=True)
        pert_grad = cal_input_loss_grad(model, pert_image, label, f_loss, backprop=backprop, cvae=True)

    clean_grad = clean_grad.view(batch_size, -1)
    pert_grad = pert_grad.view(batch_size, -1)

    cos_sim = criterion(clean_grad, pert_grad)
    cos_sim = cos_sim[~torch.isnan(cos_sim)]
    cos_sim = cos_sim[~torch.isinf(cos_sim)]
    # cos_sim = torch.mean(cos_sim)

    return cos_sim



