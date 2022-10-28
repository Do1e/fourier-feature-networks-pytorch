import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from PIL import Image
from tqdm import tqdm, trange

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
npdtype = np.float32
network_size = (4, 256)
lr = 1e-4
iters = 2000
mapping_size = 256
picdir = 'pics'
image = Image.open('test.png').convert('RGB')

if not os.path.exists(picdir):
    os.mkdir(picdir)

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return -10.0 * torch.log10(self.mse(x, y))

def input_mapping(x: Tensor, B: Tensor) -> Tensor:
    if B is None:
        return x
    else:
        x_proj = (2.0 * torch.pi * x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

def init_weight(net: nn.Module):
    for m in net.children():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Sequential):
            init_weight(m)
        else:
            pass

def make_network(num_layers: int, num_channels: int, input_channels: int) -> nn.Module:
    layers = []
    layers.append(nn.Linear(input_channels, num_channels))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(num_channels, num_channels))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_channels, 3))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def train_model(network_size: tuple, lr: float, iters: int, B: Tensor,
                train_data: tuple[Tensor, Tensor], test_data: tuple[Tensor, Tensor]) -> dict:
    net = make_network(*network_size, input_mapping(train_data[0], B).shape[-1]).to(dev)
    init_weight(net)
    net.train()
    lossFn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    psnrFn = PSNR()

    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    xs = []
    for i in trange(iters, desc='train iter', leave=False):
        x = input_mapping(train_data[0], B)
        y = train_data[1]
        y_ = net(x)
        loss = lossFn(y_, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            net.eval()
            with torch.no_grad():
                xs.append(i)
                train_psnrs.append(psnrFn(y_, y).item())
                pred_imgs.append(net(input_mapping(test_data[0], B)))
                test_psnrs.append(psnrFn(pred_imgs[-1], test_data[1]).item())
            net.train()
    return {
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': pred_imgs,
        'xs': xs
    }


if __name__ == '__main__':
    img = np.array(image, dtype=npdtype) / 255.0
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r, :]
    Image.fromarray((img*255).astype(np.uint8)).save(pjoin(picdir, 'GT.png'))
    # img.shape = (512, 512, 3)
    img = torch.tensor(img, device=dev)
    
    coords = np.linspace(0, 1, img.shape[0], endpoint=False, dtype=npdtype)
    # x_test.shape = (512, 512, 2), grid of (x, y) coordinates
    x_test = np.stack(np.meshgrid(coords, coords), axis=-1)
    x_test = torch.tensor(x_test, device=dev)
    # (x, y) coordinates (512*512, 2) -> image (512*512, 3)
    test_data = (x_test.reshape(-1, x_test.shape[-1]), img.reshape(-1, img.shape[-1]))
    # down sampling
    # (256*256, 2) -> (256*256, 3)
    train_data = ([x_test[::2, ::2, :].reshape(-1, x_test.shape[-1]), img[::2, ::2, :].reshape(-1, img.shape[-1])])

    B_dict = {}
    # different mapping methods
    B_dict['none'] = None
    B_dict['basic'] = torch.tensor(np.eye(2, dtype=npdtype), device=dev)
    for scale in (1.0, 10.0, 100.0):
        B_dict[f'gauss_{scale}'] = torch.tensor(np.random.normal(size=(mapping_size, 2), scale=scale).astype(npdtype), device=dev)

    # train
    output = {}
    for k in tqdm(B_dict):
        output[k] = train_model(network_size, lr, iters, B_dict[k], train_data, test_data)
        output[k]['img'] = (np.clip(output[k]['pred_imgs'][-1].cpu().numpy().reshape(img.shape), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(output[k]['img']).save(pjoin(picdir, f'{k}.png'))

    # plot
    fig, ax = plt.subplots(1, len(B_dict) + 1, figsize=(20, 5))
    ax[0].imshow(img.cpu().numpy())
    ax[0].set_title('GT')
    for i, k in enumerate(B_dict):
        ax[i+1].imshow(output[k]['img'])
        ax[i+1].set_title(k)
    plt.savefig(pjoin(picdir, 'all.png'), dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for i, k in enumerate(B_dict):
        ax[0].plot(output[k]['xs'], output[k]['train_psnrs'], label=k)
        ax[1].plot(output[k]['xs'], output[k]['test_psnrs'], label=k)
    ax[0].set_title('train psnr')
    ax[1].set_title('test psnr')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('iter')
    ax[1].set_xlabel('iter')
    ax[0].set_ylabel('psnr')
    ax[1].set_ylabel('psnr')
    plt.savefig(pjoin(picdir, 'psnr.png'), dpi=300)
