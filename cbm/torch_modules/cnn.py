from typing import Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cbm.torch_modules.utils as ptu
from cbm.utils.misc_untils import to_list


def build_cnn(
    input_shape,
    cnn_kernels: List[List[int]], 
    activation: Union[str, List[str]],
    output_activation: str
) -> Tuple[nn.Module, Tuple[int, int, int]]:
    act_name = to_list(activation, len(cnn_kernels)-1)
    act_name.append(output_activation)
    act_func = [ptu.get_activation(act) for act in act_name]
    module_list = []
    last_h = input_shape[1]
    last_w = input_shape[2]
    default = [None, None, None, 1, 0]
    for i, ck in enumerate(cnn_kernels):
        assert len(ck) <= 5 and len(ck) >= 3
        ck = ck + default[len(ck):]
        last_h = int( (last_h+2*ck[4]-ck[2]) / ck[3] +1) #(h+2*pad-k)/stride+1
        last_w = int( (last_w+2*ck[4]-ck[2]) / ck[3] +1)
        module_list.append(nn.Conv2d(*ck))
        module_list.append(act_func[i])
    output_shape = (cnn_kernels[-1][1], last_h, last_w)
    return nn.Sequential(*module_list), output_shape


def build_cnn_trans(
    input_shape,
    cnn_trans_kernels: List[List[int]], 
    activation: Union[str, List[str]],
    output_activation: str
) -> Tuple[nn.Module, Tuple[int, int, int]]:# no padding, stride is given
    act_name = to_list(activation, len(cnn_trans_kernels)-1)
    act_name.append(output_activation)
    act_func = [ptu.get_activation(act) for act in act_name]
    module_list = []
    last_h = input_shape[1]
    last_w = input_shape[2]
    default = [None, None, None, 1, 0, 0] #in_c, out_c, kernel, stride, pad, outpad
    for i, ck in enumerate(cnn_trans_kernels):
        assert len(ck) <= 6 and len(ck) >= 3
        ck = ck + default[len(ck):]
        last_h = (last_h - 1) * ck[3] - 2 * ck[4] + ck[2] + ck[5]
        last_w = (last_w - 1) * ck[3] - 2 * ck[4] + ck[2] + ck[5]
        module_list.append(nn.ConvTranspose2d(*ck))
        module_list.append(act_func[i])
    output_shape = (cnn_trans_kernels[-1][1], last_h, last_w)
    return nn.Sequential(*module_list), output_shape


class CNN(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(
        self, 
        input_shape,
        output_size=None,
        cnn_kernels=[[-1,32,3,2], [32,32,3,1], [32,32,3,1], [32,32,3,1]],
        activation="relu",
        output_activation="relu"
    ):
        # TODO: train and eval
        super().__init__()
        assert len(input_shape) == 3
        self.cnn_kernels = cnn_kernels
        cnn_kernels[0][0] = input_shape[0]
        self.num_layers = len(cnn_kernels)
        self.input_shape = input_shape
        # no padding, stride is given
        self.module, self.latent_shape = build_cnn(
            input_shape,
            cnn_kernels,
            activation,
            output_activation
        )
        self.latent_size = np.prod(self.latent_shape)
        if output_size is not None:
            assert self.latent_size == output_size, (self.latent_size, output_size)
        else:
            output_size = self.latent_size
        self.output_size = output_size
        self.output_shape = (output_size, )

    def process(self, obs):
        h = self.forward(obs)
        h = h.view(h.size(0), -1)
        return h
    
    def process_feature_map(self, obs):
        h = self.forward(obs)
        return h

    def process_traj(self, obs):
        # reshape for traj input
        B,L,_,_,_ = obs.shape
        obs = obs.view(B*L, *self.input_shape)
        # network forward
        h = self.forward(obs)
        h = h.view(B,L,self.output_size)
        return h

    def forward(self, obs):
        obs = obs/255.0
        h = self.module(obs)
        return h


#note: normalize target
class CNNTrans(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(
        self, 
        input_shape,
        output_shape=None,
        cnn_trans_kernels=[
            [-1, 256, 4],
            [256, 128, 3, 2, 1, 1],
            [128, 64, 3, 2, 1, 1],
            [64, 32, 3, 2, 1, 1],
            [32, 3, 5, 2, 2, 1]
        ], 
        activation="leaky_relu_0.2",
        output_activation="identity"
    ):
        super().__init__()
        if type(input_shape) is int:
            self.input_shape = (input_shape, 1, 1)
        else:
            self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.cnn_trans_kernels = cnn_trans_kernels
        cnn_trans_kernels[0][0] = input_shape[0]
        assert len(output_shape) == 3
        self.num_layers = len(cnn_trans_kernels)

        #get activation functions
        self.module, h_shape = build_cnn_trans(
            self.input_shape,
            cnn_trans_kernels,
            activation,
            output_activation
        )
        if output_shape is not None:
            for i in range(len(h_shape)):
                assert output_shape[i] == h_shape[i], (i, output_shape[i], h_shape[i])
        self.output_shape = h_shape

    def process(self, latent):
        img = latent.view(-1, *self.input_shape)
        return self.forward(img)

    def process_traj(self, latent):
        B, L = latent.shape[0], latent.shape[1]
        latent = latent.view(B*L, *self.input_shape)
        img = self.forward(latent)
        img = img.view(B, L, *self.output_shape)
        return img

    def forward(self, latent):
        return self.module(latent)



# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR

# class Net(CNNEncoder):
#     def __init__(self, **kwargs):
#         super().__init__([1,28,28], 128, [[1,32,3,1]] + [[32,32,3,1]]*2, "relu", "tanh", "nn.BatchNorm1d", **kwargs)
#         self.last_fc = nn.Linear(128, 10)

#     def forward(self, obs, detach=False):
#         out = super().forward(obs, detach=detach)
#         out = self.last_fc(out)
#         output = F.log_softmax(out, dim=1)
#         return output

    
# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=14, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--dry-run', action='store_true', default=False,
#                         help='quickly check a single pass')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()

#     torch.manual_seed(args.seed)

#     device = torch.device("cuda" if use_cuda else "cpu")

#     train_kwargs = {'batch_size': args.batch_size}
#     test_kwargs = {'batch_size': args.test_batch_size}
#     if use_cuda:
#         cuda_kwargs = {'num_workers': 1,
#                        'pin_memory': True,
#                        'shuffle': True}
#         train_kwargs.update(cuda_kwargs)
#         test_kwargs.update(cuda_kwargs)

#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     dataset1 = datasets.MNIST('../data', train=True, download=True,
#                        transform=transform)
#     dataset2 = datasets.MNIST('../data', train=False,
#                        transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

#     model = Net().to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(model, device, test_loader)
#         scheduler.step()

#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")


# if __name__ == '__main__':
#     main()