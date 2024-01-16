import cnn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
import trainer
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from time import perf_counter as pc


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, padding=2)
        self.conv2 = nn.Conv2d(2, 4, 5, stride=2)
        self.conv3 = nn.Conv2d(4, 8, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # the architecture requires uneven padding on left to
        # right and top to bottom of image.
        x = F.pad(x, (2, 1, 2, 1))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# helper function for calculating moving average
def moving_average(array, w=3):
    psum = np.cumsum(array, dtype=float)
    psum[w:] = psum[w:] - psum[:-w]
    return psum[w - 1:] / w


def plot_errors(ax, torch_cnn, my_cnn, epochs, w):
    '''
    fig, ax = plt.subplots()
    '''
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.plot(
        np.linspace(0, epochs, len(torch_cnn.errors) - w + 1),
        moving_average(torch_cnn.errors, w),
        label="PyTorch: training",
    )
    ax.plot(
        np.linspace(0, epochs, len(torch_cnn.test_errors)),
        torch_cnn.test_errors,
        label="PyTorch: test",
    )
    ax.plot(
        np.linspace(0, epochs, len(my_cnn.errors) - w + 1),
        moving_average(my_cnn.errors, w),
        label="My CNN: training",
    )
    ax.plot(
        np.linspace(0, epochs, len(my_cnn.test_errors)),
        my_cnn.test_errors,
        label="My CNN: test",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.grid()

def plot_losses(ax, torch_cnn, my_cnn, epochs, w):
    '''
    fig, ax = plt.subplots()
    '''
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.plot(
        np.linspace(0, epochs, len(torch_cnn.losses) - w + 1),
        moving_average(torch_cnn.losses, w),
        label="PyTorch: training",
    )
    ax.plot(
        np.linspace(0, epochs, len(torch_cnn.test_losses)),
        torch_cnn.test_losses,
        label="PyTorch: test",
    )
    ax.plot(
        np.linspace(0, epochs, len(my_cnn.costs) - w + 1),
        moving_average(my_cnn.costs, w),
        label="My CNN: training",
    )
    ax.plot(
        np.linspace(0, epochs, len(my_cnn.test_costs)),
        my_cnn.test_costs,
        label="My CNN: test",
    )
    
    ax.set_ylabel("Losses")
    ax.grid()
    

def main():
    # download MNIST data
    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        transform=ToTensor()
    )

    batch_size = 100
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    # choose device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device set to mps\n")
    else:
        device = torch.device("cpu")
        print("Device set to cpu\n")

    # copy data to np array and scale
    X_np = np.expand_dims(train_data.data.float().clone().detach().numpy(),
                          1) / 255.0
    y_np = train_data.targets.clone().detach().numpy()

    test_X_np = np.expand_dims(test_data.data.float().clone().detach().numpy(),
                               1) / 255.0
    test_y_np = test_data.targets.clone().detach().numpy()

    # set loss fns
    loss_fn = nn.CrossEntropyLoss()
    test_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    # set hypyerparameters
    epochs = 2
    lr = 0.001  # learning rate
    test_period = 100

    model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cnn_trainer = trainer.Trainer(
        model,
        device,
        optimizer,
        loss_fn,
        test_loss_fn,
        train_loader,
        test_loader,
        epochs,
        test_period,
    )
    print("PyTorch Model: ")
    print(model)
    print("Starting training on PyTorch model...")
    t0 = pc()
    cnn_trainer.train()
    print(f"Completed training in {((pc()-t0) / 60):.3f} minutes.")

    layer0 = cnn.ConvLayer(channels_in=1, channels_out=2, dim_in=28,
                           dim_out=28, dim_W=5, stride=1, eta=lr)
    layer1 = cnn.ConvLayer(channels_in=2, channels_out=4, dim_in=28,
                           dim_out=14, dim_W=5, stride=2, eta=lr)
    layer2 = cnn.ConvLayer(channels_in=4, channels_out=8, dim_in=14,
                           dim_out=7, dim_W=4, stride=2, eta=lr)
    layer3 = cnn.VeccingLayer(channels_in=8, dim_in=7)
    layer4 = cnn.DenseLayer(dim_in=8 * 7 * 7, dim_out=200, eta=lr)
    layer5 = cnn.SoftmaxLayer(dim_in=200, dim_out=10, eta=lr)
    my_cnn = cnn.Neural(epochs=epochs, num_classes=10)

    my_cnn.add_layer(layer0)
    my_cnn.add_layer(layer1)
    my_cnn.add_layer(layer2)
    my_cnn.add_layer(layer3)
    my_cnn.add_layer(layer4)
    my_cnn.add_layer(layer5)

    print("\nStarting training on my model...")
    t0 = pc()
    my_cnn.fit(X_np, y_np, batch_size=100, test_X=test_X_np,
               test_y=test_y_np, test_period=100)
    print(f"Completed training in {((pc()-t0) / 60):.3f} minutes.")

    w = 20 # window size of moving average
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_losses(ax1, cnn_trainer, my_cnn, epochs, w)
    plot_errors(ax2, cnn_trainer, my_cnn, epochs, w)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()