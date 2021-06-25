from __future__ import print_function
import os
from pathlib import Path

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from modules.train_utils import get_parser, train, test
from modules.model import Net


def main():
    # Training settings
    parser = get_parser()
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(args.data_folder, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(args.data_folder, train=False,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        save_folder: Path = args.results_path / os.environ.get("NEURO_JOB_ID")
        save_folder.mkdir(exist_ok=True)
        save_path = save_folder / "mnist_cnn.pt"
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
