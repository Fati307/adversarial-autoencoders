import os
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dset
import torchvision.transforms as transforms
from AAE.config_aae import data_root, img_size, batchSize, device 
from CAAE.config_caae import data_root, img_size, batchSize, device
from CAAEHybrid.config_h_caae import data_root, img_size, batchSize, device


def get_dataloaders(data_root, img_size=32, batch_size=64):
    train_path = os.path.join(data_root, "mnist")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = dset.ImageFolder(root=train_path, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=torch.cuda.is_available())
    
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_loader, test_loader
