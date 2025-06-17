import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset



class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)

def get_dataloader(path, get_path=False, batch_size = 64, shuffle=False):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if get_path == True:
        dataset = ImageFolderWithPaths(path, transform=transform)
    else: 
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    # # Random split
    # train_size = int(0.9 * len(dataset))
    # valid_size = len(dataset) - train_size
    # train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    #TODO remove below two lines for overall training
    # train_dataset = Subset(train_dataset, [1])
    # valid_dataset = train_dataset
    # batch_size=1

    if shuffle:
        loader = DataLoader(dataset, batch_size, shuffle=True) 
    else:
        loader = DataLoader(dataset, batch_size, shuffle=False)

    return loader

