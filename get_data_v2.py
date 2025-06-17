import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import random 
from collections import defaultdict
import torch 
import matplotlib.pyplot as plt 
from metrics import addlabels


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)

class BalancedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, max_samples_per_class=2000):
        super().__init__(root, transform=transform)
        self.max_samples_per_class = max_samples_per_class
        self._balance_classes()
    
    def _balance_classes(self):
        class_to_samples = defaultdict(list)
        for idx, (path, class_idx) in enumerate(self.samples):
            class_to_samples[class_idx].append((path, class_idx))
        
        balanced_samples = []
        for class_idx, samples in class_to_samples.items():
            class_name = self.classes[class_idx]
            if len(samples) > self.max_samples_per_class:
                selected_samples = random.sample(samples, self.max_samples_per_class)
                print(f"Class {class_name}: Limited to {self.max_samples_per_class} samples (from {len(samples)})")
            else:
                selected_samples = samples
                print(f"Class {class_name}: Keeping all {len(samples)} samples")
            
            balanced_samples.extend(selected_samples)
        
        self.samples = balanced_samples
        self.targets = [s[1] for s in balanced_samples]
        
        print(f"Total samples after balancing: {len(self.samples)}")


def get_dataloader_v2(path, batch_size=64, shuffle=False, max_samples_per_class=2000):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset = BalancedImageFolder(path, transform=transform, max_samples_per_class=max_samples_per_class)

    if shuffle:
        loader = DataLoader(dataset, batch_size, shuffle=True) 
    else:
        loader = DataLoader(dataset, batch_size, shuffle=False)

    return loader



if __name__ == "__main__":
    train_path = '/home/shirshak/Thesis_Data/DOES/TRAIN/'
    valid_path = '/home/shirshak/Thesis_Data/DOES/TEST_tiles/'

    max_samples_per_class = 2000

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset = BalancedImageFolder(train_path, transform=transform, max_samples_per_class=max_samples_per_class)


    class_names = ['BACKGROUND', 'E1', 'E2', 'E3', 'E40', 'E5H', 'E6', 'E8', 'EHRB']
    # Get labels from dataset
    labels = torch.tensor([label for _, label in dataset])

    # Count frequency of each class
    class_counts = [(labels == i).sum().item() for i in range(len(class_names))]

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts, color='skyblue')
    addlabels(class_names, class_counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution (Data Imbalance Check)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("updated_train_data_imbalance_check.jpg")