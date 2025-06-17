from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torchvision
import torch 
from torch.utils.data import Subset
import matplotlib.pyplot as plt 
from torchvision.transforms import transforms
from tqdm import tqdm 



def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] + 0.5, str(y[i]), ha='center', fontsize=10)


def data_imbalance_check(path):
    class_names = ['BACKGROUND', 'E1', 'E2', 'E3', 'E40', 'E5H', 'E6', 'E8', 'EHRB']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    dataset = torchvision.datasets.ImageFolder(path, transform=torchvision.transforms.ToTensor())


    # Save shapes of images larger than 512 in width
    with open('shape_of_images.txt', 'w') as file:
        file.write('Shape of different image files \n')
        for data in tqdm(dataset):
            file.write(str(data[0].shape) + '\n')

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
    plt.savefig("data_imbalance_check.jpg")