import numpy as np
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.targets = self.data.target[self.data.is_training_img == 1]
            self.data = self.data.filepath[self.data.is_training_img == 1]
        else:
            self.targets = self.data.target[self.data.is_training_img == 0]
            self.data = self.data.filepath[self.data.is_training_img == 0]
        
        self.targets = self.targets.to_numpy()-1
        self.classes = {}
        for filepath in self.data:
            class_id, class_name = filepath.split('/')[0].split('.')
            class_id = int(class_id)
            self.classes[class_id] = class_name
        self.classes = [self.classes[i] for i in range(1, len(self.classes)+1)]
        self.data = np.array([os.path.join(self.root, self.base_folder, path) for path in self.data])

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for filepath in self.data:
            if not os.path.isfile(filepath):
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        target = self.targets[idx]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    cub = Cub2011('data/', transform=transform)
    print(cub.classes)
    input("")
