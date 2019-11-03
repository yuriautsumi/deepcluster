from skimage import io
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
class CartoonDataset(Dataset):
        """Face Landmarks dataset."""
        def __init__(self, root, path_file, transform=None):
                """
                Args:
                        csv_file (string): Path to the csv file with annotations.
                        root_dir (string): Directory with all the images.
                        transform (callable, optional): Optional transform to be applied
                                on a sample.
                """
                self.paths = open(path_file, 'r').read().split('\n')[:-1]
                self.root = root
                self.transform = transform
                self.videos = sorted(list(set([p.split('/')[1] for p in self.paths])))
                self.imgs = list(map(lambda x:[os.path.join(root, x)], self.paths))
        def __len__(self):
                return len(self.paths)
        def __getitem__(self, idx):
                if torch.is_tensor(idx):
                        idx = idx.tolist()
                img_name = os.path.join(self.root,
                                                                self.paths[idx])
                image = Image.open(img_name)
                if self.transform:
                        image = self.transform(image)
                path_parts = img_name.split('/')
                video_num = self.videos.index(path_parts[-3])
                shot_num = int(path_parts[-2].split('shot_')[-1])
                frame_num = int(path_parts[-1].split('.')[0])
                sample = {'image': image, 'video_num': video_num, 'shot_num': shot_num, 'frame_num': frame_num}
                return sample
