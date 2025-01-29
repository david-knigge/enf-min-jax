import torch
from torch.utils.data import Dataset

from PIL import Image
import requests
import os
import numpy as np



class Ombria(Dataset):

    link = "https://github.com/geodrak/OMBRIA/archive/refs/heads/master.zip"

    def __init__(
            self,
            root: str, 
            split: str='train',
            transform: torch.nn.Module=None,
            target_transform: torch.nn.Module=None,
            download: bool=False
        ):
        """
        Implementation of the Ombria dataset. Download the dataset from the git, extract it, and load the images.
        From the paper: "OmbriaNet—Supervised Flood Mapping via Convolutional Neural Networks Using Multitemporal
        Sentinel-1 and Sentinel-2 Data Fusion" by Drakonakis et al. (2021).

        Args:
            root (str): The root directory where the dataset is stored.
            split (str): The split to use. Can be 'train' or 'test'.
            transform (torch.nn.Module): A transformation to apply to the input data.
            target_transform (torch.nn.Module): A transformation to apply to the target data.
            download (bool): Whether to download the dataset.
        """
        self.transform = transform
        self.target_transform = target_transform

        # Download and extract the dataset
        if download:
            if os.path.exists(root):
                print("Dataset already downloaded.")
            else:
                os.makedirs(root, exist_ok=False)
                response = requests.get(self.link)
                with open(root + "/OMBRIA.zip", "wb") as file:
                    file.write(response.content)

                # unzip
                import zipfile
                import shutil

                os.makedirs(root + '/raw', exist_ok=True)
                with zipfile.ZipFile(root + "/OMBRIA.zip", 'r') as zip_ref:
                    zip_ref.extractall(root + '/raw')
                os.remove(root + "/OMBRIA.zip")

                # Move the folders OmbriaS1 and OmbriaS2 to the root
                os.rename(root + "/raw/OMBRIA-master/OmbriaS1", root + "/OmbriaS1")
                os.rename(root + "/raw/OMBRIA-master/OmbriaS2", root + "/OmbriaS2")
                shutil.rmtree(root + "/raw/OMBRIA-master")

        # The dataset is contained in folders called "OmbriaS1" and OmbriaS2"
        self.data = []
        self.masks = []
        before_paths = os.listdir(os.path.join(root, f"OmbriaS1", split, "BEFORE"))
        ids = [path.split("/")[-1].split(".")[0].split("_")[-1] for path in before_paths]
        for id in ids:
            sample = []
            # Load the sample, before and after, S1 and S2
            for state in ["BEFORE", "AFTER"]:
                for sat in ["S1", "S2"]:
                    im = np.array(Image.open(os.path.join(root, f"Ombria{sat}", split, state, f"{sat}_{state.lower()}_{id}.png"))) / 255
                    if sat == "S1":
                        im = np.expand_dims(im, axis=-1)
                    sample.append(im)
            
            # Concatenate the sample
            sample = np.concatenate(sample, axis=-1)

            # Save the sample
            self.data.append(sample)

            # Load the mask
            mask = np.array(Image.open(os.path.join(root, f"OmbriaS1", split, "MASK", f"S1_mask_{id}.png"))) / 255
            self.masks.append(mask[..., None])

    def __getitem__(self, index: int):
        """ Return a sample from the dataset. The sample is a tuple of 4 images:
        - The Sentinel-1 image before the flood
        - The Sentinel-2 image before the flood
        - The Sentinel-1 image after the flood
        - The Sentinel-2 image after the flood

        Also return the mask of the flooded area.
        """
        return self.data[index], self.masks[index]

    def __len__(self):
        return len(self.data)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Download the dataset
    dataset = Ombria(root="./data/ombria", download=True)

    # Visualize a sample
    sample, mask = dataset[0]
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(sample[0])
    ax[1].imshow(sample[1])
    ax[2].imshow(mask)
    plt.show()

