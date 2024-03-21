from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

class PIE(Dataset):
    def __init__(self,dataset,inversion = None,category=None) -> None:
        super().__init__()
        self.data_path = os.path.join(dataset,"annotation_images")
        self.mapping_file = pd.read_json(os.path.join(dataset,"mapping_file.json")).T
        if category is not None: # 0,1,2,3,4,5,6,7,8,9
            self.mapping_file = self.mapping_file[self.mapping_file['image_path'].str.startswith(str(category))]
    
    def __len__(self):
        return len(self.mapping_file)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path,self.mapping_file.iloc[index].image_path)
        source_prompt = self.mapping_file.iloc[index].original_prompt.replace("[", "").replace("]", "")
        target_prompt = self.mapping_file.iloc[index].editing_prompt.replace("[", "").replace("]", "")
        return image_path, source_prompt, target_prompt

class PIE_NTI_Inversion(Dataset):
    def __init__(self,dataset,inversion,category=None) -> None:
        super().__init__()
        self.data_path = os.path.join(dataset,"annotation_images")
        self.mapping_file = pd.read_json(os.path.join(dataset,"mapping_file.json")).T
        if category is not None: # 0,1,2,3,4,5,6,7,8,9
            self.mapping_file = self.mapping_file[self.mapping_file['image_path'].str.startswith(str(category))]
        self.inversion_path = inversion
        self.check_inversion()

    def __len__(self):
        return len(self.mapping_file)
    
    def check_inversion(self):
        for i in tqdm(range(len(self.mapping_file)),desc="safety checking inversion files"):
            inversion_path = os.path.join(self.inversion_path,self.mapping_file.iloc[i].image_path.split(".")[0])
            if os.path.exists(os.path.join(inversion_path,"inversion_latent.pt"))==False or os.path.join(inversion_path,"uncond_embeddings_list.pt")==False:
                raise ValueError("The inversion files are not complete")

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path,self.mapping_file.iloc[index].image_path)
        inversion_path = os.path.join(self.inversion_path,self.mapping_file.iloc[index].image_path.split(".")[0])
        latent = torch.load(os.path.join(inversion_path,"inversion_latent.pt"),map_location=torch.device('cpu'))
        uncond_embeddings_list = torch.load(os.path.join(inversion_path,"uncond_embeddings_list.pt"),map_location=torch.device('cpu'))
        source_prompt = self.mapping_file.iloc[index].original_prompt.replace("[", "").replace("]", "")
        target_prompt = self.mapping_file.iloc[index].editing_prompt.replace("[", "").replace("]", "")
        return image_path, latent, uncond_embeddings_list, source_prompt, target_prompt
    
