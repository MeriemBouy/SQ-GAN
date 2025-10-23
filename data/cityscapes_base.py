import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random


class CustomLabelTransform:
    def __init__(self, relevant_obj):
        self.relevant_obj = torch.tensor(relevant_obj)

        
    def __call__(self, label):
        label = self.onehot_encode(label)

        # Select only the pixels corresponding to the relevant objects
        label = label * self.relevant_obj.unsqueeze(0).unsqueeze(2).unsqueeze(2) 
        label = self.onehot_decode(label).to(torch.long)
        label = label.unsqueeze(0)
        return label.to(torch.float32)
    
    def onehot_encode(self, label):
        label[label == 255] = 19
        onehot = torch.nn.functional.one_hot(label, num_classes=20).permute(0,3,1,2).float()
        onehot = onehot[:,:-1,:,:]
        return onehot
    
    def onehot_decode(self, onehot):
        label = torch.max(onehot, dim=1)
        return label.values
    
class Custom_shared_transform(object):
    def __init__(self, size, ratio_dim, p=0.5, crop_size=None):
        self.p = p
        self.crop_size = crop_size
        self.size = size
        self.ratio_dim = ratio_dim
        if self.crop_size:
            self.x_limit = (size*2 - crop_size)*2
            self.y_limit = size*2 - crop_size
        self.transform_label = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __call__(self, image, label):
        # random_rotation
        if random.random() < self.p:
            # rotate image of at most 5 degrees
            angle = random.randint(-5, 5)
            image = transforms.functional.rotate(img=image, angle=angle, expand=False)
            label = transforms.functional.rotate(img=label, angle=angle, interpolation=Image.NEAREST, expand=False, fill=255)
        
        # random_crop
        if random.random() < self.p and self.crop_size:
            image = transforms.functional.resize(image, (self.size*2, int(self.size * self.ratio_dim)*2))
            label = transforms.functional.resize(label, (self.size*2, int(self.size * self.ratio_dim)*2), interpolation=Image.NEAREST)
            # select at random coordinates left-top corner of the crop
            x = random.randint(25, self.x_limit-25)
            y = random.randint(15, self.y_limit-150) # I don't want to crop the ceeiling
            image = transforms.functional.crop(image, top=y, left=x, height=self.crop_size, width=self.crop_size * self.ratio_dim)
            label = transforms.functional.crop(label, top=y, left=x, height=self.crop_size, width=self.crop_size * self.ratio_dim)
        else:
            image = transforms.functional.resize(image, (self.size, int(self.size * self.ratio_dim)))
            label = transforms.functional.resize(label, (self.size, int(self.size * self.ratio_dim)), interpolation=Image.NEAREST)
        
        label = self.transform_label(label)
        
        return image, label


class ImagePaths(Dataset):
    def __init__(self, split, is_val, paths, elements_input, size=None, random_crop=False, ratio_dim=1.0, relevant_obj=False, classes=19):
        self.size = size
        self.split = split
        self.random_crop = random_crop
        self.ratio_dim = ratio_dim
        self.relevant_obj = relevant_obj
        self.classes = classes + 1
 
        self.elements_input = elements_input
        self.elements_input["file_path_"] = paths
        self._length = len(elements_input["file_path_"])

        
        self.label_transform = transforms.Compose([
                transforms.Resize((self.size, int(self.size * self.ratio_dim)), interpolation=Image.NEAREST),
                CustomLabelTransform(relevant_obj=self.relevant_obj)
                ])
        self.semantic_transform = transforms.Compose([
                transforms.Resize((self.size, int(self.size * self.ratio_dim)), interpolation=Image.NEAREST),
                ])  
        if split == "train":
            self.transform_shared_train = Custom_shared_transform(size=self.size, ratio_dim=self.ratio_dim, p=0.5, crop_size=self.size)
            self.transform_image_train = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1, saturation=0.1)], p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=5, p=0.3),
                transforms.Resize((self.size, int(self.size * self.ratio_dim))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ])
            ])
        else:
            self.transform_image_val = transforms.Compose([
                transforms.Resize((self.size, int(self.size * self.ratio_dim))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ])
            ])
            self.transform_label_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.size, int(self.size * self.ratio_dim)), interpolation=Image.NEAREST)
            ])

    def __len__(self):
        return self._length

    def preprocess(self, image_path, ssm_path):
        image = Image.open(image_path).convert("RGB")
        ssm = Image.open(ssm_path).convert("L")

        if self.split == "train":
            image, ssm = self.transform_shared_train(image, ssm)
            image = self.transform_image_train(image)
        else:
            image = self.transform_image_val(image)
            ssm = self.transform_label_val(ssm)

        ssm = torch.tensor(np.array(ssm*255), dtype=torch.long)
        ssm[ssm == 255] = 19
        label_ = self.label_transform(ssm).squeeze(0)
        ssm = self.semantic_transform(ssm)
        _,h,w = ssm.shape
        semantic_scatter = torch.zeros((self.classes,h,w), dtype=torch.float32)
        semantic_scatter = semantic_scatter.scatter_(0, ssm, 1)
        return image, label_, semantic_scatter

    def __getitem__(self, i):
        image, label, semantic = self.preprocess(self.elements_input["file_path_"][i], self.elements_input["ssm_path"][i])
        return {"image": image, "label": label, "semantic": semantic, "filename": self.elements_input["file_path_"][i].split("/")[-1]}
    
