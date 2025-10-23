import os
import glob
from torch.utils.data import Dataset, DataLoader
from data.default import DefaultDataPath
from data.cityscapes_base import ImagePaths

class CityscapesBase(Dataset):
    def __init__(self, split='train', config=None):
        self.root = DefaultDataPath.Cityscapes.root
        self.split = split
        self.config = config or {}

        self.images_dir = os.path.join(self.root, "leftImg8bit", self.split)
        self.ssm_dir = os.path.join(self.root, "gtFine", self.split)

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*", "*.png")))
        self.ssm_paths = [p.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit.png", "_gtFine_trainIds.png") for p in self.image_paths]

        assert len(self.image_paths) == len(self.ssm_paths), "Mismatch between images and labels"

    def _load_data(self):
        elements_input = {
            "file_path_": self.image_paths,
            "ssm_path": self.ssm_paths
        }
        self.data = ImagePaths(split=self.split, is_val=self.config.get("is_eval", False),
                               paths=self.image_paths, size=self.config.get("size", 256),
                               random_crop=self.config.get("random_crop", False),
                               elements_input=elements_input, ratio_dim=self.config.get("ratio_dim", 1.0),
                               relevant_obj=self.config.get("relevant_obj"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CityscapesTrain(CityscapesBase):
    def __init__(self, config=None):
        super().__init__(split='train', config=config)
        self._load_data()

class CityscapesValidation(CityscapesBase):
    def __init__(self, config=None):
        super().__init__(split='val', config=config)
        self._load_data()

        
if __name__ == "__main__":
    config = {"is_eval": False, "size": 256, "ratio_dim": 1.0}
    root = "datasets/cityscapes"
    
    train_dataset = CityscapesTrain(root=root, config=config)
    val_dataset = CityscapesValidation(root=root, config=config)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
