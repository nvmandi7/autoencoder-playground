import torch
from torch.utils.data import Dataset
from torchvision.datasets import INaturalist
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Progress toward a dataset that takes part of INaturalist


class PartialINaturalistDataset(datasets.INaturalist):
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/inaturalist.html#INaturalist
    """`iNaturalist <https://github.com/visipedia/inat_comp>`_ Dataset.

    Args:
        root (string): Root directory of dataset where the image files are stored.
            This class does not require/use annotation files.
        version (string, optional): Which version of the dataset to download/use. One of
            '2017', '2018', '2019', '2021_train', '2021_train_mini', '2021_valid'.
            Default: `2021_train`.
        target_type (string or list, optional): Type of target to use, for 2021 versions, one of:

            - ``full``: the full category (species)
            - ``kingdom``: e.g. "Animalia"
            - ``phylum``: e.g. "Arthropoda"
            - ``class``: e.g. "Insecta"
            - ``order``: e.g. "Coleoptera"
            - ``family``: e.g. "Cleridae"
            - ``genus``: e.g. "Trichodes"

            for 2017-2019 versions, one of:

            - ``full``: the full (numeric) category
            - ``super``: the super category, e.g. "Amphibians"

            Can also be a list to output a tuple with all specified target types.
            Defaults to ``full``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        version: str = "2021_train",
        target_type: Union[List[str], str] = "full",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.version = verify_str_arg(version, "version", DATASET_URLS.keys())
        super().__init__(os.path.join(root, version), transform=transform, target_transform=target_transform)

        os.makedirs(root, exist_ok=True)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.all_categories: List[str] = []

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {}

        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        if not isinstance(target_type, list):
            target_type = [target_type]
        if self.version[:4] == "2021":
            self.target_type = [verify_str_arg(t, "target_type", ("full", *CATEGORIES_2021)) for t in target_type]
            self._init_2021()
        else:
            self.target_type = [verify_str_arg(t, "target_type", ("full", "super")) for t in target_type]
            self._init_pre2021()

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.all_categories):
            files = os.listdir(os.path.join(self.root, dir_name))
            for fname in files:
                self.index.append((dir_index, fname))

    def _init_2021(self) -> None:
        """Initialize based on 2021 layout"""

        self.all_categories = sorted(os.listdir(self.root))

        # map: category type -> name of category -> index
        self.categories_index = {k: {} for k in CATEGORIES_2021}

        for dir_index, dir_name in enumerate(self.all_categories):
            pieces = dir_name.split("_")
            if len(pieces) != 8:
                raise RuntimeError(f"Unexpected category name {dir_name}, wrong number of pieces")
            if pieces[0] != f"{dir_index:05d}":
                raise RuntimeError(f"Unexpected category id {pieces[0]}, expecting {dir_index:05d}")
            cat_map = {}
            for cat, name in zip(CATEGORIES_2021, pieces[1:7]):
                if name in self.categories_index[cat]:
                    cat_id = self.categories_index[cat][name]
                else:
                    cat_id = len(self.categories_index[cat])
                    self.categories_index[cat][name] = cat_id
                cat_map[cat] = cat_id
            self.categories_map.append(cat_map)    
    

