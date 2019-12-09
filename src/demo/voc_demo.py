""" https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py """
import os
import os.path
import hashlib
import gzip
import errno
import tarfile
import zipfile

import torch
from torch.utils.model_zoo import tqdm
from torch._six import PY3


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(lambda p: os.path.isdir(os.path.join(root, p)), os.listdir(root))
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root),
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests

    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={"id": file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path) and PY3:
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path, os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(
    url,
    download_root,
    extract_root=None,
    filename=None,
    md5=None,
    remove_finished=False,
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def iterable_to_str(iterable):
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = (
                "Unknown value '{value}' for argument {arg}. "
                "Valid values are {{{valid_values}}}."
            )
            msg = msg.format(
                value=value, arg=arg, valid_values=iterable_to_str(valid_values)
            )
        raise ValueError(msg)

    return value


""" https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py """
import os
import sys
import tarfile
import collections
from torchvision.datasets.vision import VisionDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": os.path.join("VOCdevkit", "VOC2012"),
    },
    "2011": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        "filename": "VOCtrainval_25-May-2011.tar",
        "md5": "6c3384ef61512963050cb5d687e5bf1e",
        "base_dir": os.path.join("TrainVal", "VOCdevkit", "VOC2011"),
    },
    "2010": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "filename": "VOCtrainval_03-May-2010.tar",
        "md5": "da459979d0c395079b5c75ee67908abb",
        "base_dir": os.path.join("VOCdevkit", "VOC2010"),
    },
    "2009": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        "filename": "VOCtrainval_11-May-2009.tar",
        "md5": "59065e4b188729180974ef6572f6a212",
        "base_dir": os.path.join("VOCdevkit", "VOC2009"),
    },
    "2008": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "2629fa636546599198acfcfbfcf1904a",
        "base_dir": os.path.join("VOCdevkit", "VOC2008"),
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
}


class VOCSegmentation(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        year="2012",
        image_set="train",
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
        n_samples=None,
    ):
        super(VOCSegmentation, self).__init__(
            root, transforms, transform, target_transform
        )
        self.year = year
        self.url = DATASET_YEAR_DICT[year]["url"]
        self.filename = DATASET_YEAR_DICT[year]["filename"]
        self.md5 = DATASET_YEAR_DICT[year]["md5"]
        valid_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)
        base_dir = DATASET_YEAR_DICT[year]["base_dir"]
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, "JPEGImages")
        mask_dir = os.path.join(voc_root, "SegmentationClass")

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        splits_dir = os.path.join(voc_root, "ImageSets/Segmentation")

        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        if n_samples is not None:
            file_names = file_names[:n_samples]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        year="2012",
        image_set="train",
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super(VOCDetection, self).__init__(
            root, transforms, transform, target_transform
        )
        self.year = year
        self.url = DATASET_YEAR_DICT[year]["url"]
        self.filename = DATASET_YEAR_DICT[year]["filename"]
        self.md5 = DATASET_YEAR_DICT[year]["md5"]
        valid_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)

        base_dir = DATASET_YEAR_DICT[year]["base_dir"]
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, "JPEGImages")
        annotation_dir = os.path.join(voc_root, "Annotations")

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        splits_dir = os.path.join(voc_root, "ImageSets/Main")

        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [
            os.path.join(annotation_dir, x + ".xml") for x in file_names
        ]
        assert len(self.images) == len(self.annotations)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag: {
                    ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()
                }
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)


""" https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/augmentations/augmentations.py """
import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True

        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


from torchvision.transforms import ToTensor, Normalize


class ImagesToTensors(ToTensor):
    def __call__(self, img, mask):
        img_tt = super().__call__(img)
        mask_tt = super().__call__(mask)
        return (img_tt, mask_tt)


def generate_datasets(parameters={}):
    data_transform = Compose(
        [
            RandomCrop(513),
            ImagesToTensors(),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = VOCSegmentation(
        root=".",
        year="2012",
        image_set="train",
        download=False,
        transforms=data_transform,
        n_samples=10,
    )
    val_dataset = VOCSegmentation(
        root=".",
        year="2012",
        image_set="val",
        download=False,
        transforms=data_transform,
        n_samples=10,
    )
    return train_dataset, val_dataset


""" Copied from:
https://github.com/meetshah1995/pytorch-semseg/blob/801fb200547caa5b0d91b8dde56b837da029f746/ptsemseg/loader/pascal_voc_loader.py
"""

# import os
# from os.path import join as pjoin
# import collections
# import json
# import torch
# import numpy as np
# import scipy.misc as m
# import scipy.io as io
# import matplotlib.pyplot as plt
# import glob
#
# from PIL import Image
# from tqdm import tqdm
# from torch.utils import data
# from torchvision import transforms
#
#
# class pascalVOCLoader(data.Dataset):
#     """Data loader for the Pascal VOC semantic segmentation dataset.
#
#     Annotations from both the original VOC data (which consist of RGB images
#     in which colours map to specific classes) and the SBD (Berkely) dataset
#     (where annotations are stored as .mat files) are converted into a common
#     `label_mask` format.  Under this format, each mask is an (M,N) array of
#     integer values from 0 to 21, where 0 represents the background class.
#
#     The label masks are stored in a new folder, called `pre_encoded`, which
#     is added as a subdirectory of the `SegmentationClass` folder in the
#     original Pascal VOC data layout.
#
#     A total of five data splits are provided for working with the VOC data:
#         train: The original VOC 2012 training data - 1464 images
#         val: The original VOC 2012 validation data - 1449 images
#         trainval: The combination of `train` and `val` - 2913 images
#         train_aug: The unique images present in both the train split and
#                    training images from SBD: - 8829 images (the unique members
#                    of the result of combining lists of length 1464 and 8498)
#         train_aug_val: The original VOC 2012 validation data minus the images
#                    present in `train_aug` (This is done with the same logic as
#                    the validation set used in FCN PAMI paper, but with VOC 2012
#                    rather than VOC 2011) - 904 images
#     """
#
#     def __init__(
#         self,
#         root,
#         sbd_path=None,
#         split="train_aug",
#         is_transform=False,
#         img_size=512,
#         augmentations=None,
#         img_norm=True,
#         test_mode=False,
#     ):
#         self.root = root
#         self.sbd_path = sbd_path
#         self.split = split
#         self.is_transform = is_transform
#         self.augmentations = augmentations
#         self.img_norm = img_norm
#         self.test_mode = test_mode
#         self.n_classes = 21
#         self.mean = np.array([104.00699, 116.66877, 122.67892])
#         self.files = collections.defaultdict(list)
#         self.img_size = (
#             img_size if isinstance(img_size, tuple) else (img_size, img_size)
#         )
#
#         if not self.test_mode:
#             for split in ["train", "val", "trainval"]:
#                 path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
#                 file_list = tuple(open(path, "r"))
#                 file_list = [id_.rstrip() for id_ in file_list]
#                 self.files[split] = file_list
#             self.setup_annotations()
#
#         self.tf = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#
#     def __len__(self):
#         return len(self.files[self.split])
#
#     def __getitem__(self, index):
#         im_name = self.files[self.split][index]
#         im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
#         lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
#         im = Image.open(im_path)
#         lbl = Image.open(lbl_path)
#         if self.augmentations is not None:
#             im, lbl = self.augmentations(im, lbl)
#         if self.is_transform:
#             im, lbl = self.transform(im, lbl)
#         return im, lbl
#
#     def transform(self, img, lbl):
#         if self.img_size == ("same", "same"):
#             pass
#         else:
#             img = img.resize(
#                 (self.img_size[0], self.img_size[1])
#             )  # uint8 with RGB mode
#             lbl = lbl.resize((self.img_size[0], self.img_size[1]))
#         img = self.tf(img)
#         lbl = torch.from_numpy(np.array(lbl)).long()
#         lbl[lbl == 255] = 0
#         return img, lbl
#
#     def get_pascal_labels(self):
#         """Load the mapping that associates pascal classes with label colors
#
#         Returns:
#             np.ndarray with dimensions (21, 3)
#         """
#         return np.asarray(
#             [
#                 [0, 0, 0],
#                 [128, 0, 0],
#                 [0, 128, 0],
#                 [128, 128, 0],
#                 [0, 0, 128],
#                 [128, 0, 128],
#                 [0, 128, 128],
#                 [128, 128, 128],
#                 [64, 0, 0],
#                 [192, 0, 0],
#                 [64, 128, 0],
#                 [192, 128, 0],
#                 [64, 0, 128],
#                 [192, 0, 128],
#                 [64, 128, 128],
#                 [192, 128, 128],
#                 [0, 64, 0],
#                 [128, 64, 0],
#                 [0, 192, 0],
#                 [128, 192, 0],
#                 [0, 64, 128],
#             ]
#         )
#
#     def encode_segmap(self, mask):
#         """Encode segmentation label images as pascal classes
#
#         Args:
#             mask (np.ndarray): raw segmentation label image of dimension
#               (M, N, 3), in which the Pascal classes are encoded as colours.
#
#         Returns:
#             (np.ndarray): class map with dimensions (M,N), where the value at
#             a given location is the integer denoting the class index.
#         """
#         mask = mask.astype(int)
#         label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
#         for ii, label in enumerate(self.get_pascal_labels()):
#             label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
#         label_mask = label_mask.astype(int)
#         return label_mask
#
#     def decode_segmap(self, label_mask, plot=False):
#         """Decode segmentation class labels into a color image
#
#         Args:
#             label_mask (np.ndarray): an (M,N) array of integer values denoting
#               the class label at each spatial location.
#             plot (bool, optional): whether to show the resulting color image
#               in a figure.
#
#         Returns:
#             (np.ndarray, optional): the resulting decoded color image.
#         """
#         label_colours = self.get_pascal_labels()
#         r = label_mask.copy()
#         g = label_mask.copy()
#         b = label_mask.copy()
#         for ll in range(0, self.n_classes):
#             r[label_mask == ll] = label_colours[ll, 0]
#             g[label_mask == ll] = label_colours[ll, 1]
#             b[label_mask == ll] = label_colours[ll, 2]
#         rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
#         rgb[:, :, 0] = r / 255.0
#         rgb[:, :, 1] = g / 255.0
#         rgb[:, :, 2] = b / 255.0
#         if plot:
#             plt.imshow(rgb)
#             plt.show()
#         else:
#             return rgb
#
#     def setup_annotations(self):
#         """Sets up Berkley annotations by adding image indices to the
#         `train_aug` split and pre-encode all segmentation labels into the
#         common label_mask format (if this has not already been done). This
#         function also defines the `train_aug` and `train_aug_val` data splits
#         according to the description in the class docstring
#         """
#         sbd_path = self.sbd_path
#         target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
#         if not os.path.exists(target_path):
#             os.makedirs(target_path)
#         path = pjoin(sbd_path, "dataset/train.txt")
#         sbd_train_list = tuple(open(path, "r"))
#         sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
#         train_aug = self.files["train"] + sbd_train_list
#
#         # keep unique elements (stable)
#         train_aug = [
#             train_aug[i] for i in sorted(np.unique(train_aug, return_index=True)[1])
#         ]
#         self.files["train_aug"] = train_aug
#         set_diff = set(self.files["val"]) - set(train_aug)  # remove overlap
#         self.files["train_aug_val"] = list(set_diff)
#
#         pre_encoded = glob.glob(pjoin(target_path, "*.png"))
#         expected = np.unique(self.files["train_aug"] + self.files["val"]).size
#
#         if len(pre_encoded) != expected:
#             print("Pre-encoding segmentation masks...")
#             for ii in tqdm(sbd_train_list):
#                 lbl_path = pjoin(sbd_path, "dataset/cls", ii + ".mat")
#                 data = io.loadmat(lbl_path)
#                 lbl = data["GTcls"][0]["Segmentation"][0].astype(np.int32)
#                 lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
#                 m.imsave(pjoin(target_path, ii + ".png"), lbl)
#
#             for ii in tqdm(self.files["trainval"]):
#                 fname = ii + ".png"
#                 lbl_path = pjoin(self.root, "SegmentationClass", fname)
#                 lbl = self.encode_segmap(m.imread(lbl_path))
#                 lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
#                 m.imsave(pjoin(target_path, fname), lbl)
#
#         assert expected == 9733, "unexpected dataset sizes"
#
#
# # Leave code for debugging purposes
# # import ptsemseg.augmentations as aug
# # if __name__ == '__main__':
# # # local_path = '/home/meetshah1995/datasets/VOCdevkit/VOC2012/'
# # bs = 4
# # augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
# # dst = pascalVOCLoader(root=local_path, is_transform=True, augmentations=augs)
# # trainloader = data.DataLoader(dst, batch_size=bs)
# # for i, data in enumerate(trainloader):
# # imgs, labels = data
# # imgs = imgs.numpy()[:, ::-1, :, :]
# # imgs = np.transpose(imgs, [0,2,3,1])
# # f, axarr = plt.subplots(bs, 2)
# # for j in range(bs):
# # axarr[j][0].imshow(imgs[j])
# # axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
# # plt.show()
# # a = raw_input()
# # if a == 'ex':
# # break
# # else:
# # plt.close()
#
#
# def generate_datasets(parameters={}):
#     data_transform = Compose([RandomCrop(size=513)])
#     voc_params = dict(
#         root="./VOCdevkit/VOC2012",
#         sbd_path=None,
#         split="train",
#         is_transform=True,
#         img_size=513,
#         augmentations=data_transform,
#         img_norm=True,
#         test_mode=False,
#     )
#     train_dataset = pascalVOCLoader(**voc_params)
#     voc_val_params = voc_params.copy()
#     voc_val_params.update(dict(split="val"))
#     val_dataset = pascalVOCLoader(**voc_val_params)
#     return train_dataset, val_dataset

from pipelinex import CrossEntropyLoss2d


class VocCrossEntropyLoss2d(CrossEntropyLoss2d):
    def forward(self, input_dict, mask):
        img = input_dict["out"]
        mask = mask.type(torch.LongTensor)
        return super().forward(img, mask)
