import nibabel as nib
import numpy as np
import os
import pandas as pd
import torch

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

NP_TO_TORCH_DTYPES = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}

# ALLOWED_SPLITS = {
#     # Regex for matching all columns excluding the ones in the dev and test splits
#     # using negative pattern matching
#     "train": r"(?!(S4|S8|S15|S17)-).*",
#     "dev": r"((S4|S8)-)",
#     "test": r"((S15|S17)-)",
#     "full": r".*",
# }


# Dataset for fMRI data that stores everything in memory
class RESTfMRIDataset(Dataset):
    def __init__(
        self,
        data_dir="./REST-meta-MDD/fMRI/AAL",
        metadata_path="./REST-meta-MDD/metadata.csv",
        split="full",
        cache_path=None,
    ):
        super().__init__()
        metadata = self._load_metadata(metadata_path, split)
        self.ids = metadata.subID

        if cache_path is None:
            self.cache_path = os.path.join("cache", data_dir)
        else:
            self.cache_path = os.path.join(cache_path, data_dir)

        if not self._cache_exists():
            self._create_cache(data_dir)

        self.edge_indices, self.node_features = self._load_cache()
        self.num_samples = self.node_features.shape[0]
        self.num_nodes = self.node_features.shape[1]
        self.labels = torch.from_numpy(metadata.label.values)

    def _load_metadata(self, metadata_path, split):
        metadata = pd.read_csv(metadata_path)
        sites = metadata["subID"].str.extract(r"S(\d+)-")[0].astype(int)
        train, test = train_test_split(
            metadata, test_size=0.2, random_state=42, stratify=sites
        )
        dev, test = train_test_split(
            test, test_size=0.5, random_state=42, stratify=sites[test.index]
        )

        if split == "train":
            self.sites = sites[train.index]
            return train.reset_index(drop=True)
        elif split == "dev":
            self.sites = sites[dev.index]
            return dev.reset_index(drop=True)
        elif split == "test":
            self.sites = sites[test.index]
            return test.reset_index(drop=True)
        elif split == "full":
            self.sites = sites
            return metadata.reset_index(drop=True)
        else:
            raise ValueError("Invalid split. Use 'train', 'dev', 'test' or 'full'.")

        # cond = metadata.subID.str.match(ALLOWED_SPLITS[split])
        # return metadata[cond].reset_index(drop=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return Data(
            x=self.node_features[idx],
            edge_index=self.edge_indices[idx],
            y=self.labels[idx],
        )

    def _create_corr(self, data):
        eps = 1e-16
        R = np.corrcoef(data)
        R[np.isnan(R)] = 0
        R = R - np.diag(np.diag(R))
        R[R >= 1] = 1 - eps
        corr = 0.5 * np.log((1 + R) / (1 - R))
        return corr

    def _create_cache(self, data_dir):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        num_edges = None

        for subid in self.ids:
            edge_index_path = os.path.join(self.cache_path, f"{subid}_edge_index.npy")
            node_features_path = os.path.join(
                self.cache_path, f"{subid}_node_features.npy"
            )

            if os.path.exists(edge_index_path) and os.path.exists(node_features_path):
                continue

            # Read the raw signals and create correlation matrix
            filepath = os.path.join(data_dir, f"{subid}.npy")
            raw_signals = np.load(filepath)
            corr = self._create_corr(raw_signals)

            # Transform correlation matrix to adjacency matrix
            node_features = torch.tensor(corr, dtype=torch.float32)
            topk = node_features.reshape(-1)
            topk, _ = torch.sort(abs(topk), dim=0, descending=True)
            threshold = topk[int(node_features.shape[0] ** 2 / 20 * 2)]
            adj = (torch.abs(node_features) >= threshold).to(int)
            edge_index = dense_to_sparse(adj)[0]

            if num_edges is None:
                num_edges = edge_index.shape[1]

            edge_index = edge_index[:, :num_edges]

            np.save(edge_index_path, edge_index)
            np.save(node_features_path, node_features)

    def _cache_exists(self):
        if not os.path.exists(self.cache_path):
            return False

        for subid in self.ids:
            if not os.path.exists(os.path.join(self.cache_path, f"{subid}.npy")):
                return False

        return True

    def _load_cache(self):
        edge_indices_list = []
        node_features_list = []

        for subid in self.ids:
            edge_index_path = os.path.join(self.cache_path, f"{subid}_edge_index.npy")
            node_features_path = os.path.join(
                self.cache_path, f"{subid}_node_features.npy"
            )

            edge_index = torch.tensor(np.load(edge_index_path))
            node_features = torch.tensor(np.load(node_features_path))
            edge_indices_list.append(edge_index)
            node_features_list.append(node_features)

        edge_indices = torch.stack(edge_indices_list)
        node_features = torch.stack(node_features_list)
        return edge_indices, node_features


class RESTsMRIDataset(Dataset):
    IMAGE_TYPES = [
        "c1",  # Gray matter density in native space
        "c2",  # White matter density in native space
        "c3",  # Cerebrospinal fluid density in native space
        "wc1",  # Gray matter density in MNI space
        "wc2",  # White matter density in MNI space
        "wc3",  # Cerebrospinal fluid density in MNI space
        "mwc1",  # Gray matter volume in MNI space
        "mwc2",  # White matter volume in MNI space
        "mwc3",  # Cerebrospinal fluid volume density in MNI space
    ]

    def __init__(
        self,
        data_dir="./REST-meta-MDD/REST-meta-MDD-VBM-Phase1-Sharing",
        metadata_path="./REST-meta-MDD/metadata.csv",
        imgtypes=[
            "wc1"
        ],  # Can be any combination of allowed imgtypes (["wc1", "mwc1"...])
        split="full",  # Can be "train", "dev", "test" or "full"
        normalize=True,  # Normalize images
        dtype=np.float32,  # Reduce to save memory
        inmemory=False,  # Load everything in memory
        cache_path=None,
    ):
        super().__init__()
        imgtypes = sorted(imgtypes)
        metadata = self._load_metadata(metadata_path, split)
        self.ids = metadata.subID
        self.normalize = normalize
        self.dtype = dtype

        if cache_path is None:
            self.cache_path = os.path.join("cache", data_dir, "-".join(imgtypes))
        else:
            self.cache_path = os.path.join(cache_path, data_dir, "-".join(imgtypes))

        self.labels = torch.from_numpy(metadata.label.values)
        self.dshape = self._get_data_shape(data_dir, imgtypes)

        if not self._cache_exists():
            self._create_cache(data_dir, imgtypes)

        if inmemory:
            self._load_data_to_memory()
            self.load_data_fn = self._inmemory_load_data
        else:
            self.load_data_fn = self._lazy_load_data

        self.num_samples = len(self.ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.load_data_fn(idx)

    def _load_metadata(self, metadata_path, split):
        metadata = pd.read_csv(metadata_path)
        sites = metadata["subID"].str.extract(r"S(\d+)-")[0].astype(int)
        train, test = train_test_split(
            metadata, test_size=0.2, random_state=42, stratify=sites
        )
        dev, test = train_test_split(
            test, test_size=0.5, random_state=42, stratify=sites[test.index]
        )

        if split == "train":
            self.sites = sites[train.index]
            return train.reset_index(drop=True)
        elif split == "dev":
            self.sites = sites[dev.index]
            return dev.reset_index(drop=True)
        elif split == "test":
            self.sites = sites[test.index]
            return test.reset_index(drop=True)
        elif split == "full":
            self.sites = sites
            return metadata.reset_index(drop=True)
        else:
            raise ValueError("Invalid split. Use 'train', 'dev', 'test' or 'full'.")

        # cond = metadata.subID.str.match(ALLOWED_SPLITS[split])
        # return metadata[cond].reset_index(drop=True)

    def _cache_exists(self):
        if not os.path.exists(self.cache_path):
            return False

        for subid in self.ids:
            if not os.path.exists(os.path.join(self.cache_path, f"{subid}.npy")):
                return False

        return True

    def _create_cache(self, data_dir, imgtypes):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        for subid in self.ids:
            if os.path.exists(os.path.join(self.cache_path, f"{subid}.npy")):
                continue

            dpoint = np.zeros(self.dshape[1:], dtype=self.dtype)

            for i, split in enumerate(imgtypes):

                filepath = os.path.join(data_dir, split, f"{subid}.nii.gz")
                image = nib.load(filepath).get_fdata()

                if self.normalize:
                    image -= image.mean()

                dpoint[i] = image

            np.save(os.path.join(self.cache_path, f"{subid}.npy"), dpoint)

    def _lazy_load_data(self, idx):
        filename = self.ids[idx]
        data = np.load(os.path.join(self.cache_path, f"{filename}.npy"))
        data_t = torch.tensor(data, dtype=NP_TO_TORCH_DTYPES[self.dtype])
        return data_t, self.labels[idx]

    def _inmemory_load_data(self, idx):
        return self.data[idx], self.labels[idx]

    def _load_data_to_memory(self):
        self.data = torch.zeros(self.dshape, dtype=NP_TO_TORCH_DTYPES[self.dtype])
        for i, subid in enumerate(self.ids):
            filepath = os.path.join(self.cache_path, f"{subid}.npy")
            self.data[i] = torch.tensor(np.load(filepath))

    def _get_data_shape(self, data_dir, imgtypes):
        subid = self.ids[0]
        imgtype = imgtypes[0]
        filepath = os.path.join(data_dir, imgtype, f"{subid}.nii.gz")
        image_shape = nib.load(filepath).get_fdata().shape
        return (len(self.ids), len(imgtypes), *image_shape)


# class RESTsMRIDinoDataset(RESTsMRIDataset):
#     def __getitem__(self, idx):
#         x, y = self.load_data_fn(idx)
#         x = x.unbind(dim=-1)
#         x = torch.concat(x, dim=-1)
#         x = x.unbind(dim=0)
#         x = torch.concat(x, dim=-1)
#         h, w = x.shape
#         print(x.shape)
#         x = x[: h // 14 * 14, : w // 14 * 14]
#         x = x.unsqueeze(0).expand(3, -1, -1)

#         return x, y


class RESTJointDataset(Dataset):
    def __init__(
        self,
        fmri_data_dir="./REST-meta-MDD/fMRI/AAL",
        smri_data_dir="./REST-meta-MDD/REST-meta-MDD-VBM-Phase1-Sharing",
        metadata_path="./REST-meta-MDD/metadata.csv",
        imgtypes=[
            "wc1"
        ],  # Can be any combination of allowed imgtypes (["wc1", "mwc1"...])
        split="full",  # Can be "train", "dev", "test" or "full"
        normalize=True,  # Normalize images
        dtype=np.float32,  # Reduce to save memory
        inmemory=False,  # Load everything in memory
        cache_path=None,
        # dino=False,
    ):
        super().__init__()
        self.fmri = RESTfMRIDataset(
            data_dir=fmri_data_dir,
            metadata_path=metadata_path,
            split=split,
            cache_path=cache_path,
        )

        # if dino:
        #     self.smri = RESTsMRIDinoDataset(
        #         data_dir=smri_data_dir,
        #         metadata_path=metadata_path,
        #         imgtypes=imgtypes,
        #         split=split,
        #         normalize=normalize,
        #         dtype=dtype,
        #         inmemory=inmemory,
        #         cache_path=cache_path,
        #     )
        # else:
        self.smri = RESTsMRIDataset(
            data_dir=smri_data_dir,
            metadata_path=metadata_path,
            imgtypes=imgtypes,
            split=split,
            normalize=normalize,
            dtype=dtype,
            inmemory=inmemory,
            cache_path=cache_path,
        )

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx):
        img, label = self.smri[idx]
        f_data = self.fmri[idx]
        graph = Data(
            x=f_data.x,
            edge_index=f_data.edge_index,
        )

        return img, graph, label


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        # augment=True,
        patch_size=64,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        super().__init__(
            dataset, batch_size, shuffle, collate_fn=self._joint_batch_data, **kwargs
        )
        # self.augment = augment
        self.patch_size = patch_size

    def _joint_batch_data(self, x):
        imgs = [i[0] for i in x]

        # if self.augment:
        #     imgs = self._augment_data(imgs)

        imgs_batch = torch.stack(imgs)

        graphs = [i[1] for i in x]
        labels = [i[2] for i in x]

        graphs_batch = Batch.from_data_list(graphs)
        labels_batch = torch.tensor(labels)

        return imgs_batch, graphs_batch, labels_batch

    # def _augment_data(self, X):
    #     "Apply augmentation"

    #     X_aug = patch_extraction(X, sizePatches=self.patch_size, Npatches=1)
    #     X_aug = aug_batch(X_aug)

    #     return [torch.tensor(x.copy()).to(torch.float32) for x in X_aug]
