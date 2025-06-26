import GCL.augmentors as A
import nibabel as nib
import numpy as np
import os
import pandas as pd
import scipy.io as scio
import torch_geometric.transforms as T
import torch_geometric
import torch

from functools import lru_cache
from src.utils import create_corr, corr_to_graph
from torch_geometric.data import Data, Batch
from torch_geometric.loader import CachedLoader
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
        data_dir="./Data/REST-meta-MDD/fMRI/AAL",
        metadata_path="./Data/REST-meta-MDD/metadata.csv",
        split="full",  # Options: 'train', 'dev', 'test', 'full'
        label="mdd",  # "mdd", "sex", "age"
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

        if label == "mdd":
            self.labels = torch.from_numpy(metadata.label.values)
        elif label == "sex":
            self.labels = torch.from_numpy(metadata.sex.values - 1)
        elif label == "age":
            self.labels = torch.from_numpy(metadata.age.values)

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

    def _create_cache(self, data_dir):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        for subid in self.ids:
            edge_index_path = os.path.join(self.cache_path, f"{subid}_edge_index.npy")
            node_features_path = os.path.join(
                self.cache_path, f"{subid}_node_features.npy"
            )

            if os.path.exists(edge_index_path) and os.path.exists(node_features_path):
                continue

            filepath = os.path.join(data_dir, f"{subid}.npy")
            raw_signals = np.load(filepath)
            corr = create_corr(raw_signals)
            node_features, edge_index = corr_to_graph(corr)

            np.save(edge_index_path, edge_index)
            np.save(node_features_path, node_features)

    def _cache_exists(self):
        return False
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

            edge_index = torch.from_numpy(np.load(edge_index_path))
            node_features = torch.from_numpy(np.load(node_features_path))
            edge_indices_list.append(edge_index)
            node_features_list.append(node_features)

        edge_indices = torch.stack(edge_indices_list)
        node_features = torch.stack(node_features_list)
        return edge_indices, node_features


class ABIDEfMRIDataset(Dataset):
    def __init__(self, data_dir="./Data/QLiData/ABIDE", split="full"):
        super().__init__()
        self.num_classes = None
        self.dataset = self.prepare_data(data_dir, split)
        self.num_nodes = self.dataset[0].x.shape[0] if self.dataset else 0

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def prepare_data(self, data_dir, split):
        dataset = []

        for file_name in os.listdir(data_dir):
            if file_name.endswith(".npy"):
                study_id = file_name.split(".")[0]

                if "_" in study_id:
                    study_id = study_id.split("_")[-1]

                time_series = np.load(os.path.join(data_dir, file_name))
                conn_matrix = create_corr(time_series.T)
                node_features, edge_index = corr_to_graph(conn_matrix)
                graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                )
                dataset.append(graph)

        train, rest = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42,
        )
        dev, test = train_test_split(
            rest,
            test_size=0.5,
            random_state=42,
        )

        if split == "train":
            return train
        elif split == "dev":
            return dev
        elif split == "test":
            return test
        elif split == "full":
            return dataset


class ABIDEfMRIDataset2(Dataset):
    def __init__(
        self,
        data_dir="./Data/ABIDE",
        split="full",  # Options: 'train', 'dev', 'test', 'full'
    ):
        # Data is available at google drive (https://drive.google.com/drive/folders/1EkvBOoXF0MB2Kva9l4GQbuWX25Yp81a8?usp=sharing).
        super().__init__()
        self.dataset = self.prepare_data(data_dir, split)

    def prepare_data(self, data_dir, split):
        data_path = os.path.join(data_dir, "md_AAL_0.4.mat")
        data = scio.loadmat(data_path)
        # graph_struct = data["graph_struct"][0]
        label = data["label"]

        train, test = train_test_split(
            pd.DataFrame(label), test_size=0.2, random_state=42, stratify=label
        )
        dev, test = train_test_split(
            pd.DataFrame(label[test.index]),
            test_size=0.5,
            random_state=42,
            stratify=label[test.index],
        )

        dataset = []

        for graph_index in range(len(data["label"])):
            graph_struct = data["graph_struct"][0]
            edge = torch.Tensor(graph_struct[graph_index]["edge"])
            ROI = torch.Tensor(graph_struct[graph_index]["ROI"])

            y = torch.Tensor(label[graph_index])
            A = torch.sparse_coo_tensor(
                indices=edge[:, :2].t().long(),
                values=edge[:, -1]
                .reshape(
                    -1,
                )
                .float(),
                size=(116, 116),
            )
            G = (A.t() + A).coalesce()

            graph = Data(
                x=ROI.reshape(-1, 116).float(),
                edge_index=G.indices().reshape(2, -1).long(),
                edge_attr=G.values().reshape(-1, 1).float(),
                y=y.long(),
            )
            dataset.append(graph)

        if split == "train":
            return [dataset[i] for i in train.index]
        elif split == "dev":
            return [dataset[i] for i in dev.index]
        elif split == "test":
            return [dataset[i] for i in test.index]
        elif split == "full":
            return dataset
        else:
            raise ValueError("Invalid split. Use 'train', 'dev', 'test' or 'full'.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class COBREfMRIDataset(Dataset):
    FILE_STRUCTURES = {
        "UCLA_COBRE_ALL116.mat": ["UCLA_HC", "UCLA_SZ"],
        "COBRE_HC_SZ_AAL116_Updated.mat": [
            "COBRE_1_SC_NS_",
            "COBRE_2_SC_S_",
            "COBRE_3_HC_NS_",
            "COBRE_4_HC_S_",
        ],
    }

    def __init__(
        self,
        data_dir="./Data/COBRE",
        split="full",  # Options: 'train', 'dev', 'test', 'full'
    ):
        super().__init__()
        self.dataset = self.prepare_data(data_dir, split)

    def prepare_data(self, data_dir, split):
        data, sites = self._preprocess_raw_signal(data_dir)
        train, test = train_test_split(
            pd.DataFrame(sites), test_size=0.2, random_state=42, stratify=sites
        )
        dev, test = train_test_split(
            pd.DataFrame(sites[test.index]),
            test_size=0.5,
            random_state=42,
            stratify=sites[test.index],
        )

        dataset = []

        for signal in data:
            corr = create_corr(signal)
            node_features, edge_index = corr_to_graph(corr)
            graph = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
            )
            dataset.append(graph)

        if split == "train":
            return [dataset[i] for i in train.index]
        elif split == "dev":
            return [dataset[i] for i in dev.index]
        elif split == "test":
            return [dataset[i] for i in test.index]
        elif split == "full":
            return dataset
        else:
            raise ValueError("Invalid split. Use 'train', 'dev', 'test' or 'full'.")

    def _preprocess_raw_signal(self, data_dir):
        corr_matrices = []
        sites = []

        for file_name, sub in self.FILE_STRUCTURES.items():
            data_path = os.path.join(data_dir, file_name)
            data = scio.loadmat(data_path)
            for s in sub:
                for signal in data[s]:
                    corr_matrix = create_corr(signal[0])
                    corr_matrices.append(corr_matrix)
                    sites.append(s)

        return np.array(corr_matrices), np.array(sites)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# class HCPfMRIDataset(Dataset):
#     def __init__(self, data_dir="./Data/HCP1200/", split="full"):
#         super().__init__()
#         self.dataset = self.prepare_data(data_dir, split)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

#     def __len__(self):
#         return len(self.dataset)

#     def prepare_data(self, data_dir, split):
#         dataset = []
#         data_dir = os.path.join(data_dir, "AAL116")

#         for file_name in os.listdir(data_dir):
#             if file_name.endswith(".npy"):
#                 conn_matrix = np.load(os.path.join(data_dir, file_name))
#                 node_features, edge_index = corr_to_graph(conn_matrix)
#                 graph = Data(
#                     x=torch.tensor(node_features, dtype=torch.float),
#                     edge_index=torch.tensor(edge_index, dtype=torch.long),
#                 )
#                 dataset.append(graph)

#         if split == "full":
#             return dataset
#         else:
#             raise ValueError(
#                 "Invalid split. Use 'full' for HCP dataset as it does not have train/dev/test splits."
#             )


class HCPfMRIDataset(Dataset):
    def __init__(self, data_dir="./Data/QLiData/HCP", split="full"):
        super().__init__()
        self.num_classes = None
        self.dataset = self.prepare_data(data_dir, split)
        self.num_nodes = self.dataset[0].x.shape[0] if self.dataset else 0

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def prepare_data(self, data_dir, split):
        dataset = []

        for file_name in os.listdir(data_dir):
            if file_name.endswith(".npy"):
                study_id = file_name.split(".")[0]

                if "_" in study_id:
                    study_id = study_id.split("_")[-1]

                time_series = np.load(os.path.join(data_dir, file_name))
                conn_matrix = create_corr(time_series.T)
                node_features, edge_index = corr_to_graph(conn_matrix)
                graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                )
                dataset.append(graph)

        train, rest = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42,
        )
        dev, test = train_test_split(
            rest,
            test_size=0.5,
            random_state=42,
        )

        if split == "train":
            return train
        elif split == "dev":
            return dev
        elif split == "test":
            return test
        elif split == "full":
            return dataset


class AOMICfMRIDataset(Dataset):
    def __init__(self, data_dir="./Data/AOMIC/", split="full"):
        super().__init__()
        self.dataset = self.prepare_data(data_dir, split)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def prepare_data(self, data_dir, split):
        dataset = []
        data_dir = os.path.join(data_dir, "AAL116")

        for file_name in os.listdir(data_dir):
            if file_name.endswith(".npy"):
                conn_matrix = np.load(os.path.join(data_dir, file_name))
                node_features, edge_index = corr_to_graph(conn_matrix)
                graph = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                )
                dataset.append(graph)

        if split == "full":
            return dataset
        else:
            raise ValueError(
                "Invalid split. Use 'full' for HCP dataset as it does not have train/dev/test splits."
            )


class BSNIPfMRIDataset(Dataset):
    def __init__(self, data_dir="./Data/QLiData/BSNIP", split="full"):
        super().__init__()
        self.num_classes = None
        self.dataset = self.prepare_data(data_dir, split)
        self.num_nodes = self.dataset[0].x.shape[0] if self.dataset else 0

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def prepare_data(self, data_dir, split):
        dataset = []
        meta = pd.read_csv(os.path.join(data_dir, "bsnip_label.csv"), index_col=1)
        self.num_classes = len(meta.group.unique())
        lab_to_id = {lab: i for i, lab in enumerate(meta.group.unique())}

        for file_name in os.listdir(data_dir):
            if file_name.endswith(".npy"):
                study_id = file_name.split(".")[0]

                if "_" in study_id:
                    study_id = study_id.split("_")[-1]

                group = meta.loc[study_id].group
                label = lab_to_id[group]

                time_series = np.load(os.path.join(data_dir, file_name))
                conn_matrix = create_corr(time_series.T)
                node_features, edge_index = corr_to_graph(conn_matrix)
                graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                    y=torch.tensor([label], dtype=torch.long),
                )
                dataset.append(graph)

        train, rest = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42,
            stratify=[data.y.item() for data in dataset],
        )
        dev, test = train_test_split(
            rest,
            test_size=0.5,
            random_state=42,
            stratify=[data.y.item() for data in rest],
        )

        if split == "train":
            return train
        elif split == "dev":
            return dev
        elif split == "test":
            return test
        elif split == "full":
            return dataset

    @property
    def labels(self):
        return torch.tensor([data.y.item() for data in self.dataset], dtype=torch.long)


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
        data_dir="./Data/REST-meta-MDD/REST-meta-MDD-VBM-Phase1-Sharing",
        metadata_path="./Data/REST-meta-MDD/metadata.csv",
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
        data_t = torch.from_numpy(data).to(NP_TO_TORCH_DTYPES[self.dtype])
        return data_t, self.labels[idx]

    def _inmemory_load_data(self, idx):
        return self.data[idx], self.labels[idx]

    def _load_data_to_memory(self):
        self.data = torch.zeros(self.dshape, dtype=NP_TO_TORCH_DTYPES[self.dtype])
        for i, subid in enumerate(self.ids):
            filepath = os.path.join(self.cache_path, f"{subid}.npy")
            self.data[i] = torch.from_numpy(np.load(filepath))

    def _get_data_shape(self, data_dir, imgtypes):
        subid = self.ids[0]
        imgtype = imgtypes[0]
        filepath = os.path.join(data_dir, imgtype, f"{subid}.nii.gz")
        image_shape = nib.load(filepath).get_fdata().shape
        return (len(self.ids), len(imgtypes), *image_shape)


class RESTJointDataset(Dataset):
    def __init__(
        self,
        fmri_data_dir="./Data/REST-meta-MDD/fMRI/AAL",
        smri_data_dir="./Data/REST-meta-MDD/REST-meta-MDD-VBM-Phase1-Sharing",
        metadata_path="./Data/REST-meta-MDD/metadata.csv",
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
        labels_batch = torch.from_numpy(labels)

        return imgs_batch, graphs_batch, labels_batch


class GPSConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.num_nodes = datasets[0].num_nodes if datasets else 0
        self.pe_transform = T.AddRandomWalkPE(walk_length=60, attr_name="pe")
        self.cache = self._prepare_cache()

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return super().__len__()

    def _prepare_cache(self):
        cache = {}

        for i in range(len(self)):
            cache[i] = self.pe_transform(super().__getitem__(i))

        return cache

    @property
    def num_classes(self):
        return self.datasets[0].num_classes if self.datasets else 0


class BTDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        super().__init__(
            dataset, batch_size, shuffle, collate_fn=self._batch_data, **kwargs
        )
        self.aug1 = self._construct_aug()
        self.aug2 = self._construct_aug()

    def _batch_data(self, x):
        x1_batch, x2_batch = [], []

        for d in x:
            x1 = Data(*self.aug1(d.x, d.edge_index))
            x2 = Data(*self.aug2(d.x, d.edge_index))

            if hasattr(d, "pe"):
                x1["pe"] = d["pe"]
                x2["pe"] = d["pe"]

            x1_batch.append(x1)
            x2_batch.append(x2)

        x1_batch = Batch.from_data_list(x1_batch)
        x2_batch = Batch.from_data_list(x2_batch)

        return x1_batch, x2_batch

    def _construct_aug(self):
        return A.RandomChoice(
            [
                A.RWSampling(num_seeds=1000, walk_length=10),
                A.NodeDropping(pn=0.1),
                A.FeatureMasking(pf=0.1),
                # A.EdgeAdding(pe=0.1),
                A.EdgeRemoving(pe=0.1),
            ],
            num_choices=1,
        )


class HFMCADataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        num_views,
        batch_size=1,
        shuffle=False,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        super().__init__(
            dataset, batch_size, shuffle, collate_fn=self._batch_data, **kwargs
        )
        self.augs = [self._construct_aug() for _ in range(num_views)]

    def _batch_data(self, x):
        views = []

        for aug in self.augs:
            for d in x:
                augmented = Data(*aug(d.x, d.edge_index))

                if hasattr(d, "pe"):
                    augmented["pe"] = d["pe"]

                views.append(augmented)

        views = Batch.from_data_list(views)

        return views

    def _construct_aug(self):
        return A.RandomChoice(
            [
                A.RWSampling(num_seeds=1000, walk_length=10),
                A.NodeDropping(pn=0.1),
                A.FeatureMasking(pf=0.1),
                # A.EdgeAdding(pe=0.1),
                A.EdgeRemoving(pe=0.1),
            ],
            num_choices=1,
        )


class GPSCachedLoader(CachedLoader):
    def __init__(self, dataset, batch_size, num_workers, shuffle=False):
        loader = torch_geometric.loader.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        transform = T.AddRandomWalkPE(walk_length=60, attr_name="pe")
        super().__init__(loader, transform=transform)

        self.dataset = dataset
        self.batch_size = loader.batch_size
