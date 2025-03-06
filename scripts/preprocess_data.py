import pandas as pd
import numpy as np
import scipy.io
import os


DATA_DIR = "./REST-meta-MDD"
SUBJ_PATH = "REST-meta-MDD-Phase1-Sharing/Results/ROISignals_FunImgARglobalCWF"
XLSX_SUBJ_FILENAME = "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
# You can comment out splits that you're not interested in
SPLITS = [
    ("AAL", 116),
    ("Harvard-Oxford", 228),
    ("Craddock", 428),
    ("Zalesky", 1408),
    ("Dosenbach", 1568),
    ("GlobalSignal", 1569),
    ("Power264", 1833),
]


# Function used to parse .xlsx metadata into pandas dataframe; currently not used
def preprocess_xlsx_metadata():
    data_mdd = pd.read_excel(
        os.path.join(DATA_DIR, XLSX_SUBJ_FILENAME), sheet_name="MDD", index_col="ID"
    )
    data_controls = pd.read_excel(
        os.path.join(DATA_DIR, XLSX_SUBJ_FILENAME),
        sheet_name="Controls",
        index_col="ID",
    )
    data_mdd["label"] = True
    data_controls["label"] = False
    data_full = pd.concat([data_mdd, data_controls])

    nullable_cols = ["If first episode?", "On medication?"]
    alias_cols = ["first_episode", "on_medication"]

    for col, newcol in zip(nullable_cols, alias_cols):
        nanidx = data_full[col] == -9999.0
        falseidx = (data_full[col] == 2) | (data_full[col] == -1)
        trueidx = data_full[col] == 1

        data_full.loc[falseidx, newcol] = False
        data_full.loc[trueidx, newcol] = True
        data_full.loc[nanidx, newcol] = np.nan

        data_full.drop([col], axis=1, inplace=True)

    data_full.drop(data_full.filter(regex="HAM").columns, axis=1, inplace=True)
    data_full.rename(
        columns={
            "Illness duration (months)": "illness_duration_months",
            "Education (years)": "education_years",
            "Sex": "sex",
            "Age": "age",
        },
        inplace=True,
    )
    data_full["subID"] = data_full.index
    data_full.index = range(len(data_full))
    data_full.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)


def metadata_mat_to_csv():
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    # The Stat_Sub_Info_848MDDvs794NC.mat file comes from https://github.com/Chaogan-Yan/PaperScripts/tree/master/Yan_2019_PNAS
    # A git repository for "Reduced default mode network functional connectivity in patients with recurrent major depressive disorder"
    # It's a file with filtered metadata based on the criterions presented in the paper
    matpath = os.path.join(DATA_DIR, "Stat_Sub_Info_848MDDvs794NC.mat")
    mat = scipy.io.loadmat(matpath)
    data = pd.DataFrame(
        data={
            "sex": mat["Sex"].reshape(-1),
            "age": mat["Age"].reshape(-1),
            "education_years": mat["Edu"].reshape(-1),
            "site": mat["Site"].reshape(-1),
            "motion": mat["Motion"].reshape(-1),
            "label": (mat["Dx"].reshape(-1) > 0).astype(int),
        },
        index=(x[0] for x in mat["SubID"].reshape(-1)),
    )
    data.index.name = "subID"
    data.to_csv(metadata_path)
    return data


def create_fmri_splits():
    if not os.path.exists(os.path.join(DATA_DIR, "fMRI")):
        os.makedirs(os.path.join(DATA_DIR, "fMRI"))

    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    for id in metadata.subID:
        filepath = os.path.join(DATA_DIR, SUBJ_PATH, f"ROISignals_{id}.mat")
        dpoint = scipy.io.loadmat(filepath)["ROISignals"]

        last_split = 0
        for split, split_slice in SPLITS:
            if split_slice > dpoint.shape[1]:
                break

            save_dir = os.path.join(DATA_DIR, "fMRI", f"{split}")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, f"{id}")
            data = dpoint[:, last_split:split_slice].transpose(
                1, 0
            )  # [n_regions, n_timepoints]
            last_split = split_slice
            np.save(save_path, data)


def main():
    # TODO: Add data download script from https://www.scidb.cn/en/detail?dataSetId=cbeb3c7124bf47a6af7b3236a3aaf3a8
    # The scripts expects that both files (REST-meta-MDD-Phase1-Sharing.zip and REST-meta-MDD-VBM-Phase1-Sharing.tar.gz)
    # are extracted in the ./REST-meta-MDD directory.
    # .mat metadata file can be found in https://github.com/Chaogan-Yan/PaperScripts/tree/master/Yan_2019_PNAS/StatsSubInfo
    # and it's also expected to be in the ./REST-meta-MDD directory.
    metadata_mat_to_csv()
    create_fmri_splits()


if __name__ == "__main__":
    main()
