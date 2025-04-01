# mri-eigenencoder
This repository aims to reproduce results from [Learning Cortico-Muscular Dependence through
Orthonormal Decomposition of Density Ratios](https://arxiv.org/pdf/2410.14697) for sMRI and fMRI data with REST-meta-MDD dataset.

## Dataset
The repo expects a directory/soft link named "REST-meta-MDD" with the following structure:

REST-meta-MDD  
├── Stat_Sub_Info_848MDDvs794NC.mat  
├── REST-meta-MDD-Phase1-Sharing  
│   ├── Masks  
│   ├── PicturesForChkNormalization  
│   ├── QC  
│   ├── RealignParameter  
│   └── Results  
├── REST-meta-MDD-VBM-Phase1-Sharing  
│   ├── c1  
│   ├── c2  
│   ├── c3  
│   ├── mwc1  
│   ├── mwc2  
│   ├── mwc3  
│   ├── u_rc1  
│   ├── wc1  
│   ├── wc2  
│   └── wc3  

The data can be found on the rFMRI website https://rfmri.org/REST-meta-MDD. The filtered metadata Stat_Sub_Info_848MDDvs794NC.mat can be found in https://github.com/Chaogan-Yan/PaperScripts/tree/master/Yan_2019_PNAS/StatsSubInfo. Once `REST-meta-MDD` directory with the structure above is in the root directory, you can copy scripts/preprocess_data.py to the root directory and run it.

After preprocessing you can directly use dataset objects from src/dataset.py.