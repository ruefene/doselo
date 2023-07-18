# Dose Guidance for Radiotherapy-oriented Deep Learning Segmentation (DOSELO)
This repository contains the code for the paper "Dose Guidance for Radiotherapy-oriented Deep Learning Segmentation 
(DOSELO)" which is accepted for the MICCAI 2023 in Vancouver.

## Paper Abstract
Deep learning-based image segmentation for radiotherapy is intended to speed up the planning process and yield 
consistent results. However, most of these segmentation methods solely rely on distribution and geometry-associated 
training objectives without considering tumor control and the sparing of healthy tissues. To incorporate dosimetric
effects into segmentation models, we propose a new training loss function that extends current state-of-the-art 
segmentation model training via a dose-based guidance method. We hypothesized that adding sucha dose-guidance 
mechanism improves the robustness of the segmentation with respect to the dose (i.e., resolves distant outliers and 
focuses on locations of high dose/dose gradient). We demonstrate the effectiveness of the proposed method on Gross 
Tumor Volume segmentation for glioblastoma treatment. The obtained dosimetry-based results show reduced dose errors 
relative to the ground truth dose map using the proposed dosimetry-segmentation guidance, outperforming state-of-the-art
distribution and geometry-based segmentation losses.

## Repository Structure
This repository contains the following functionality for reproducing our results:
- Training procedure for the dose predictor training (see `main_train_dose.py`)
- Training procedure for the segmentation model training (see `main_train_segment.py`)
- Evaluation procedure for the dose predictor and the segmentation model (see `main_test.py`)
- Script for dataset construction (see `doseseg/construction/dataset_construction.py`)

All provided code is located in the `./doseseg/` folder and all persistent and temporary data is found in the `./data/` 
folder. The datasets (i.e., one for training the dose predictor and one for training the segmentation model) are not
included in this repository because no ethical approval for sharing the data was obtained. However, we share with you 
all non-confidential data and code that is necessary to reproduce our results on your own data.

## Training Workflow
Using the proposed method with your own data requires the following steps:

1. Acquire a dataset comprising for each patient: 
    - A CT image
    - All necessary MR images (optional)
    - All OAR segmentations
    - A target segmentation
    - A simulated dose map
2. Convert the clinical data to a discretized file format (e.g., NIfTI) and co-register all images. The images and the segmentations should have the same physical properties (i.e., size, spacing, direction, orientation). Converting and harmonizing the data is best achieved using our [PyRaDiSe](https://pyradise.readthedocs.io/en/latest/) package.
3. Construct a dataset for training the dose predictor and another one for training the segmentation networks (see `doseseg/construction/dataset_construction.py`)
4. Train the dose predictor (see `main_train_dose.py`) using the dose predictor dataset
5. Train the segmentation model (see `main_train_segment.py`) using the segmentation dataset and the previously trained dose predictor
6. Evaluate the trained dose predictor and segmentation model (see `main_test.py`)
7. (Optional) Use the trained dose predictor and segmentation model for dose prediction and segmentation of new patients
8. Congratulations, you have successfully trained a dose aware segmentation model on your own data :)

## Citation
If you use our code for your own work, please cite our paper:
```
@InProceedings{Ruefenacht2023a,
  author    = {Rüfenacht, Elias and Kamath, Amith and Poel, Robert and Ermis, Ekin and Scheib, Stefan and Fix, Michael K. and Reyes, Mauricio},
  booktitle = {Medical Image Computing and Computer Assisted Intervention – MICCAI 2023},
  title     = {{Dose Guidance for Radiotherapy-oriented Deep Learning Segmentation}},
  year      = {2023},
  publisher = {Springer},
}
```
