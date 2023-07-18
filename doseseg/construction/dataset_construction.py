import argparse
import enum
import glob
import os
import typing as t

import SimpleITK as sitk
import numpy as np

import pymia.data as data
import pymia.data.conversion as conv
import pymia.data.creation as crt
import pymia.data.transformation as tfm
import pymia.data.creation.fileloader as file_load


"""This is an exemplary dataset construction script and has to be changed accordingly be the user. This script assumes
that the data is available as NIFTI files and organized in a single folder per subject. If this is not the case, we
recommend using the PyRaDiSe package for converting the DICOM-RT data to NIFTI files and organize the data 
accordingly."""


class NormalizeCT(tfm.Transform):

    def __init__(
            self,
            entries: t.Tuple[str] = ('ct',)
    ) -> None:
        super().__init__()

        self.entries = entries

    def __call__(
            self,
            sample: dict
    ) -> dict:
        for cat in self.entries:
            if cat not in sample:
                continue

            sample[cat] = self._normalize(sample[cat])

        return sample

    @staticmethod
    def _normalize(
            array: np.ndarray
    ) -> np.ndarray:
        return np.clip(array, -1024, 2000).astype(np.float32) / 1000.


class FileTypes(enum.Enum):
    T1c = 1
    T1w = 2
    T2w = 3
    FLAIR = 4
    CT = 5  # The T1-weighted MR image
    OAR = 6  # The organ at risk image
    GTV = 7  # The gross tumor volume image
    MASK = 8  # The body mask image


class Subject(data.SubjectFile):

    def __init__(
            self,
            subject: str,
            files: dict
    ) -> None:
        super().__init__(subject,
                         images={FileTypes.T1c.name: files[FileTypes.T1c],
                                 FileTypes.T1w.name: files[FileTypes.T1w],
                                 FileTypes.T2w.name: files[FileTypes.T2w],
                                 FileTypes.FLAIR.name: files[FileTypes.FLAIR]},
                         ct={FileTypes.CT.name: files[FileTypes.CT]},
                         oar={FileTypes.OAR.name: files[FileTypes.OAR]},
                         gtv={FileTypes.GTV.name: files[FileTypes.GTV]},
                         mask={FileTypes.MASK.name: files[FileTypes.MASK]})
        self.subject_path = files.get(subject, '')


class LoadData(file_load.Load):

    def __call__(
            self,
            file_name: str,
            id_: str,
            category: str,
            subject_id: str
    ) -> t.Tuple[np.ndarray, t.Union[conv.ImageProperties, None]]:
        if category in ('images', 'ct'):
            img = sitk.ReadImage(file_name, sitk.sitkFloat32)
        else:
            # this is the ground truth (defs.KEY_LABELS) and mask, which will be loaded as unsigned integer
            img = sitk.ReadImage(file_name, sitk.sitkUInt8)

        # return both the image intensities as np.ndarray and the properties of the image
        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)



def get_subjects(data_dir: str) -> t.List[Subject]:

    # get subjects
    subject_dirs = [subject_dir for subject_dir in glob.glob(os.path.join(data_dir, '*')) if
                    os.path.isdir(subject_dir) and os.path.basename(subject_dir).startswith('ISAS_') and
                    'GBM' in os.path.basename(subject_dir)]
    sorted(subject_dirs)

    # the keys of the data to write to the dataset
    keys = [FileTypes.T1c, FileTypes.T1w, FileTypes.T2w, FileTypes.FLAIR, FileTypes.CT, FileTypes.OAR, FileTypes.GTV,
            FileTypes.MASK]

    subjects = []
    # for each subject on file system, initialize a Subject object
    for subject_dir in subject_dirs:
        id_ = os.path.basename(subject_dir)

        file_dict = {id_: subject_dir}  # init dict with id_ pointing to the path of the subject
        for file_key in keys:
            if file_key == FileTypes.T1c:
                file_name = f'img_{id_}_masked-T1c-normalized.nii.gz'
                # file_name = f'img_{id_}_T1c.nii.gz'
            elif file_key == FileTypes.T1w:
                file_name = f'img_{id_}_masked-T1w-normalized.nii.gz'
                # file_name = f'img_{id_}_T1w.nii.gz'
            elif file_key == FileTypes.T2w:
                file_name = f'img_{id_}_masked-T2w-normalized.nii.gz'
                # file_name = f'img_{id_}_T2w.nii.gz'
            elif file_key == FileTypes.FLAIR:
                file_name = f'img_{id_}_masked-FLAIR-normalized.nii.gz'
                # file_name = f'img_{id_}_FLAIR.nii.gz'
            elif file_key == FileTypes.CT:
                file_name = f'img_{id_}_CT.nii.gz'
            elif file_key == FileTypes.OAR:
                file_name = f'seg_{id_}_OAR_all.nii.gz'
            elif file_key == FileTypes.GTV:
                file_name = f'seg_{id_}_NA_masked-GTVp.nii.gz'
            elif file_key == FileTypes.MASK:
                file_name = f'seg_{id_}_NA_Brain.nii.gz'
            else:
                raise ValueError('Unknown key')

            file_dict[file_key] = os.path.join(subject_dir, file_name)

        subjects.append(Subject(id_, file_dict))

    return subjects


def main(
        input_dir_path: str,
        dataset_output_path: str
) -> None:
    # get the subjects
    subjects = get_subjects(input_dir_path)

    # remove the "old" dataset if it exists
    if os.path.exists(dataset_output_path):
        os.remove(dataset_output_path)

    with crt.get_writer(dataset_output_path) as writer:
        # initialize the callbacks that will actually write the data to the dataset file
        callbacks = crt.get_default_callbacks(writer)

        # add the transforms that will be applied to the data before writing it to the dataset
        transform = tfm.ComposeTransform([NormalizeCT(),])

        # run through the subject files (loads them, applies transformations, and calls the callback for writing them)
        traverser = crt.Traverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(), transform=transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the ISAS dataset to a HDF5 dataset file.')
    parser.add_argument('input_dir_path', type=str, help='The path to the directory containing the ISAS dataset.')
    parser.add_argument('dataset_output_path', type=str, help='The path to the HDF5 dataset file to write to.')
    args = parser.parse_args()

    main(args.input_dir_path, args.dataset_output_path)
