import os

import numpy as np
import SimpleITK as sitk
from pymia.data.assembler import SubjectAssembler
import pymia.data.extraction as extr
from pymia.data.conversion import NumpySimpleITKImageBridge
import torch


def c2d_evaluation_fn(trainer):

    # instantiate an empty list to store the dose score
    list_dose_score = []

    # instantiate the subject assembler to accumulate the dose per subject
    assembler = SubjectAssembler(trainer.setting.val_loader.dataset.datasource)
    datasource = trainer.setting.val_loader.dataset.datasource

    # instantiate the extractor to extract the necessary data from the dataset
    extractor = extr.ComposeExtractor([
        extr.ImagePropertiesExtractor(),
        extr.SubjectExtractor(),
        extr.DataExtractor(categories=('dose',))
    ])

    # test the current model on the validation loader
    with torch.no_grad():
        trainer.setting.network.eval()

        for i, batch in enumerate(trainer.setting.val_loader):

            # noinspection DuplicatedCode
            images = np.stack(batch['images'], axis=0)
            oar = np.stack(batch['oar'], axis=0)
            ptv = np.stack(batch['ptv'], axis=0)
            input_ = torch.from_numpy(np.concatenate([images, oar, ptv], axis=1)).to(trainer.setting.device,
                                                                                     dtype=torch.float)

            # predict the dose
            _, prediction = trainer.setting.network(input_)
            prediction = np.array(prediction.cpu().numpy()).transpose((0, 2, 3, 1))

            # mask the prediction
            mask = np.stack(batch['mask'], axis=0).transpose((0, 2, 3, 1))
            prediction[np.logical_or(mask < 1, prediction < 0)] = 0

            # accumulate the dose per subject
            assembler.add_batch(prediction, np.array(batch['sample_index']))

        # loop over the validation subjects, write them to a directory for later analysis, and evaluate the dose score
        for subject_idx in assembler.subjects_ready:
            prediction = assembler.get_assembled_subject(subject_idx)
            sample = datasource.direct_extract(extractor, subject_idx)

            prediction_image = NumpySimpleITKImageBridge.convert(prediction, sample['properties'])
            output_path_pred = os.path.join(trainer.setting.output_dir,
                                            f'prediction_{trainer.log.epoch}_{sample["subject"]}.nii.gz')
            sitk.WriteImage(prediction_image, output_path_pred)

            # evaluate the dose
            dose_score = 70 * np.mean(np.abs(prediction - sample['dose']))
            list_dose_score.append(dose_score)

    if trainer.setting.wandb_session is not None:
        trainer.setting.wandb_session.log({'dose_score': np.mean(list_dose_score),
                                           'epoch': trainer.log.epoch})

    # Evaluation score (higher is better)
    return -np.mean(list_dose_score)
