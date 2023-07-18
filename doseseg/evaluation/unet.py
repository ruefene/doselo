import os

import numpy as np
import SimpleITK as sitk
from pymia.data.assembler import SubjectAssembler
import pymia.data.extraction as extr
from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation.metric import DiceCoefficient, HausdorffDistance
from pymia.evaluation.writer import ConsoleWriter
from pymia.data.conversion import NumpySimpleITKImageBridge
import torch


def unet_evaluation_fn(trainer):
    assembler = SubjectAssembler(trainer.setting.val_loader.dataset.datasource)
    datasource = trainer.setting.val_loader.dataset.datasource

    extractor = extr.ComposeExtractor([
        extr.ImagePropertiesExtractor(),
        extr.SubjectExtractor(),
        extr.DataExtractor(categories=('gtv',))
    ])

    labels = {1: 'GTV', }

    metrics = [
        DiceCoefficient(metric='DICE'),
        HausdorffDistance(percentile=95, metric='HDRFDST95'),
        HausdorffDistance(percentile=100, metric='HDRFDST100'),
    ]

    evaluator = SegmentationEvaluator(metrics, labels)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []

    with torch.no_grad():
        trainer.setting.network.eval()

        for i, batch in enumerate(trainer.setting.val_loader):
            # noinspection DuplicatedCode
            images = np.stack(batch['images'], axis=0)
            input_ = torch.from_numpy(images).to(trainer.setting.device, dtype=torch.float)
            target = torch.from_numpy(np.stack(batch['gtv'], axis=0)).to(trainer.setting.device, dtype=torch.float)

            # predict the gtv
            prediction_raw = trainer.setting.network(input_)
            prediction = torch.sigmoid(prediction_raw)

            # compute the loss
            loss = loss_fn(prediction_raw, target)
            losses.append(float(loss.item()))

            prediction = np.expand_dims(np.array(torch.ge(prediction, 0.5).cpu().numpy()), -1)

            # accumulate the dose per subject
            is_last_batch = False
            if i + 1 == len(datasource):
                is_last_batch = True
            assembler.add_batch(prediction, np.array(batch['sample_index']), is_last_batch)

        # loop over the validation subjects
        for i, subject_idx in enumerate(assembler.subjects_ready):
            prediction = assembler.get_assembled_subject(subject_idx)
            sample = datasource.direct_extract(extractor, subject_idx)

            prediction_image = NumpySimpleITKImageBridge.convert(prediction, sample['properties'])
            label_image = NumpySimpleITKImageBridge.convert(sample['gtv'], sample['properties'])

            output_path_pred = os.path.join(trainer.setting.output_dir,
                                            f'prediction_{trainer.log.epoch}_{sample["subject"]}.nii.gz')
            sitk.WriteImage(prediction_image, output_path_pred)

            evaluator.evaluate(prediction_image, label_image, sample['subject'])

    writer = ConsoleWriter()
    writer.write(evaluator.results)

    # get the scores / performance values
    dice_list = []
    hd95_list = []
    hd100_list = []

    for res in evaluator.results:
        if res.metric == 'DICE':
            dice_list.append(res.value)

        if res.metric == 'HDRFDST95':
            hd95_list.append(res.value)

        if res.metric == 'HDRFDST100':
            hd100_list.append(res.value)

    performance_value = float(np.mean(dice_list))
    mean_loss = float(np.mean(np.array(losses)))

    if trainer.setting.wandb_session is not None:
        trainer.setting.wandb_session.log({"mean_DICE": float(np.mean(dice_list)),
                                           "std_DICE": float(np.std(dice_list)),
                                           "mean_HD95": float(np.mean(hd95_list)),
                                           "std_HD95": float(np.std(hd95_list)),
                                           "min_HD95": float(np.min(hd95_list)),
                                           "max_HD95": float(np.max(hd95_list)),
                                           "mean_HD100": float(np.mean(hd100_list)),
                                           "std_HD100": float(np.std(hd100_list)),
                                           "min_HD100": float(np.min(hd100_list)),
                                           "max_HD100": float(np.max(hd100_list)),
                                           "val_loss": float(mean_loss),
                                           "performance_index": float(performance_value),
                                           "epoch": trainer.log.epoch})
    else:
        trainer.print_log_to_file(f"mean_DICE: {float(np.mean(dice_list))}", 'a')
        trainer.print_log_to_file(f"std_DICE: {float(np.std(dice_list))}", 'a')
        trainer.print_log_to_file(f"mean_HD95: {float(np.mean(hd95_list))}", 'a')
        trainer.print_log_to_file(f"std_HD95: {float(np.std(hd95_list))}", 'a')
        trainer.print_log_to_file(f"min_HD95: {float(np.min(hd95_list))}", 'a')
        trainer.print_log_to_file(f"max_HD95: {float(np.max(hd95_list))}", 'a')
        trainer.print_log_to_file(f"mean_HD100: {float(np.mean(hd100_list))}", 'a')
        trainer.print_log_to_file(f"std_HD100: {float(np.std(hd100_list))}", 'a')
        trainer.print_log_to_file(f"min_HD100: {float(np.min(hd100_list))}", 'a')
        trainer.print_log_to_file(f"max_HD100: {float(np.max(hd100_list))}", 'a')
        trainer.print_log_to_file(f"val_loss: {float(mean_loss)}", 'a')
        trainer.print_log_to_file(f"performance_index: {float(performance_value)}", 'a')
        trainer.print_log_to_file(f"epoch: {trainer.log.epoch}", 'a')

    # Evaluation score (higher is better)
    return performance_value
