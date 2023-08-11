import typing as t
import os

import torch
import numpy as np
import SimpleITK as sitk
import pymia.data.assembler as assm
import pymia.data.conversion as conv
from pymia.data.extraction import ComposeExtractor, DataExtractor, SubjectExtractor, ImagePropertiesExtractor
from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation.metric import DiceCoefficient, HausdorffDistance
from pymia.evaluation.writer import ConsoleWriter, CSVWriter, StatisticsAggregator
from pymia.data.conversion import NumpySimpleITKImageBridge


def noise(tensor: torch.Tensor, std: float, mean: float = 0.) -> torch.Tensor:
    noise_ = torch.randn(*tensor.shape, device=tensor.device, dtype=torch.float) * std + mean
    return tensor + noise_  # type: ignore


def gamma(tensor: torch.Tensor, gamma_: float) -> torch.Tensor:
    return torch.pow(tensor, np.exp(gamma_))


class TesterSettings:

    def __init__(self):
        self.output_dir = None
        self.model_path = None
        self.model_name = None
        self.dose_model_path = None

        self.model = None
        self.dose_model = None
        self.device = None

        self.test_loader = None

        self.loss_fn = None

        self.to_feed_tensor_fn = None
        self.post_forward_fn = None
        self.evaluation_batch_fn = None
        self.evaluation_fn = None

        self.predict_all = False
        self.tta = False


class Tester:

    def __init__(self, settings: TesterSettings):
        self.settings = settings

    def load_model(self):
        # load model
        model = self.settings.model
        model.load_state_dict(torch.load(self.settings.model_path).get('network_state_dict'))
        model.to(self.settings.device)
        model.eval()
        self.settings.model = model

    def load_dose_model(self):
        # load model
        model = self.settings.dose_model
        model.load_state_dict(torch.load(self.settings.dose_model_path).get('network_state_dict'))
        model.to(self.settings.device)
        model.eval()
        self.settings.dose_model = model

    def test(self):
        for i, batch in enumerate(self.settings.test_loader):
            print(f'Processing batch {i + 1}/{len(self.settings.test_loader)} of subject {batch["subject"][0]}...')

            # get the input and target
            input_, target = self.settings.to_feed_tensor_fn(self, batch)

            # forward pass
            output = self.settings.model(input_)

            # calculate loss
            target = target.unsqueeze(1).float()
            loss = self.settings.loss_fn(output, target).item()

            # get the class prediction
            output = self.settings.post_forward_fn(output)

            # calculate evaluation
            self.settings.evaluation_batch_fn(self, output, target, batch, loss)

        # evaluate the result
        test_loss = self.settings.evaluation_fn(self)

        return test_loss


class DualTester(Tester):

    def __init__(self, settings: TesterSettings):
        super().__init__(settings)

        self.dose_assembler = None
        self.dose_assembler_p = None
        self.segment_assembler = None

    def init(self):
        self.load_dose_model()
        self.load_model()

        self.dose_assembler = assm.SubjectAssembler(self.settings.test_loader.dataset.datasource)
        self.dose_assembler_p = assm.SubjectAssembler(self.settings.test_loader.dataset.datasource)
        self.segment_assembler = assm.SubjectAssembler(self.settings.test_loader.dataset.datasource)

    def save_assembler_content(
            self,
            assembler: assm.SubjectAssembler,
            postfix: str,
            categories: t.Tuple[str, ...] = ('gtv',),
            data_type: type = np.uint8
    ) -> None:
        # build the extractor
        extractor = ComposeExtractor([
            SubjectExtractor(),
            ImagePropertiesExtractor(),
            DataExtractor(categories=categories)
        ])

        for subject_idx in assembler.subjects_ready:
            prediction = assembler.get_assembled_subject(subject_idx).astype(data_type)
            sample = self.settings.test_loader.dataset.datasource.direct_extract(extractor, subject_idx)

            # cast to SimpleITK image
            prediction_image = conv.NumpySimpleITKImageBridge.convert(prediction, sample['properties'])

            # save the prediction
            output_path_pred = os.path.join(self.settings.output_dir, f'{sample["subject"]}_{postfix}.nii.gz')
            sitk.WriteImage(prediction_image, output_path_pred)

    def save_from_dataset(
            self,
            subject_indices: t.Tuple[int, ...],
            categories: t.Tuple[str, ...] = ('oar',),
            dtypes: t.Tuple[type, ...] = (np.uint8,)
    ) -> None:
        # validate the arguments
        if len(categories) != len(dtypes):
            raise ValueError('The number of categories and dtypes must be the same.')

        # build the extractor
        extractor = ComposeExtractor([
            SubjectExtractor(),
            ImagePropertiesExtractor(),
            DataExtractor(categories=categories)
        ])

        # iterate over the assembled subjects and save the requested data
        for subject_idx in subject_indices:

            sample = self.settings.test_loader.dataset.datasource.direct_extract(extractor, subject_idx)

            for category, data_type in zip(categories, dtypes):

                # cast to SimpleITK image
                image = conv.NumpySimpleITKImageBridge.convert(sample[category].astype(data_type), sample['properties'])

                # save the prediction
                output_path = os.path.join(self.settings.output_dir, f'{sample["subject"]}_{category}_gt.nii.gz')
                sitk.WriteImage(image, output_path)

    def save_image_from_dataset(
            self,
            subject_indices: t.Tuple[int, ...],
            image_names: t.Tuple[str, ...] = ('T1c', 'T1w', 'T2w', 'FLAIR')
    ) -> None:
        # build the extractor
        extractor = ComposeExtractor([
            SubjectExtractor(),
            ImagePropertiesExtractor(),
            DataExtractor(categories=('images',))
        ])

        # iterate over the assembled subjects and save the requested data
        for subject_idx in subject_indices:

            sample = self.settings.test_loader.dataset.datasource.direct_extract(extractor, subject_idx)

            for i, image_name in enumerate(image_names):
                image = conv.NumpySimpleITKImageBridge.convert(sample['images'][..., i].astype(np.float),
                                                               sample['properties'])

                # save the prediction
                output_path = os.path.join(self.settings.output_dir, f'{sample["subject"]}_{image_name}.nii.gz')
                sitk.WriteImage(image, output_path)

    def _evaluate_segmentations(
            self,
            subject_indices: t.Tuple[int, ...]
    ):
        metrics = [DiceCoefficient('DSC'),
                   HausdorffDistance(metric='HD100'),
                   HausdorffDistance(percentile=95., metric='HD95')]
        labels = {1: 'GTV'}
        evaluator = SegmentationEvaluator(metrics, labels)

        extractor = ComposeExtractor([
            SubjectExtractor(),
            ImagePropertiesExtractor(),
            DataExtractor(categories=('gtv',))
        ])

        for subject_idx in subject_indices:
            sample = self.settings.test_loader.dataset.datasource.direct_extract(extractor, subject_idx)
            prediction = self.segment_assembler.predictions.get(subject_idx).get('__prediction').astype(np.uint8)

            prediction_image = NumpySimpleITKImageBridge.convert(prediction, sample['properties'])
            target_image = NumpySimpleITKImageBridge.convert(sample['gtv'].astype(np.uint8), sample['properties'])

            evaluator.evaluate(prediction_image, target_image, sample['subject'])

        ConsoleWriter().write(evaluator.results)
        CSVWriter(os.path.join(self.settings.output_dir, 'results.csv')).write(evaluator.results)

        aggregation_fns = {'MEAN': np.mean, 'STD': np.std, 'MIN': np.min, 'MAX': np.max}
        aggregator = StatisticsAggregator(aggregation_fns)
        aggregated_results = aggregator.calculate(evaluator.results)

        ConsoleWriter().write(aggregated_results)
        CSVWriter(os.path.join(self.settings.output_dir, 'results_aggregated.csv')).write(aggregated_results)

    def test(self):
        # check if the output directory exists
        if not os.path.exists(self.settings.output_dir):
            os.makedirs(self.settings.output_dir)

        with torch.no_grad():
            for i, batch in enumerate(self.settings.test_loader):
                print(f'Processing batch {i + 1}/{len(self.settings.test_loader)} of subject {batch["subject"][0]}...')

                # get the input and target
                input_ = torch.from_numpy(np.stack(batch['images'], axis=0)).to(self.settings.device, dtype=torch.float)
                mask = torch.from_numpy(np.stack(batch['mask'], axis=0)).to(self.settings.device, dtype=torch.long)
                ct = torch.from_numpy(np.stack(batch['ct'], axis=0)).to(self.settings.device, dtype=torch.float)
                oar = torch.from_numpy(np.stack(batch['oar'], axis=0)).to(self.settings.device, dtype=torch.float)
                gtv = torch.from_numpy(np.stack(batch['gtv'], axis=0)).to(self.settings.device, dtype=torch.float)
                input_dose_gt = torch.cat([ct, oar, gtv], dim=1).to(self.settings.device, dtype=torch.float)

                # forward pass segmentation
                if self.settings.tta:
                    num_tta_steps = 9
                    noise_std = 0.05
                    gamma_ = 0.1

                    probabilities = torch.zeros((input_.shape[0], 1, input_.shape[2], input_.shape[3]),
                                                dtype=torch.float, device=self.settings.device)

                    for j in range(num_tta_steps):
                        if j == 0:
                            output = self.settings.model(input_)
                            probabilities += torch.sigmoid(output)
                        elif j == 1:
                            input_tta = torch.flip(input_, dims=(2,))
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(torch.flip(output, dims=(2,)))
                        elif j == 2:
                            input_tta = torch.flip(input_, dims=(3,))
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(torch.flip(output, dims=(3,)))
                        elif j == 3:
                            input_tta = noise(torch.clone(input_), noise_std)
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(output)
                        elif j == 4:
                            input_tta = noise(torch.flip(input_, dims=(2,)), noise_std)
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(torch.flip(output, dims=(2,)))
                        elif j == 5:
                            input_tta = noise(torch.flip(input_, dims=(3,)), noise_std)
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(torch.flip(output, dims=(3,)))
                        elif j == 6:
                            input_tta = gamma(torch.clone(input_), gamma_)
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(output)
                        elif j == 7:
                            input_tta = gamma(torch.flip(input_, dims=(2,)), gamma_)
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(torch.flip(output, dims=(2,)))
                        elif j == 8:
                            input_tta = gamma(torch.flip(input_, dims=(3,)), gamma_)
                            output = self.settings.model(input_tta)
                            probabilities += torch.sigmoid(torch.flip(output, dims=(3,)))
                        else:
                            print(f'Error: TTA not implemented for index {j}!')

                    output_seg = probabilities / num_tta_steps  # type: torch.Tensor
                    classes_seg = torch.ge(output_seg, 0.5).squeeze(1).unsqueeze(-1)
                else:
                    output_seg = self.settings.model(input_)
                    classes_seg = torch.ge(torch.sigmoid(output_seg), 0.5).squeeze(1).unsqueeze(-1)

                # construct the input for the dose prediction
                input_dose_pred = torch.cat([ct, oar, torch.sigmoid(output_seg)], dim=1).to(self.settings.device,
                                                                                            dtype=torch.float)

                # forward pass dose
                output_dose_gt = self.settings.dose_model(input_dose_gt)[1]
                output_dose_gt = output_dose_gt if self.settings.predict_all else output_dose_gt * mask
                output_dose_gt = output_dose_gt.squeeze(1).unsqueeze(-1)
                output_dose_gt = torch.clip(output_dose_gt, 0, 1)
                output_dose_pred = self.settings.dose_model(input_dose_pred)[1]
                output_dose_pred = output_dose_pred if self.settings.predict_all else output_dose_pred * mask
                output_dose_pred = output_dose_pred.squeeze(1).unsqueeze(-1)
                output_dose_pred = torch.clip(output_dose_pred, 0, 1)

                # analyze if last batch
                last_batch = False
                for index_expr, shape in zip(batch['index_expr'], batch['shape']):
                    if index_expr.get_indexing()[0] == shape[-1] - 1:
                        last_batch = True
                        break

                # assemble the dose
                self.dose_assembler.add_batch(output_dose_gt.detach().cpu().numpy().astype(np.float),
                                              np.array(batch['sample_index']),
                                              last_batch)
                self.dose_assembler_p.add_batch(output_dose_pred.detach().cpu().numpy().astype(np.float),
                                                np.array(batch['sample_index']),
                                                last_batch)

                # assemble the segmentation
                self.segment_assembler.add_batch(classes_seg.detach().cpu().numpy().astype(np.uint8),
                                                 np.array(batch['sample_index']),
                                                 last_batch)

            # get the indices of the assembled subjects
            subject_indices = tuple(self.segment_assembler.subjects_ready)

            # evaluate the segmentations
            print('Evaluating the segmentations...')
            self._evaluate_segmentations(subject_indices)

            # save the predictions
            print('Saving the predictions...')
            self.save_assembler_content(self.segment_assembler, 'gtv_pred', categories=('gtv',))
            self.save_assembler_content(self.dose_assembler, 'dose_gt', categories=('gtv',),
                                        data_type=np.float)
            self.save_assembler_content(self.dose_assembler_p, 'dose_pred', categories=('gtv',),
                                        data_type=np.float)

            # save the ground truth
            print('Saving the ground truths...')
            self.save_from_dataset(subject_indices,
                                   categories=('gtv', 'oar', 'ct'),
                                   dtypes=(np.uint8, np.uint8, np.float))
            self.save_image_from_dataset(subject_indices)
