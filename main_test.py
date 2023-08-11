import typing as t
import argparse

import torch
import torch.nn as nn

from doseseg import (
    TesterSettings,
    DualTester,
    UNet2D,
    C2DModel,
    get_segment_loader,
)


def to_feed_tensor(_: DualTester, _a: t.Dict[str, t.Any]):
    raise NotImplementedError()


def post_forward(_: torch.Tensor):
    raise NotImplementedError()


def eval_batch_fn(
        _1: DualTester,
        _2: torch.Tensor,
        _3: torch.Tensor,
        _4: t.Dict[str, t.Any],
        _5: float
) -> None:
    raise NotImplementedError()


def eval_fn(_: DualTester) -> float:
    raise NotImplementedError()


def main(
        segmentation_model_path: str,
        dose_model_path: str,
        dataset_path: str,
        split_config_path: str,
        output_dir: str,
        batch_size: int,
        use_tta: bool
) -> None:
    # initialize the test loader
    test_loader = get_segment_loader(dataset_path, split_config_path, batch_size, batch_size, batch_size).get('test')

    # initialize the settings
    settings = TesterSettings()
    settings.use_tta = use_tta
    settings.output_dir = output_dir
    settings.model_path = segmentation_model_path
    settings.dose_model_path = dose_model_path
    settings.model_name = 'UNet2D'
    settings.model = UNet2D(4, 1)
    settings.dose_model = C2DModel(
        in_ch=3,
        out_ch=1,
        list_ch_a=[-1, 16, 32, 64, 128, 256],
        list_ch_b=[-1, 32, 64, 128, 256, 512],
    )
    settings.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    settings.test_loader = test_loader
    settings.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.]).to(torch.device('cuda:0')))
    settings.to_feed_tensor_fn = to_feed_tensor
    settings.post_forward_fn = post_forward
    settings.evaluation_batch_fn = eval_batch_fn
    settings.evaluation_fn = eval_fn
    settings.predict_all = False

    # initialize and start the tester
    tester = DualTester(settings)
    tester.init()
    tester.test()

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_model_path', type=str, help='Path to the segmentation model.')
    parser.add_argument('--dose_model_path', type=str, help='Path to the dose model.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset.',
                        default='./data/dataset/segment_dataset.h5')
    parser.add_argument('--split_config_path', type=str, help='Path to the split config file.',
                        default='./data/split_config_segment/split_config_fold_0.json')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory.', default='./results/')
    parser.add_argument('--batch_size', type=int, help='Batch size (default: 16).', default=16)
    parser.add_argument('--use_tta', type=bool, help='Use test time augmentation (default: False).', default=False)

    args = parser.parse_args()

    main(args.segmentation_model_path, args.dose_model_path, args.dataset_path, args.split_config_path, args.output_dir,
         args.batch_size, args.use_tta)
