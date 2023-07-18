import typing as t
import json

from torch.utils.data import (
    SequentialSampler,
    DataLoader
)  # noqa
import pymia.data.transformation as tfm
import pymia.data.extraction as extr
import pymia.data.backends.pytorch as pymia_torch
import numpy as np
import matplotlib.pyplot as plt

from doseseg.data import (
    IndicesCalculator,
    OnlyBackgroundWithImageSelection,
    OnlyForegroundSelection,
    RandomForegroundRatioSampler,
    Segmentation2DAugmentation
)


def show(batch: t.Dict[str, t.Any],
         keys: t.Tuple[str] = ('images', 'ct', 'oar', 'mask')
         ) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, key in enumerate(keys):
        axes[i // 2, i % 2].imshow(batch[key][0][0, :, :])
        axes[i // 2, i % 2].set_title(key)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def default_collate_fn(batch_: t.List) -> t.Dict:
    return dict(zip(batch_[0], zip(*[d.values() for d in batch_])))


def get_indexing_strategy(use_2d: bool,
                          slice_axis: int,
                          **kwargs) -> extr.IndexingStrategy:
    if use_2d:
        return extr.SliceIndexing(slice_axis)

    return extr.PatchWiseIndexing(patch_shape=kwargs.get('patch_shape'),
                                  ignore_incomplete=kwargs.get('ignore_incomplete'))


def get_extractor() -> extr.Extractor:
    categories = ('images', 'ct', 'oar', 'mask', 'gtv')

    extractors = [extr.DataExtractor(categories),
                  extr.ImagePropertiesExtractor(),
                  extr.ImagePropertyShapeExtractor(),
                  extr.NamesExtractor(True, categories),
                  extr.IndexingExtractor(),
                  extr.SubjectExtractor()]

    return extr.ComposeExtractor(extractors)


def get_test_transforms() -> t.Optional[tfm.Transform]:
    categories = ('images', 'ct', 'oar', 'mask', 'gtv')
    return tfm.Permute((2, 0, 1), entries=categories)


def get_train_transforms() -> t.Optional[tfm.Transform]:
    augmentation = Segmentation2DAugmentation()
    return augmentation.get_transforms()


def get_segment_loader(
        path: str,
        split_config_path: str,
        batch_size_train: int,
        batch_size_val: int,
        batch_size_test: int,
        slice_axis: int = 0,
        indices_dir_path: str = './data/indices',
        **kwargs
) -> t.Dict[str, DataLoader]:
    # set a random seed for the augmentations
    np.random.seed(53215341)

    # load the split configuration
    with open(split_config_path, 'r') as f:
        split_config = json.load(f)

    if not split_config:
        raise ValueError('The split configuration is empty!')

    # get the selection strategy
    selection_strategy = OnlyForegroundSelection('gtv')

    # construct the training dataset
    train_dataset = extr.PymiaDatasource(path,
                                         get_indexing_strategy(True, slice_axis, **kwargs),
                                         get_extractor(),
                                         get_train_transforms(),
                                         subject_subset=split_config['train'])

    # compute or load the training indices
    calc_dose = IndicesCalculator(train_dataset,
                                  selection_strategy,
                                  indices_dir_path,
                                  False)
    train_indices = calc_dose.update()

    calc_no_dose = IndicesCalculator(train_dataset,
                                     OnlyBackgroundWithImageSelection('gtv', 'images'),
                                     indices_dir_path,
                                     False)
    train_indices_no_dose = calc_no_dose.update()

    # construct the sampler
    train_dataset = pymia_torch.PytorchDatasetAdapter(train_dataset)
    train_sampler = RandomForegroundRatioSampler(train_indices, train_indices_no_dose, batch_size_train)
    # noinspection PyTypeChecker
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=default_collate_fn,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # construct the validation dataset
    val_dataset = extr.PymiaDatasource(path,
                                       get_indexing_strategy(True, slice_axis, **kwargs),
                                       get_extractor(),
                                       get_test_transforms(),
                                       subject_subset=split_config['valid'])
    val_dataset = pymia_torch.PytorchDatasetAdapter(val_dataset)

    # construct the validation loader
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size_val,
                            sampler=val_sampler,
                            collate_fn=default_collate_fn)

    # construct the test dataset
    test_dataset = extr.PymiaDatasource(path,
                                        get_indexing_strategy(True, slice_axis, **kwargs),
                                        get_extractor(),
                                        get_test_transforms(),
                                        subject_subset=split_config['test'])
    test_dataset = pymia_torch.PytorchDatasetAdapter(test_dataset)

    # construct the test loader
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size_test,
                             sampler=test_sampler,
                             collate_fn=default_collate_fn)

    return {'train': train_loader, 'valid': val_loader, 'test': test_loader}


# test this file
# if __name__ == '__main__':
#     loaders = get_loader(r"E:\DataBackupsConversion\segment_dataset.h5",
#                          '../../split_config/split_config_segment_fold_0.json',
#                          2, 1, 1, 0,
#                          '../../indices/')
#
#     loader = loaders['train']
#     for i, batch in enumerate(loader):
#         show(batch)
#         print(i)
#
#     print('finished')
