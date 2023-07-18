import typing as t
import os
import json
from time import time
from zlib import crc32

from pymia.data.extraction.datasource import PymiaDatasource
from pymia.data.extraction import SelectionStrategy


class IndicesCalculator:

    def __init__(self,
                 dataset: PymiaDatasource,
                 selection_strategy: SelectionStrategy,
                 output_dir: t.Optional[str],
                 force: bool = False,
                 patch_shape: t.Optional[t.Tuple[int, ...]] = None
                 ) -> None:
        super().__init__()

        self.dataset = dataset
        self.selection_strategy = selection_strategy

        if not output_dir and force:
            raise ValueError('There must be a path provided when indices generation is forced!')

        if output_dir:
            if not os.path.exists(output_dir):
                raise Exception(f'The output path {output_dir} is invalid!')

            if not os.path.isdir(output_dir):
                raise NotADirectoryError(f'The output path {output_dir} is not a directory!')

        self.output_dir = output_dir
        self.force = force
        self.patch_shape = patch_shape

    @staticmethod
    def _get_hash(dataset: PymiaDatasource,
                  selection_strategy: SelectionStrategy,
                  patch_shape: t.Tuple[int]
                  ) -> str:
        to_hash = os.path.basename(dataset.dataset_path) + \
                  ''.join(sorted(dataset.subject_subset)) + \
                  repr(dataset.indexing_strategy) + \
                  repr(selection_strategy)
        if patch_shape:
            to_hash += str(patch_shape)

        return hex(crc32(bytes(to_hash, encoding='utf-8')) & 0xffffffff)

    @staticmethod
    def _compute_indices(hash_: str,
                         selection: SelectionStrategy,
                         dataset: PymiaDatasource
                         ) -> t.List[int]:
        print(f'Calculate indices for hash {hash_} ({selection.__class__.__name__})...')
        indices = []
        start_time = time()
        # noinspection PyTypeChecker
        for i, sample in enumerate(dataset):
            if selection(sample):
                indices.append(i)
        print(f'Duration of indices calculation: {(time() - start_time):.2f}sec')

        if not indices:
            raise ValueError('No training indices could be computed!')

        return indices

    def update(self) -> t.Tuple[int]:
        hash_ = self._get_hash(self.dataset, self.selection_strategy, self.patch_shape)

        file_path = os.path.join(self.output_dir, f'{hash_}.json')

        if os.path.exists(file_path) and not self.force:
            with open(file_path, 'r') as file:
                indices = json.load(file)

            return tuple(indices)

        indices = self._compute_indices(hash_, self.selection_strategy, self.dataset)

        with open(file_path, 'w') as file:
            json.dump(indices, file, indent=4)

        return tuple(indices)
