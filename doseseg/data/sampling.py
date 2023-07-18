from typing import (Iterator, Sequence, List, Any, Optional)

import torch
from torch import (randperm, randint)
from torch.utils.data.sampler import Sampler


class RandomForegroundRatioSampler(Sampler[int]):
    """A class for random sampling of cases under the constraint that at least a certain percentage of foreground slices
        must be provided in each mini-batch."""

    def __init__(self,
                 indices_foreground: Sequence[int],
                 indices_background: Sequence[int],
                 batch_size: int,
                 percentage_foreground: float = 0.3,
                 generator: Optional[torch.Generator] = None
                 ) -> None:
        """Represents a foreground ratio sampler which ensures that the specified percentage_foreground is provided
            in each batch.

        Args:
            indices_foreground (Sequence[int]): The identifiers for the data with foreground.
            indices_background (Sequence[int]): The identifiers for the data with just background.
            batch_size (int): The batch size of the mini-batch.
            percentage_foreground (float): The percentage of foreground in each batch.
            generator (Optional[torch.Generator]): The generator for deterministic computation of the permutations.
        """
        super().__init__(None)

        assert len(indices_foreground) > 0, 'The number of foreground indices must be larger than 0, but is 0!'
        self.indices_fg = indices_foreground

        assert len(indices_background) > 0, 'The number of background indices must be larger than 0, but is 0!'
        self.indices_bg = indices_background

        assert batch_size > 0, 'The batch size must be larger than zero!'

        assert 0 < percentage_foreground < 1, f'The foreground percentage must be larger than 0 and smaller than 1, ' \
                                              f'but is {percentage_foreground}!'

        self.fg_batch_size = round(percentage_foreground * batch_size)
        self.fg_batch_size = 1 if self.fg_batch_size < 1 else self.fg_batch_size

        self.bg_batch_size = batch_size - self.fg_batch_size
        assert self.bg_batch_size >= 0, f'The background batch size must be larger than 0, ' \
                                        f'but is {self.bg_batch_size}!'

        num_batches_fg = len(self.indices_fg) // self.fg_batch_size
        num_batches_bg = len(self.indices_bg) // self.bg_batch_size

        self.num_batches = max(num_batches_fg, num_batches_bg)

        self.permuted_fg_indices = []
        self.permuted_bg_indices = []

        self.generator = generator

    def _adjust_list_size(self,
                          data: List[Any],
                          size: int
                          ) -> List[Any]:
        """Adjusts the length of a list by adding or removing elements.

        Args:
            data (List[Any]): The data list.
            size (int): The output size of the list.

        Returns:
            List[Any]: The size-adjusted list.
        """
        if len(data) < size:
            difference = size - len(data)
            additional_indices = [data[i] for i in randint(0, len(data), (difference,), generator=self.generator)]
            data.extend(additional_indices)

        if len(data) > size:
            data = data[:size]

        return data

    def _generate_indices(self) -> None:
        """Generates the permuted indices for the foreground and the background.

        Returns:
            None
        """
        permuted_fg_indices = [self.indices_fg[i] for i in randperm(len(self.indices_fg), generator=self.generator)]
        permuted_bg_indices = [self.indices_bg[i] for i in randperm(len(self.indices_bg), generator=self.generator)]

        self.permuted_fg_indices = self._adjust_list_size(permuted_fg_indices,
                                                          self.num_batches * self.fg_batch_size)
        self.permuted_bg_indices = self._adjust_list_size(permuted_bg_indices,
                                                          self.num_batches * self.bg_batch_size)

    def __iter__(self) -> Iterator[int]:
        batch = []

        self._generate_indices()

        while len(self.permuted_fg_indices) > self.fg_batch_size:

            for _ in range(self.fg_batch_size):
                batch.append(self.permuted_fg_indices.pop(0))

            for _ in range(self.bg_batch_size):
                batch.append(self.permuted_bg_indices.pop(0))

            shuffle_idx = torch.randperm(len(batch), generator=self.generator).tolist()
            batch = [batch[i] for i in shuffle_idx]

            yield batch

            batch = []

    def __len__(self) -> int:
        return self.num_batches
