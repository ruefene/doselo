import typing as t

import numpy as np
from pymia.data.extraction import SelectionStrategy


class NoConstantSelection(SelectionStrategy):

    def __init__(self,
                 loop_axis: t.Optional[int],
                 category: str = 'dose'
                 ) -> None:
        super().__init__()

        self.loop_axis = loop_axis
        self.category = category

    def __call__(self, sample: dict) -> bool:
        data = sample[self.category]

        if self.loop_axis is None:
            return not self._all_equal(data)

        slicing: t.List[t.Union[int, slice]] = [slice(None) for _ in range(data.ndim)]
        for i in range(data.shape[self.loop_axis]):
            slicing[self.loop_axis] = i
            slice_data = data[slicing]
            if not self._all_equal(slice_data):
                return True

    @staticmethod
    def _all_equal(data):
        return np.all(data == data.ravel()[0])

    def __repr__(self) -> str:
        return '{}({}, {})'.format(self.__class__.__name__, self.loop_axis, self.category)


class OnlyForegroundSelection(SelectionStrategy):

    def __init__(self,
                 category: str = 'mask'
                 ) -> None:
        super().__init__()

        self.category = category

    def __call__(self, sample) -> bool:
        return (sample[self.category]).any()

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.category)


class OnlyBackgroundSelection(SelectionStrategy):

    def __init__(self, category: str = 'gtv') -> None:
        super().__init__()

        self.category = category

    def __call__(self, sample) -> bool:
        return not (sample[self.category]).any()

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.category)


class OnlyBackgroundWithImageSelection(SelectionStrategy):

    def __init__(self,
                 background_category: str = 'gtv',
                 image_category: str = 'images'
                 ) -> None:
        super().__init__()

        self.background_category = background_category
        self.image_category = image_category

    def __call__(self, sample) -> bool:
        return not (sample[self.background_category]).any() and (sample[self.image_category]).any()

    def __repr__(self) -> str:
        return '{}({}, {})'.format(self.__class__.__name__, self.background_category, self.image_category)

