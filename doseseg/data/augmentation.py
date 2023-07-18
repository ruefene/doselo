from abc import (ABC, abstractmethod)

import typing as t
import numpy as np
import torchio as tio
import SimpleITK as sitk
from pymia.data.transformation import (
    Transform,
    Permute,
    raise_error_if_entry_not_extracted,
    ENTRY_NOT_EXTRACTED_ERR_MSG,
    check_and_return,
    ComposeTransform
)
import torch


class TorchIOTransform(Transform):
    """Example wrapper for `TorchIO <https://github.com/fepegar/torchio>`_ transformations."""

    def __init__(self,
                 transforms: list,
                 image_entries=('images', 'ct'),
                 label_entries=('gtv', 'oar', 'mask')
                 ) -> None:
        super().__init__()
        self.transforms = transforms
        self.entries = list(image_entries) + list(label_entries)

        self.label_entries = label_entries
        self.image_entries = image_entries
        self.device = torch.device('cuda:0')

    def __call__(self,
                 sample: dict
                 ) -> dict:
        # pylint: disable=too-many-branches

        # Unsqueeze samples to be 4-D tensors, as required by TorchIO
        require_squeeze = False
        for entry in self.entries:
            if entry not in sample:
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)
            if np_entry.ndim == 3:
                sample[entry] = np.expand_dims(np_entry, -1)
                require_squeeze = True
            elif np_entry.ndim == 4:
                sample[entry] = np_entry
            else:
                raise ValueError('The number of dimensions must be 3 or 4')

        subject_dict = {}
        for key in self.label_entries:
            subject_dict.update({key: tio.LabelMap(tensor=sample[key])})
        for key in self.image_entries:
            subject_dict.update({key: tio.ScalarImage(tensor=sample[key])})

        tio_sample = tio.Subject(subject_dict)
        for transform in self.transforms:
            tio_sample = transform(tio_sample)

        for key in self.label_entries:
            sample[key] = tio_sample.get(key).numpy()
        for key in self.image_entries:
            sample[key] = tio_sample.get(key).numpy()

        # squeeze samples back to original format
        if require_squeeze:
            for entry in self.entries:
                if isinstance(sample[entry], np.ndarray):
                    np_entry = sample[entry]
                else:
                    np_entry = check_and_return(sample[entry].numpy(), np.ndarray)
                sample[entry] = np_entry.squeeze(-1)

        return sample


class BaseAugmentation(ABC):
    """A base class for augmentations."""

    def __init__(self,
                 image_categories: tuple,
                 label_categories: tuple,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.image_categories = image_categories
        self.label_categories = label_categories
        self.categories = self.image_categories + self.label_categories

    @abstractmethod
    def get_transforms(self) -> Transform:
        """Returns the transformations of the augmentation procedure.
        Returns:
            Transform: The transforms.
        """
        raise NotImplementedError


class Segmentation2DAugmentation(BaseAugmentation):
    """A class for the standard augmentation procedure of 2D-models."""

    def __init__(self,
                 image_categories: tuple = ('images', 'ct'),
                 label_categories: tuple = ('gtv', 'oar', 'mask'),
                 interpolation_method: str = 'linear',
                 **kwargs
                 ) -> None:
        super().__init__(image_categories, label_categories, **kwargs)

        if interpolation_method not in ('linear', 'NEAREST'):
            raise ValueError(f'The selected interpolation method ({interpolation_method}) is not available!')

        self.interpolation_method = interpolation_method

    def get_transforms(self) -> Transform:
        transforms = [Permute(permutation=(2, 0, 1), entries=self.categories)]

        spatial_transforms = {
            tio.RandomAffine(scales=(1, 1.2, 1, 1.2, 1, 1),
                             degrees=(0, 0, 0, 0, -15, 15),
                             isotropic=False,
                             translation=(10, 10, 0),
                             default_pad_value='mean',
                             image_interpolation=self.interpolation_method,
                             include=self.categories): 0.5,
            tio.RandomElasticDeformation(num_control_points=(5, 5, 4),
                                         max_displacement=(25, 25, 0),
                                         locked_borders=1,
                                         image_interpolation=self.interpolation_method,
                                         include=self.categories): 0.5,
            tio.RandomFlip(axes=(0, 1), include=self.categories): 0.5,
        }

        intensity_transforms = {
            tio.RandomNoise(std=(0, 0.1), include=self.image_categories): 0.5,
            tio.RandomGamma(log_gamma=(-0.3, 0.3), include=self.image_categories): 0.3,
            tio.RandomMotion(degrees=5, translation=5, num_transforms=2, include=self.image_categories): 0.3,
        }

        augmentation_transforms = [
            TorchIOTransform(
                [
                    tio.OneOf(spatial_transforms, p=0.5, include=self.categories),
                    tio.OneOf(intensity_transforms, p=0.5, include=self.image_categories)
                ]
            )
        ]
        transforms.extend(augmentation_transforms)

        return ComposeTransform(transforms)


class RandomRotation(Transform):

    def __init__(self,
                 axis: int = 0,
                 p: float = 1.0,
                 angles: t.Tuple[float, ...] = (0, 40, 80, 120, 160, 200, 240, 280, 320),
                 entries: t.Tuple[str, ...] = ('images', 'dose', 'oar', 'ptv'),
                 interpolators: t.Tuple[str, ...] = ('bspline', 'bspline', 'nearest', 'nearest')) -> None:
        super().__init__()

        if not isinstance(axis, int):
            raise ValueError('axis must be an int')
        self.axis = axis

        if not 0. <= p <= 1.:
            raise ValueError('p must be in [0, 1]')
        self.p = p

        if not any(0 <= angle <= 360. for angle in angles):
            raise ValueError('Angles must be between 0 and 360.')
        self.angles = angles

        if not len(entries) == len(interpolators):
            raise ValueError('entries and interpolators must have the same length')
        self.entries = entries
        self.interpolators = interpolators

        if not len(angles) > 0:
            raise ValueError('At least one angle must be given')
        self.num_angles = len(angles)

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        angle_idx = np.random.randint(self.num_angles)
        angle = self.angles[angle_idx]

        for entry, interpolator in zip(self.entries, self.interpolators):
            if entry not in sample:
                raise ValueError('Entry {} not in sample'.format(entry))

            sample[entry] = self._rotate(sample[entry], angle, interpolator)

        return sample

    def _rotate(self,
                data: np.ndarray,
                angle: float,
                interpolator: str
                ) -> np.ndarray:
        angles = [0. for _ in range(data.ndim)]
        angles[self.axis] = float(np.radians(angle))

        centers = [float(d / 2) for d in data.shape]

        transform = sitk.Euler3DTransform()
        transform.SetCenter(centers)
        transform.SetRotation(*angles)

        if interpolator == 'nearest':
            interpolator_ = sitk.sitkNearestNeighbor
            min_value = int(np.min(data).round(0))

        elif interpolator == 'linear':
            interpolator_ = sitk.sitkLinear
            min_value = float(np.min(data))

        elif interpolator == 'bspline':
            interpolator_ = sitk.sitkBSpline
            min_value = float(np.min(data))

        else:
            raise ValueError('Unknown interpolator: {}'.format(interpolator))

        data_sitk = sitk.GetImageFromArray(data.reshape(data.shape[::-1]))
        data_sitk = sitk.Resample(data_sitk, data_sitk, transform, interpolator_, min_value)
        data_np = sitk.GetArrayFromImage(data_sitk).reshape(data.shape)

        return data_np


class RandomShift(Transform):

    def __init__(self,
                 shift: t.Union[int, tuple],
                 axis: t.Union[int, tuple] = None,
                 p: float = 1.0,
                 entries=('images', 'labels')):
        """Randomly shifts the sample along axes by a value from the interval [-p * size(axis), +p * size(axis)],
        where p is the percentage of shifting and size(axis) is the size along an axis.

        Args:
            shift (int, tuple): The percentage of shifting of the axis' size.
                If axis is not defined, the shifting will be applied from the first dimension onwards of the sample.
                Use None to exclude an axis or define axis to specify the axis/axes to crop.
                E.g.:

                - shift=0.2 with the default axis parameter shifts the sample along the 1st axis.
                - shift=(0.2, 0.1) with the default axis parameter shifts the sample along the 1st and 2nd axes.
                - shift=(None, 0.2) with the default axis parameter shifts the sample along the 2st axis.
                - shift=(0.2, 0.1) with axis=(1, 0) shifts the sample along the 1st and 2nd axes.
                - shift=(None, 0.1, 0.2) with axis=(1, 2, 0) shifts the sample along the 1st and 3rd axes.
            axis (int, tuple): Axis or axes to which the shift int or tuple correspond(s) to.
                If defined, must have the same length as shape.
            p (float): The probability of the shift to be applied.
            entries (tuple): The sample's entries to apply the shifting to.
        """
        super().__init__()
        if isinstance(shift, int):
            shift = (shift,)

        if axis is None:
            axis = tuple(range(len(shift)))
        if isinstance(axis, int):
            axis = (axis,)

        if len(axis) != len(shift):
            raise ValueError('If specified, the axis parameter must be of the same length as the shift')

        # filter out any axis where shift is None
        self.axis = tuple([a for a, s in zip(axis, shift) if s is not None])
        self.shift = tuple([s for s in shift if s is not None])

        self.p = p
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

        shifts_maximums = [int(s * sample[self.entries[0]].shape[a]) for a, s in zip(self.axis, self.shift)]
        shifts = [np.random.randint(-s_max, s_max) if s_max != 0 else 0 for s_max in shifts_maximums]

        for entry in self.entries:
            for axis, shift in zip(self.axis, shifts):
                modified = np.roll(sample[entry], shift, axis)

                if shift > 0:
                    slicing = [slice(None) for _ in range(sample[entry].ndim)]
                    slicing[axis] = slice(0, shift)
                    modified[tuple(slicing)] = np.min(sample[entry])

                elif shift < 0:
                    slicing = [slice(None) for _ in range(sample[entry].ndim)]
                    slicing[axis] = slice(sample[entry].shape[axis] + shift, sample[entry].shape[axis])
                    modified[tuple(slicing)] = np.min(sample[entry])

                else:
                    continue

                sample[entry] = modified

        return sample


class RandomNoise(Transform):

    def __init__(self,
                 mean: float,
                 std: float = 0.003,
                 p: float = 1.0,
                 entries=('images',),
                 seed: int = 543210
                 ) -> None:
        super().__init__()

        self.mean = mean
        self.std = std if std > 0 else 0
        self.p = p
        self.entries = entries

        self.random_state = np.random.RandomState(seed)

    def _add_noise(self, data: np.ndarray, std: float, mean: float = 0) -> np.ndarray:
        noise = self.random_state.normal(mean, std, data.shape)
        image = data + noise
        return image.astype(data.dtype)

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        std = self.random_state.uniform(0, self.std)

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            sample[entry] = self._add_noise(sample[entry], std)

        return sample
