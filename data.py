from typing import Tuple, Union, Generator

import cv2
import keras.backend as K
import numpy as np
import pandas as pd

if K.backend() == 'tensorflow':
    K.set_image_data_format('channels_last')
else:
    raise ValueError('This package only works with Tensorflow at the moment.')


def unet_flow_from_csv(filename: str, batch_size: int, img_shape: Union[Tuple[int, int], Tuple[int, int, int]] = (512, 512, 3), seed=None,
                       fraction_validation: float = 0.1) -> Tuple[Generator, Generator, int, int]:
    """
    Use a csv of matching image/mask pairs to load minibatches of
    images for training and validation. The csv should be set up so that
    it is in 2 columns with the headers 'image_filepath' and
    'mask_filepath' (in that order). The filepaths should be able to be
    resolved from whatever folder this script is being run. (This is
    linux only for now.)
    """

    filepaths_df = pd.read_csv(filename).sample(frac=1, random_state=seed)
    validation_filepaths_df = filepaths_df[:int(len(filepaths_df) * fraction_validation)]
    training_filepaths_df = filepaths_df[int(len(filepaths_df) * fraction_validation):]

    train_generator = _get_generator_from_dataframe(dataframe=training_filepaths_df, batch_size=batch_size, img_shape=img_shape)
    validation_generator = _get_generator_from_dataframe(dataframe=validation_filepaths_df, batch_size=batch_size, img_shape=img_shape)

    return train_generator, validation_generator, len(training_filepaths_df), len(validation_filepaths_df)


def _get_generator_from_dataframe(dataframe: pd.DataFrame, batch_size: int, img_shape: Union[Tuple[int, int], Tuple[int, int, int]]) -> Generator:
    """
    Utility function to compile a generator from a given Pandas
    DataFrame. The DataFrame is expected to be in the format that the
    unet_flow_from_csv function uses.
    """
    for start in range(0, len(dataframe), batch_size):
        end = min(start + batch_size, len(dataframe))
        image_batch = np.zeros((end - start) * np.prod(img_shape), dtype=np.float32).reshape((end - start), *img_shape)
        mask_batch = np.zeros((end - start) * np.prod(img_shape), dtype=np.float32).reshape((end - start), *img_shape)

        for i, (_, row) in enumerate(dataframe[start:end].iterrows()):
            image_batch[i] = cv2.imread(row['image_filepath'], cv2.IMREAD_COLOR) / 255.0
            mask_batch[i] = cv2.imread(row['mask_filepath'], cv2.IMREAD_COLOR) / 255.0

        yield image_batch, mask_batch
