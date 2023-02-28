"""
@Author: Nicolo' Bonettini
@Author: Luca Bondi
@Author: Francesco Picetti
"""
import random
import numpy as np
from skimage.util import view_as_windows, view_as_blocks


# Score functions ---

def mid_intensity_high_texture(in_content):
    """
    Quality function that returns higher scores for mid intensity patches with high texture levels. Empirical.
    :type in_content: ndarray
    :param in_content : 2D or 3D ndarray. Values are expected in [0,1] if in_content is float, in [0,255] if in_content is uint8
    :return score: float
        score in [0,1].
    """

    if in_content.dtype == np.uint8:
        in_content = in_content / 255.

    mean_std_weight = .7

    in_content = in_content.flatten()

    mean_val = in_content.mean()
    std_val = in_content.std()

    ch_mean_score = -4 * mean_val ** 2 + 4 * mean_val
    ch_std_score = 1 - np.exp(-2 * np.log(10) * std_val)

    score = mean_std_weight * ch_mean_score + (1 - mean_std_weight) * ch_std_score
    return score


def count_patches(in_size, patch_size, patch_stride):
    """
    Compute the number of patches
    :param in_size:
    :param patch_size:
    :param patch_stride:
    :return:
    """
    win_indices_shape = (((np.array(in_size) - np.array(patch_size))
                          // np.array(patch_stride)) + 1)
    return int(np.prod(win_indices_shape))


class PatchExtractor:

    def __init__(self, dim, offset=None, stride=None, rand=None, function=None, threshold=None,
                 num=None, indexes=None):

        """
        N-dimensional patch extractor
        Args:
        :param in_content : ndarray
            the content to process as a numpy array of ndim dimensions

        :param dim : tuple
            patch_array dimensions as a tuple of ndim elements

        Named args:
        :param offset : tuple
            the offsets along each axis as a tuple of ndim elements

        :param stride : tuple
            the stride of each axis as a tuple of ndim elements

        :param rand : bool
            randomize patch_array order. Mutually exclusive with function_handler

        :param function : function
            patch quality function handler. Mutually exclusive with rand

        :param threshold: float
            minimum quality threshold

        :param num : int
            maximum number of returned patch_array.  Mutually exclusive with indexes

        :param indexes : list|ndarray
            explicitly return corresponding patch indexes (function_handler or C order used to index patch_array).
            Mutually exclusive with num

        :return ndarray: patch_array
            array of patch_array
            if rand==False and function_handler==None and num==None and indexes==None:
                patch_array.ndim = 2 * in_content.ndim
            else:
                patch_array.ndim = 1 + in_content.ndim
        """

        # Arguments parser ---
        if not isinstance(dim, tuple):
            raise ValueError('dim must be a tuple')
        self.dim = dim

        ndim = len(dim)
        self.ndim = ndim

        if offset is None:
            offset = tuple([0] * ndim)
        if not isinstance(offset, tuple):
            raise ValueError('offset must be a tuple')
        if len(offset) != ndim:
            raise ValueError('offset must a tuple of length {:d}'.format(ndim))
        self.offset = offset

        if stride is None:
            stride = dim
        if not isinstance(stride, tuple):
            raise ValueError('stride must be a tuple')
        if len(stride) != ndim:
            raise ValueError('stride must a tuple of length {:d}'.format(ndim))
        self.stride = stride

        if rand is not None and function is not None:
            raise ValueError('rand and function cannot be set at the same time')

        if rand is None:
            rand = False
        if not isinstance(rand, bool):
            raise ValueError('rand must be a boolean')
        self.rand = rand

        if function is not None and not callable(function):
            raise ValueError('function must be a function handler')
        self.function_handler = function

        if threshold is None:
            threshold = 0.0
        if not isinstance(threshold, float):
            raise ValueError('threshold must be a float')
        self.threshold = threshold

        if num is not None and indexes is not None:
            raise ValueError('num and indexes cannot be set at the same time')

        if num is not None and not isinstance(num, int):
            raise ValueError('num must be an int')
        self.num = num

        if indexes is not None and not isinstance(indexes, list) and not isinstance(indexes, np.ndarray):
            raise ValueError('indexes must be an list or a 1d ndarray')
        if indexes is not None:
            indexes = np.array(indexes).flatten()
        self.indexes = indexes

        self.in_content_original_shape = None
        self.in_content_cropped_shape = None

    def extract(self, in_content):

        if not isinstance(in_content, np.ndarray):
            raise ValueError('in_content must be of type: ' + str(np.ndarray))

        if in_content.ndim != self.ndim:
            raise ValueError('in_content shape must a tuple of length {:d}'.format(self.ndim))
        
        self.in_content_original_shape = in_content.shape

        # Offset ---
        for dim_idx, dim_offset in enumerate(self.offset):
            dim_max = in_content.shape[dim_idx]
            in_content = in_content.take(range(dim_offset, dim_max), axis=dim_idx)

        # Patch list ---
        if self.dim == self.stride:
            in_content_crop = in_content
            for dim_idx in range(self.ndim):
                dim_max = (in_content.shape[dim_idx] // self.dim[dim_idx]) * self.dim[dim_idx]
                in_content_crop = in_content_crop.take(range(0, dim_max), axis=dim_idx)
            patch_array = view_as_blocks(in_content_crop, self.dim)
        else:
            patch_array = view_as_windows(in_content, self.dim, self.stride)

        patch_array = np.ascontiguousarray(patch_array)

        patch_idx = patch_array.shape[:self.ndim]
        self.in_content_cropped_shape = tuple((np.asarray(patch_idx) - 1) * np.asarray(self.stride) + np.asarray(self.dim))

        # Evaluate patch_array or rand sort ---
        if self.rand:
            patch_array.shape = (-1,) + self.dim
            random.shuffle(patch_array)
        else:
            if self.function_handler is not None:
                patch_array.shape = (-1,) + self.dim
                patch_scores = np.asarray(list(map(self.function_handler, patch_array)))
                sort_idxs = np.argsort(patch_scores)[::-1]
                patch_scores = patch_scores[sort_idxs]
                patch_array = patch_array[sort_idxs]
                patch_array = patch_array[patch_scores >= self.threshold]

        if self.num is not None:
            patch_array.shape = (-1,) + self.dim
            patch_array = patch_array[:self.num]

        if self.indexes is not None:
            patch_array.shape = (-1,) + self.dim
            patch_array = patch_array[self.indexes]

        return patch_array

    def extract_call(self, args):  # TODO: verify
        in_content = args.pop('in_content')
        dim = args.pop('dim')

        return self.extract(in_content)

    def reconstruct(self, patch_array):
        """
        Reconstruct the N-dim image from the patch_array that has been extracted previously
        :param patch_array: array of patches as output of patch_extractor
        :return:
        """
        # Arguments parser ---
        if not isinstance(patch_array, np.ndarray):
            raise ValueError('patch_array must be of type: ' + str(np.ndarray))

        ndim = patch_array.ndim // 2

        # if not isinstance(patch_stride, tuple):
        #     raise ValueError('patch_stride must be a tuple')
        # if len(patch_stride) != ndim:
        #     raise ValueError('patch_stride must be a tuple of length {:d}'.format(ndim))
        #
        # if not isinstance(image_shape, tuple):
        #     raise ValueError('patch_idx must be a tuple')
        # if len(image_shape) != ndim:
        #     raise ValueError('patch_idx must be a tuple of length {:d}'.format(ndim))

        patch_stride = self.stride
        image_shape = self.in_content_cropped_shape

        patch_shape = patch_array.shape[-ndim:]
        patch_idx = patch_array.shape[:ndim]
        image_shape_computed = tuple((np.array(patch_idx) - 1) * np.array(patch_stride) + np.array(patch_shape))
        if not image_shape == image_shape_computed:
            raise ValueError('There is something wrong with the dimensions!')

        if ndim > 4:
            raise ValueError('For now, it works only in 4D, sorry!')
        numpatches = count_patches(image_shape, patch_shape, patch_stride)
        patch_array_unwrapped = patch_array.reshape(numpatches, *patch_shape)
        image_recon = np.zeros(image_shape)
        norm_mask = np.zeros(image_shape)
        counter = 0

        for h in np.arange(0, image_shape[0] - patch_shape[0] + 1, patch_stride[0]):
            if ndim > 1:
                for i in np.arange(0, image_shape[1] - patch_shape[1] + 1, patch_stride[1]):
                    if ndim > 2:
                        for j in np.arange(0, image_shape[2] - patch_shape[2] + 1, patch_stride[2]):
                            if ndim > 3:
                                for k in np.arange(0, image_shape[3] - patch_shape[3] + 1, patch_stride[3]):
                                    image_recon[h:h + patch_shape[0], i:i + patch_shape[1], j:j + patch_shape[2],
                                    k:k + patch_shape[3]] += patch_array_unwrapped[counter, :, :, :, :]
                                    norm_mask[h:h + patch_shape[0], i:i + patch_shape[1], j:j + patch_shape[2],
                                    k:k + patch_shape[3]] += 1
                                    counter += 1
                            else:
                                image_recon[h:h + patch_shape[0], i:i + patch_shape[1],
                                j:j + patch_shape[2]] += patch_array_unwrapped[counter, :, :, :]
                                norm_mask[h:h + patch_shape[0], i:i + patch_shape[1], j:j + patch_shape[2]] += 1
                                counter += 1
                    else:
                        image_recon[h:h + patch_shape[0], i:i + patch_shape[1]] += patch_array_unwrapped[counter, :, :]
                        norm_mask[h:h + patch_shape[0], i:i + patch_shape[1]] += 1
                        counter += 1
            else:
                image_recon[h:h + patch_shape[0]] += patch_array_unwrapped[counter, :]
                norm_mask[h:h + patch_shape[0]] += 1
                counter += 1

        image_recon /= norm_mask

        return image_recon


def main():
    in_shape = (644, 481, 3)
    dim = (120, 120, 3)
    stride = (7, 90, 90, 3)
    offset = (1, 0, 0, 0)
    in_content = np.random.randint(256, size=in_shape).astype(np.uint8)
    # args = {'in_content': in_content,
    #         'dim': dim,
    #         'offset': offset,
    #         'stride': stride,
    #         }

    # patch_array = patch_extractor_call(args)
    pe = PatchExtractor(dim)
    patch_array = pe.extract(in_content)
    print('patch_array.shape = ' + str(patch_array.shape))
    img_recon = pe.reconstruct(patch_array)
    print('img_recon.shape = ' + str(img_recon.shape))


if __name__ == "__main__":
    main()
