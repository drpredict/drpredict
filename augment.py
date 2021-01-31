from abc import ABC, abstractmethod

# from skimage.filters import gaussian
import numpy as np
import random
import cv2
from logs import get_logger
from typing import Callable
import traceback
import inspect
import pickle
import os.path
import zlib
# import torchvision

logger = get_logger('augment.logger')


class Transform(ABC):
    counter = 0

    def __init__(self):
        self.id = Transform.counter
        Transform.counter += 1

    def __call__(self, *img):
        """
        this is the function that calls the actual transformation function.
        I added this layer for easy managing, such as caching, logging etc.
        DO NOT override this function.
        :param img: input image
        :param sample_id: optional id of input image, useful in caching
        :return:the transformed image
        """
        logger.debug(f'calling:{self.__class__.__name__}_{self.id}')
        if len(img) == 0 or img[0] is None:
            return None
        try:
            args = inspect.getfullargspec(self.transform)
            if args.varargs is None:
                output_image = (self.transform(img[0]), *img[1:])
            else:
                output_image = self.transform(*img)
        except Exception as e:
            logger.error(e)
            traceback.print_tb(e.__traceback__)
            raise e
            return None
        if type(output_image) in (list, tuple) and len(output_image) == 1:
            logger.debug('single output')
            return output_image[0]
        # logger.debug(output_image.shape)
        return output_image

    @abstractmethod
    def transform(self, img):
        """
        This is the actual transformation function
        :param img:
        :return:
        """
        raise NotImplementedError()


class CompostImageAndLabel(Transform):
    """
    the first one is input image and the following
    images are labels.
    """
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def transform(self, *images):
        img = images[0]
        label = list(images[1:])
        for i in self.transforms:
            if type(i) in (tuple, list):
                imgtrans, labelTrans = i
                if imgtrans is not None:
                    logger.debug(
                        f'call image transform '
                        f'{imgtrans.__class__.__name__} on {type(img)}')
                    img = imgtrans(img)
                if labelTrans is not None:
                    for labelIndex in range(len(label)):
                        label[labelIndex] = i[1](label[labelIndex])
            else:
                outputs = i(img, *label)
                if type(outputs) in (tuple, list):
                    img = outputs[0]
                    label = list(outputs[1:])
                else:
                    logger.debug(f'single output {label}')
                    img = outputs
        if label is None or len(label) == 0:
            logger.debug(f'single output {img.shape}')
            return img
        if len(label) == 1:
            label = label[0]
            return img, label
        return (img, *label)

    def init_env(self):
        for t in self.transforms:
            if hasattr(t, 'init_env'):
                t.init_env()


class ImageReader(Transform):

    def transform(self, img):
        if img is None:
            return None
        if type(img) is np.ndarray:
            logger.error("dont need do this")
            pass
        elif type(img) is str:
            img = cv2.imread(img)
        elif type(img) is bytes:
            encoded = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        elif hasattr(img, 'read'):
            encoded = img.read()
            encoded = np.frombuffer(encoded, np.uint8)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return img


class ImageSaver(Transform):
    def transform(self, img):
        if img is None:
            logger.error('No image')
            return None
        # if type(img) not in (tuple, list) or len(img) != 2:
        #     logger.error('input type not correct')
        #     return None
        # name, img = img
        if img is None:
            return None
        try:
            succ, result = cv2.imencode('.png', img)
            if not succ:
                return None
            return result.tobytes()
        except Exception as e:
            traceback.print_exc()
            return None
        return name

    def __init__(self):
        super().__init__()


class Identity(Transform):
    """Identity function, return as it is"""

    def __init__(self):
        super().__init__()

    def transform(self, img):
        return img


class RandomCrop(Transform):
    def __init__(self, size):
        super().__init__()
        if type(size) not in (list, tuple):
            size = (size, size)
        self.size = size

    def transform(self, *imgs):
        ranges = (
            imgs[0].shape[0] - self.size[0],
            imgs[0].shape[1] - self.size[1]
        )
        start_point = [random.randint(0, ranges[i]) for i in range(2)]
        result = []
        for img in imgs:
            indices = [slice(start_point[i], start_point[i]+self.size[i]) for
                       i in range(2)]
            indices += [slice(0, l) for l in img.shape[2:]]
            indices = tuple(indices)
            result.append(img[indices])
        return result


class ResizeKeepAspectRatio(Transform):
    """resize image and keep the aspect ratio, filling
        empty area with black.
    """
    def __init__(self, size):
        super().__init__()
        if type(size) not in (tuple, list):
            size = (size, size)
        self.size = size

    def _resize(self, img):
        ratio = self.size[0] / img.shape[0]
        ratio = max(ratio, self.size[1] / img.shape[1])
        if ratio == 1:
            return img
        target_size = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        img = cv2.resize(img, target_size[::-1])
        # if len(img.shape) == 3:
        #     BLACK = [0, 0, 0]
        # else:
        #     BLACK = 0
        # logger.debug(img.shape)
        # img = cv2.copyMakeBorder(
        #     img,
        #     (self.size[0] - img.shape[0]) // 2,
        #     self.size[0] - (self.size[0] - img.shape[0]) // 2 - img.shape[0],
        #     (self.size[1] - img.shape[1]) // 2,
        #     self.size[1] - (self.size[1] - img.shape[1]) // 2 - img.shape[1],
        #     cv2.BORDER_CONSTANT, value=BLACK
        # )
        return img

    def transform(self, *imgs):
        result = [self._resize(x) for x in imgs]
        return result


class Resize(Transform):
    """resize image so thar it fit the net"""

    def __init__(self, size=229):
        super().__init__()
        self.size = size

    def transform(self, *imgs):
        results = []
        for img in imgs:
            if len(img.shape) == 3:
                result = np.zeros((self.size, self.size, img.shape[2]),
                                  img.dtype)
                cv2.resize(
                    img, result.shape[:2],
                    result, interpolation=cv2.INTER_LINEAR)
            elif len(img.shape) == 2:
                result = np.zeros((self.size, self.size), img.dtype)
                cv2.resize(img, result.shape[:2], result)
            results.append(result)
        if len(results) == 1:
            return results[0]
        return results


class RangeLimit(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, img):
        if np.issubdtype(img.dtype, np.floating):
            img[img > 1] = 1
            img[img < 0] = 0
        elif np.issubdtype(img.dtype, np.integer):
            img[img > 255] = 255
            img[img < 0] = 0
        return img


class RandomNoise(Transform):
    """docstring for RandomNoise"""

    def __init__(self, mean=0.1, var=0.02, amount=0.001, salt_vs_pepper=0.5):
        super().__init__()
        self.mean = mean
        self.var = var
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def transform(self, image):
        assert np.issubdtype(image.dtype, np.floating), str(image.dtype)
        maxval = np.max(image)
        var = [maxval * self.var * (random.random() + 0.01) for _ in range(3)]
        mean = [maxval * self.mean * (random.random() - 0.5) * 2
                for _ in range(3)]
        var = [i if i > 0 else 0 for i in var]
        noise = np.random.normal(mean, var, image.shape)
        image += noise
        return image


class Normalize(Transform):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.mean = mean
        self.std = std

    def transform(self, image):
        return ((image - self.mean) / self.std)


class ToTensor(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, img):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        logger.debug(f'type is: {type(img)}, {img}')
        assert np.issubdtype(img.dtype, np.floating)
        # logger.info(img.shape)
        return img.transpose((2, 0, 1))


class ToFloat(Transform):
    """docstring for ToFloat"""

    def __init__(self):
        super().__init__()

    def transform(self, img):
        if img.dtype == np.float32 or \
                img.dtype == np.float64 or\
                img.dtype == np.float:
            return img
        elif img.dtype == np.uint8:
            return img.astype(np.float32) / 255
        else:
            raise Exception('Image data type not recognized')


class ToInt(Transform):
    """docstring for ToFloat"""

    def __init__(self):
        super().__init__()

    def transform(self, img):
        if img.dtype in [np.float32, np.float64, np.float]:
            return (img * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            return img
        else:
            raise Exception('Image data type not recognized')


class ToNumpyType(Transform):
    """docstring for ToFloat"""

    def __init__(self, typeClass):
        super().__init__()
        self.typeClass = typeClass

    def transform(self, img):
        return img.astype(self.typeClass)


class GlobalNorm(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, img):
        return (img - mean) / std


class PerImageNorm(Transform):
    """docstring for PerImageNorm"""

    def __init__(self):
        super(PerImageNorm, self).__init__()

    def transform(self, img):
        # mask = (image[:, :, 1] > 5 / 255) * (image[:, :, 1] < 250 / 255)
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        img = (img - mean) / std
        # image[:, :, 0] *= mask
        # image[:, :, 1] *= mask
        # image[:, :, 2] *= mask
        return img


class RangeCenter(Transform):
    """docstring for PerImageNorm"""

    def __init__(self):
        super(RangeCenter, self).__init__()

    def transform(self, image):
        # assert image.max() <= 1
        # assert image.min() >= 0
        assert image.dtype in [np.float64, np.float, np.float32]
        image -= 0.5
        image *= 2
        return image


class LocalNorm(Transform):
    """docstring for PerImageNorm"""

    def __init__(self):
        super(LocalNorm, self).__init__()

    def transform(self, image):
        filtered_image = gaussian(image, multichannel=True, sigma=5)
        normed = filtered_image - image
        return normed


class Compose(Transform):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list
            of transforms to compose.
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, img):
        self.init_env()
        if img is None:
            return None
        for tran in self.transforms:
            try:
                img = tran(img)
            except Exception as e:
                logger.error(e)
                return None
            if img is None:
                return None
        return img

    def transform(self, img):
        raise NotImplementedError

    def init_env(self):
        for t in self.transforms:
            if hasattr(t, 'init_env'):
                t.init_env()


class FundusAOICrop(Transform):
    """Crop images so that the area of interest is centered"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def find_at_thresh(arr, step, thresh=10):
        start = int(len(arr) / 2)
        while 0 < start < len(arr) and thresh < arr[start] < 255 - thresh:
            start += step
        return start

    def find_attr(self, img):
        assert img.dtype == np.uint8, AssertionError(str(img.dtype))
        gchannel = img[:, :, 1]
        rsum = gchannel.sum(axis=0) / gchannel.shape[0]
        csum = gchannel.sum(axis=1) / gchannel.shape[1]
        rmin = self.find_at_thresh(csum, -1)
        rmax = self.find_at_thresh(csum, 1)
        cmin = self.find_at_thresh(rsum, -1)
        cmax = self.find_at_thresh(rsum, 1)
        if rmax - rmin < 50 or cmax - cmin < 50:
            rmin = 0
            rmax = len(csum)
            cmin = int((len(rsum) - len(csum)) / 2)
            cmax = len(rsum) - 2 * cmin
        return rmax, rmin, cmax, cmin

    def transform(self, *images):
        assert len(images) > 0
        img = images[0]
        restImage = list(images[1:])
        rmax, rmin, cmax, cmin = self.find_attr(img)
        img = img[rmin:rmax, cmin:cmax, :]
        for i in range(len(restImage)):
            restImage[i] = restImage[i][rmin:rmax, cmin:cmax]
        if len(restImage) == 0:
            logger.debug('single image')
            return img
        return (img, *restImage)


class FixedCrop(Transform):
    def __init__(self, s1, l1, s2, l2):
        super().__init__()
        self.crop = (s1, l1, s2, l2)

    def transform(self, img):
        assert self.crop[1] < img.shape[0], f'{self.crop[1]} < {img.shape[0]}'
        assert self.crop[3] < img.shape[1], f'{self.crop[3]} < {img.shape[1]}'
        return img[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]


class SingleChannel(Transform):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def transform(self, img):
        img = img[:, :, self.channel]
        return img


class RandRotate(Transform):
    """random rotate images"""

    def __init__(self):
        super().__init__()

    def transform(self, *images):
        assert len(images) > 0
        img = images[0]
        restImage = list(images[1:])
        rnd = random.random() * 360
        # scale = random.gauss(1, 0.08)
        transform_matrix = cv2.getRotationMatrix2D(
            (img.shape[0] // 2, img.shape[1] // 2), rnd, 1)
        # oimg = np.zeros(img.shape, dtype=img.dtype)
        oimg = cv2.warpAffine(img, transform_matrix, img.shape[:2])
        for i in range(len(restImage)):
            restImage[i] = cv2.warpAffine(
                restImage[i],
                transform_matrix,
                restImage[i].shape[:2])
        if len(restImage) == 0:
            return oimg
        return (oimg, *restImage)


class OneOf(Transform):
    """This class applies one of the funcs given"""

    def __init__(self, transforms: [Callable]):
        """
        :param transforms: must be none-empty array
        """
        super().__init__()
        assert transforms is not None and len(transforms) > 0
        self.transforms = transforms

    def __call__(self, img, sample_id=None):
        selection = random.choice(self.transforms)
        return selection(img, sample_id)

    def transform(self, img):
        raise NotImplementedError


class GausianBlur(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, img):
        kernel_size = random.choice((3, 5, 7, 9))
        sigma = (random.random() * 0.75 + 0.25) ** 2.5 * 6
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        return blur


class AverageBlur(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, img):
        kernel_size = random.choice((3, 5, 7, 9))
        blur = cv2.blur(img, (kernel_size, kernel_size))
        return blur


class MedianBlur(Transform):
    def __init__(self):
        super().__init__()
        self.to_int = ToInt()
        self.to_floay = ToFloat()

    def transform(self, img):
        kernel_size = random.choice((3, 5, 7, 9))
        if np.issubdtype(img.dtype, np.floating):
            img = self.to_int(img)
            img = cv2.medianBlur(img, kernel_size)
            img = self.to_floay(img)
            return img
        return cv2.medianBlur(img, kernel_size)


class RandBlur(OneOf):
    def transform(self, img):
        raise NotImplementedError

    def __init__(self):
        super().__init__([
            AverageBlur(),
            MedianBlur(),
            GausianBlur(),
            Identity(),
        ])


class RandFlip(Transform):
    """random rotate images"""

    def __init__(self):
        super().__init__()

    def transform(self, *images):
        flip = list(images)
        if random.random() > 0.5:
            for i in range(len(flip)):
                flip[i] = (flip[i][:, ::-1]).copy()
        if random.random() > 0.5:
            for i in range(len(flip)):
                flip[i] = (flip[i][::-1, :]).copy()
        return flip


augment_map = dict(
    CompostImageAndLabel=CompostImageAndLabel,
    ImageReader=ImageReader,
    ImageSaver=ImageSaver,
    Identity=Identity,
    RandomCrop=RandomCrop,
    ResizeKeepAspectRatio=ResizeKeepAspectRatio,
    Resize=Resize,
    RangeLimit=RangeLimit,
    RandomNoise=RandomNoise,
    Normalize=Normalize,
    ToTensor=ToTensor,
    ToFloat=ToFloat,
    ToInt=ToInt,
    ToNumpyType=ToNumpyType,
    GlobalNorm=GlobalNorm,
    PerImageNorm=PerImageNorm,
    RangeCenter=RangeCenter,
    LocalNorm=LocalNorm,
    Compose=Compose,
    FundusAOICrop=FundusAOICrop,
    FixedCrop=FixedCrop,
    SingleChannel=SingleChannel,
    RandRotate=RandRotate,
    OneOf=OneOf,
    GausianBlur=GausianBlur,
    AverageBlur=AverageBlur,
    MedianBlur=MedianBlur,
    RandFlip=RandFlip,
)
