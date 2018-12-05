import torch


class NormalizeRangeTanh(object):
    ''' Normalizes a tensor with values from [0, 1] to [-1, 1]. '''

    def __init__(self):
        pass

    def __call__(self, sample):
        sample = sample * 2.0 - 1.0
        return sample


class UnNormalizeRangeTanh(object):
    ''' Unnormalizes a tensor with values from [-1, 1] to [0, 1]. '''

    def __init__(self):
        pass

    def __call__(self, sample):
        sample = (sample + 1.0) * 0.5
        return sample


class UnNormalize(object):
    ''' from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3'''

    def __init__(self, mean, std):
        mean_arr = []
        for dim in range(len(mean)):
            mean_arr.append(dim)
        std_arr = []
        for dim in range(len(std)):
            std_arr.append(dim)
        self.mean = torch.Tensor(mean_arr).view(1, len(mean), 1, 1)
        self.std = torch.Tensor(std_arr).view(1, len(std), 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor *= self.std
        tensor += self.mean
        return tensor
