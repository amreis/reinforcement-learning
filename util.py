"""
We need some utility functions for binning, for example. I think that's all for now, in fact.
"""
from gym.spaces import Box
import numpy as np

class Binner:
    """
    A binner will take a vector of continuous values and turn them into integers,
    so that we can have a discrete representation of this space.
    This process is done by determining where each value belongs if we create
    bins on the space.
    The idea is to use this with the Box space type from OpenAI Gym.
    """
    def __init__(self, box_or_low_limits, high_limits=None, step=None, lo=None, hi=None, centered=True):
        if isinstance(box_or_low_limits, Box):
            low_limits = box_or_low_limits.low
            high_limits = box_or_low_limits.high
        else:
            low_limits = box_or_low_limits

        if step is not None:
            pass
        elif lo is not None and hi is not None:
            pass
        else:
            raise ValueError("Either step should be informed or lo and hi")
        self._mapper = Binner._map(box_or_low_limits, high_limits, lo, hi)

    @staticmethod
    def _map(source_lo, source_hi, target_lo, target_hi):
        def func(data):
            transformed = target_lo + ((data - source_lo) / (source_hi - source_lo)) * (target_hi - target_lo)
            # Now clip!
            transformed[transformed > target_hi] = target_hi + 1
            transformed[transformed < target_lo] = target_lo - 1
            return np.ceil(transformed).astype(int)
        return func

    def transform(self, data):
        return self._mapper(data)

class LinearApproximator:
    def __init__(self, feature_extractor, feature_vec_dim):
        self._feature_extractor = feature_extractor
        self._w = np.random.rand(feature_vec_dim)

    @property
    def feature_extractor(self):
        return self._feature_extractor

    def state_value(self, state):
        feats = self.feature_extractor(state)
        return np.dot(self._w, feats)

    def grad(self, state):
        return state

    def update_w(self, delta, learning_rate=0.01):
        self._w += learning_rate * delta