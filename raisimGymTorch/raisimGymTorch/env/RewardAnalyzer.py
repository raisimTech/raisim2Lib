# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import numpy as np


class RewardAnalyzer:

    def __init__(self, env, writer):
        self.writer = writer
        self.data_tags = list(env.get_reward_names())
        self.data_size = 0
        self.data_mean = np.zeros(shape=(len(self.data_tags),), dtype=np.double)
        self.data_square_sum = np.zeros(shape=(len(self.data_tags),), dtype=np.double)
        self.data_min = np.inf * np.ones(shape=(len(self.data_tags),), dtype=np.double)
        self.data_max = -np.inf * np.ones(shape=(len(self.data_tags),), dtype=np.double)

    def add_reward_info(self, info):
        info = np.asarray(info)
        if info.ndim == 1:
            info = info.reshape(1, -1)
        if info.size == 0:
            return

        self.data_size += info.shape[0]
        self.data_square_sum += np.sum(info * info, axis=0, dtype=np.double)
        self.data_mean += np.sum(info, axis=0, dtype=np.double)
        self.data_min = np.minimum(self.data_min, np.min(info, axis=0))
        self.data_max = np.maximum(self.data_max, np.max(info, axis=0))

    def analyze_and_plot(self, step):
        if self.data_size == 0:
            return

        self.data_mean /= self.data_size
        data_std = np.sqrt((self.data_square_sum - self.data_size * self.data_mean * self.data_mean) / (self.data_size - 1 + 1e-16))

        for data_id in range(len(self.data_tags)):
            self.writer.add_scalar(self.data_tags[data_id]+'/mean', self.data_mean[data_id], global_step=step)
            self.writer.add_scalar(self.data_tags[data_id]+'/std', data_std[data_id], global_step=step)
            self.writer.add_scalar(self.data_tags[data_id]+'/min', self.data_min[data_id], global_step=step)
            self.writer.add_scalar(self.data_tags[data_id]+'/max', self.data_max[data_id], global_step=step)

        self.data_size = 0
        self.data_mean = np.zeros(shape=(len(self.data_tags),), dtype=np.double)
        self.data_square_sum = np.zeros(shape=(len(self.data_tags),), dtype=np.double)
        self.data_min = np.inf * np.ones(shape=(len(self.data_tags),), dtype=np.double)
        self.data_max = -np.inf * np.ones(shape=(len(self.data_tags),), dtype=np.double)
