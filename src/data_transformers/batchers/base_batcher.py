from abc import ABC, abstractmethod


class BaseBatcher(ABC):

    @abstractmethod
    def get_single_data_batch(self):
        pass

    @abstractmethod
    def reset(self):
        pass