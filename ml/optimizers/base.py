
from ml.base import Trainable
from abc import ABC, abstractmethod

class Optimizer(ABC):

    @abstractmethod
    def pre_update_params(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_params(self, trainable: Trainable) -> None:
        raise NotImplementedError

    @abstractmethod
    def post_update_params(self) -> None:
        raise NotImplementedError
