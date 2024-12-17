from abc import abstractmethod

import numpy as np


class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)