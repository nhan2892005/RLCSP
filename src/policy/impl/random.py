from ..policy import Policy
import random

class RandomPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Random choice a stock idx
                pos_x, pos_y = None, None
                for _ in range(100):
                    # random choice a stock
                    stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                    stock = observation["stocks"][stock_idx]

                    # Random choice a position
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x = random.randint(0, stock_w - prod_w)
                        pos_y = random.randint(0, stock_h - prod_h)
                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x = random.randint(0, stock_w - prod_h)
                        pos_y = random.randint(0, stock_h - prod_w)
                        if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                            prod_size = prod_size[::-1]
                            break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}