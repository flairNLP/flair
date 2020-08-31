import random
import numpy as np

class func():

    def choice(options: list):
        return random.choice(options)

    def uniform(bounds: list):
        if len(bounds) != 2:
            raise Exception("Please provide a upper and lower bound for uniform.")
        return np.random.uniform(bounds[0], bounds[1])