import cv2
import numpy as np
from skimage.filters import gaussian

class COLORS:

    RED = 0
    GREEN = 1
    BLUE = 2

    N_COLORS = 3

    @staticmethod
    def get_by_id(id):
        return [COLORS.RED, COLORS.GREEN, COLORS.BLUE][id]

    @staticmethod
    def get_name(id):
        return ['RED', 'GREEN', 'BLUE'][id]

class SHAPES:

    SQUARE = 0
    TRIANGLE = 1
    CIRCLE = 2

    N_SHAPES = 3

    @staticmethod
    def get_by_id(id):
        return [SHAPES.SQUARE, SHAPES.TRIANGLE, SHAPES.CIRCLE][id]

    @staticmethod
    def get_name(id):
        return ['SQUARE', 'TRIANGLE', 'CIRCLE'][id]

class Button:

    def __init__(self, color_id, shape_id):
        self._color_id = color_id
        self._shape_id = shape_id

        if self._color_id == COLORS.RED:
            self._color = (255, 0, 0)
        elif self._color_id == COLORS.GREEN:
            self._color = (0, 255, 0)
        elif self._color_id == COLORS.BLUE:
            self._color = (0, 0, 255)
        else:
            raise ValueError('Unknown color id! Got {}'.format(color_id))

    @property
    def color(self):
        return self._color_id

    @property
    def shape(self):
        return self._shape_id

    def render(self, im_size=64):
        im = np.zeros((im_size, im_size, 3)).astype('uint8')

        length = int(im_size * 0.9)
        diff = (im_size - length) // 2

        if self._shape_id == SHAPES.SQUARE:            
            im = cv2.rectangle(im, (diff, diff), (length + diff, length + diff), self._color, -1)
        elif self._shape_id == SHAPES.CIRCLE:
            im = cv2.circle(im, (im_size//2, im_size//2), length//2, self._color, -1)
        elif self._shape_id == SHAPES.TRIANGLE:
            ct = np.array([im_size, im_size]) / 2 + np.array([diff * 2, 0])
            r = length / np.sqrt(3)
            pts = np.array([
                ct + np.array([-r, 0]),
                ct + np.array([r/2, -length/2]),
                ct + np.array([r/2, length/2]),
            ]).astype('int')
            im = cv2.fillConvexPoly(im, pts.reshape(-1, 1, 2), self._color)
        
        return gaussian(im, multichannel=True)

    @staticmethod
    def sample(n_samples):
        bs = []
        for _ in range(n_samples):
            color = COLORS.get_by_id(int(np.random.rand() * COLORS.N_COLORS))
            shape = SHAPES.get_by_id(int(np.random.rand() * SHAPES.N_SHAPES))
            bs.append(Button(color, shape))

        return bs
