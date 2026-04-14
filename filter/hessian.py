import math
from scipy import signal
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


class Hessian:

    def __init__(self):

        self.defaultoptions = {}
        # self.defaultoptions = {'FrangiScaleRange': (3, 11), 'FrangiScaleRatio': 2,
        #               'verbose': False, 'BlackWhite': True} 


    def process(self, X):
        """
        input: X
        output: {"ellipsoid": , "peak":, "stripe": , ...}
        """

        # padding, against interference from "image" border
        padded = np.pad(X, pad_width=30, mode='symmetric')

        response_ell = SpeedUpFilter(padded)

        # cut back to the original size
        response_ell = response_ell[30:-30, 30:-30]

        return {"ellipse": response_ell}


def SpeedUpFilter(image):
    image = np.array(image, dtype=float)
    options = {'FrangiScaleRange': (3, 9), 'FrangiScaleRatio': 2, 'BlackWhite': True}
    sigmas = np.arange(options['FrangiScaleRange'][0], options['FrangiScaleRange'][1], options['FrangiScaleRatio'])
    sigmas.sort()

    filteredd_list = []

    for sigma in sigmas:
        hessians = hessian_matrix(image, sigma=sigma, order='rc', use_gaussian_derivatives=True)
        hessians = [h * sigma**2 for h in hessians]  # sigma **2 * hessian
        lambda1, lambda2 = hessian_matrix_eigvals(hessians)  # |lambda1| ≤ |lambda2|

        if options['BlackWhite']:
            R1 = np.where(lambda1 < 0, -lambda1, np.spacing(1))
            R2 = np.where(lambda2 < 0, -lambda2, np.spacing(1))
        else:
            R1 = np.where(lambda1 > 0, lambda1, np.spacing(1))
            R2 = np.where(lambda2 > 0, lambda2, np.spacing(1))

        R = R1 * R2
        S = (2 * R) / (R1 ** 2 + R2 ** 2)

        # Compute the output image
        Ifiltered = R * (3 - 2 * S)

        filteredd_list.append(Ifiltered)

    filtered = np.stack(filteredd_list, axis=-1)
    if len(sigmas) > 1:
        outIm = filtered.max(2)
    else:
        outIm = filtered[:, :, 0]

    return outIm
