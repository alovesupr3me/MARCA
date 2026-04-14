import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import convolve1d
from scipy.ndimage import convolve


class OENormalizerContextNoneZero:

    def __init__(self, context_bin):
        self.r = context_bin
        

    def norm(self, data, if_cis):
        
        if if_cis:
            data_oe, mask_nan = cropped_oe_norm_cis(data, self.r)
        
        else:
            data_oe, mask_nan = cropped_oe_norm_trans(data, self.r)            
        
        return data_oe, mask_nan
    

def collect_all_diags(matrix):
    m, n = matrix.shape
    all_diags = []
    diag_lens = []

    for offset in range(-m + 1, n):
        diag = matrix.diagonal(offset=offset)
        all_diags.append(diag)
        diag_lens.append(len(diag))

    max_len = max(diag_lens)
    num_diags = len(all_diags)
    
    diags_mat = np.full((num_diags, max_len), np.nan, dtype=matrix.dtype)
    for i, diag in enumerate(all_diags):
        diags_mat[i, :len(diag)] = diag

    offsets = np.arange(-m + 1, n)
    return diags_mat, offsets, diag_lens


def cropped_oe_norm_cis(data, context_bin):
    H, W = data.shape

    win_size = 2 * (context_bin) + 1

    diags_mat, offsets, len_idx = collect_all_diags(data)

    mask = ~((diags_mat == 0) | np.isnan(diags_mat))

    vals = np.where(mask, diags_mat, 0)

    kernel = np.ones(win_size)

    sum_vals = convolve1d(vals, kernel, axis=1, mode='constant', cval=0)
    count_vals = convolve1d(mask.astype(float), kernel, axis=1, mode='constant', cval=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        expected = sum_vals / count_vals
    expected[count_vals == 0] = 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normed_diag = np.where(mask, diags_mat / expected, np.nan)

    result = data.copy()
    for idx, offset in enumerate(offsets):
        diag = normed_diag[idx, :len_idx[idx]]
        if offset >= 0:
            i = np.arange(min(H, W - offset))
            j = i + offset
        else:
            j = np.arange(min(W, H + offset))
            i = j - offset
        result[i, j] = diag

    cropped = result[context_bin:H - context_bin, context_bin:W - context_bin]

    mask_nan = np.isnan(data)
    mask_nan = mask_nan[context_bin:H - context_bin, context_bin:W - context_bin].astype(float)

    return cropped, mask_nan


def collect_window_sum(matrix, k):

    H, W = matrix.shape
    sums = (matrix[2*k:, 2*k:] 
          - matrix[:H-2*k, 2*k:]
          - matrix[2*k:, :W-2*k] 
          + matrix[:H-2*k, :W-2*k])
    return sums


def cropped_oe_norm_trans(data, context_bin):

    H, W = data.shape

    mask = ~((data == 0) | np.isnan(data))

    vals = np.where(mask, data, 0)

    val_accum = vals.cumsum(axis=0).cumsum(axis=1)
    mask_accum = mask.cumsum(axis=0).cumsum(axis=1)

    sum_vals = collect_window_sum(val_accum, k=context_bin)
    count_vals = collect_window_sum(mask_accum, k=context_bin)

    with np.errstate(divide='ignore', invalid='ignore'):
        expected = sum_vals / count_vals
    expected[count_vals == 0] = 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cropped = vals[context_bin:H - context_bin, context_bin:W - context_bin]
        normed = np.where(mask[context_bin:H - context_bin, context_bin:W - context_bin], cropped / expected, np.nan)

    mask_nan = np.isnan(data)
    mask_nan = mask_nan[context_bin:H - context_bin, context_bin:W - context_bin].astype(float)

    return normed, mask_nan
