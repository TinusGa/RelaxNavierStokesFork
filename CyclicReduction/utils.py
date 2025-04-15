import numpy as np

def back_substitution_indices(reduction_levels,offset=0):
    """
    Generate indices for back substitution in cyclic reduction.
    
    This convenience function computes and returns a list of lists containing
    the indices of the removed variables at each level of the reduction process.
    Each inner list corresponds to a specific reduction level.
    
    Parameters
    ----------
    reduction_levels : int
        The number of levels of reduction.
    offset : int
        The offset to apply to the indices.
    
    Returns
    -------
    list of list of int
        A list where each inner list contains the indices for back substitution.
    
    Examples
    --------
    >>> back_substitution_indices(3, 0)
    [[1, 3, 5, 7], [2, 6], [4]]
    
    >>> back_substitution_indices(3, 1)
    [[2, 4, 6, 7], [3, 7], [5]]
    """
    rows = int(2**reduction_levels)
    indices = np.arange(offset, rows + offset)
    index_list = []
    for _ in range(reduction_levels):
        even_indices = indices[::2]
        odd_indices = indices[1::2]
        index_list.append(odd_indices)
        indices = even_indices

    return index_list