from typing import Sequence, TypeVar, Optional

T = TypeVar('T')

def is_subsequence(subsequence: Sequence[T], sequence: Sequence[T]) -> bool:
    """Check if a subsequence exists within a sequence.
    
    A subsequence is a sequence that can be derived from another sequence by 
    deleting some or no elements without changing the order of the remaining elements.
    
    Args:
        subsequence: The subsequence to check for
        sequence: The main sequence to search in
        
    Returns:
        True if subsequence is a subsequence of sequence, False otherwise
        
    Examples:
        >>> is_subsequence([1, 3, 5], [1, 2, 3, 4, 5])
        True
        >>> is_subsequence("abc", "ahbgdc")
        True
        >>> is_subsequence("axc", "ahbgdc")
        False
    """
    if not subsequence:
        return True
    
    if not sequence:
        return False
    
    sub_idx = 0
    seq_idx = 0
    
    while sub_idx < len(subsequence) and seq_idx < len(sequence):
        if subsequence[sub_idx] == sequence[seq_idx]:
            sub_idx += 1
        seq_idx += 1
        
    # If we've gone through the entire subsequence, it exists in the sequence
    return sub_idx == len(subsequence)


def find_subsequence_indices(subsequence: Sequence[T], sequence: Sequence[T]) -> list[int]:
    """Find the indices in the main sequence where elements of the subsequence occur.
    
    Args:
        subsequence: The subsequence to find
        sequence: The main sequence to search in
        
    Returns:
        A list of indices in the main sequence that correspond to the subsequence elements
        Empty list if subsequence is not found
        
    Examples:
        >>> find_subsequence_indices([1, 3, 5], [1, 2, 3, 4, 5])
        [0, 2, 4]
        >>> find_subsequence_indices("abc", "ahbgdc")
        [0, 2, 5]
    """
    if not subsequence:
        return []
    
    if not sequence:
        return []
    
    indices = []
    sub_idx = 0
    
    for seq_idx, item in enumerate(sequence):
        if sub_idx < len(subsequence) and subsequence[sub_idx] == item:
            indices.append(seq_idx)
            sub_idx += 1
    
    # If we didn't find the complete subsequence, return empty list
    if sub_idx != len(subsequence):
        return []
        
    return indices