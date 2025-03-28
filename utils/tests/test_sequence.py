"""Tests for sequence utilities."""

import pytest
from utils.sequence import is_subsequence, find_subsequence_indices


def test_is_subsequence():
    """Test the is_subsequence function."""
    # Test with lists
    assert is_subsequence([1, 3, 5], [1, 2, 3, 4, 5])
    assert is_subsequence([], [1, 2, 3])
    assert not is_subsequence([1, 2, 3], [])
    assert not is_subsequence([1, 3, 7], [1, 2, 3, 4, 5])
    
    # Test with strings
    assert is_subsequence("abc", "ahbgdc")
    assert is_subsequence("", "ahbgdc")
    assert not is_subsequence("axc", "ahbgdc")
    
    # Test with mixed types
    assert is_subsequence([1, "b", 3], [1, "a", "b", 2, 3])


def test_find_subsequence_indices():
    """Test the find_subsequence_indices function."""
    # Test with lists
    assert find_subsequence_indices([1, 3, 5], [1, 2, 3, 4, 5]) == [0, 2, 4]
    assert find_subsequence_indices([], [1, 2, 3]) == []
    assert find_subsequence_indices([1, 2, 3], []) == []
    assert find_subsequence_indices([1, 3, 7], [1, 2, 3, 4, 5]) == []
    
    # Test with strings
    assert find_subsequence_indices("abc", "ahbgdc") == [0, 2, 5]
    assert find_subsequence_indices("", "ahbgdc") == []
    assert find_subsequence_indices("axc", "ahbgdc") == []
    
    # Test with repeated elements
    assert find_subsequence_indices([1, 1], [1, 2, 1, 1, 3]) == [0, 2]
    assert find_subsequence_indices("aa", "abaa") == [2, 3]