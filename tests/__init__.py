import pandas as pd
import numpy as np
from autora.experimentalist.autora_experimentalist_example import sample


def test_sample_returns_dataframe():
    """Test that sample returns a pandas DataFrame with correct shape"""
    reference = pd.DataFrame(
        [(i/10, j/10) for i in range(10) for j in range(10)], 
        columns=['x1', 'x2']
    )
    tested = reference.iloc[[0, 10]]
    
    result = sample(
        conditions=tested,
        reference_conditions=reference,
        num_samples=5,
        epsilon=0.3,
        random_state=42
    )
    
    assert isinstance(result, pd.DataFrame), "Output must be a DataFrame"
    assert len(result) == 5, f"Expected 5 samples, got {len(result)}"
    assert list(result.columns) == ['x1', 'x2'], f"Expected columns ['x1', 'x2'], got {list(result.columns)}"


def test_sample_avoids_tested_conditions():
    """Test that sample doesn't return already tested points"""
    reference = pd.DataFrame(
        [(i, j) for i in range(5) for j in range(5)], 
        columns=['x1', 'x2']
    )
    tested = reference.iloc[[0, 1, 2]]
    
    result = sample(
        conditions=tested,
        reference_conditions=reference,
        num_samples=3,
        epsilon=0.0,  # No randomness, pure exploitation
        random_state=42
    )
    
    # Check no overlap with tested points
    tested_tuples = set(tested.apply(tuple, axis=1))
    result_tuples = set(result.apply(tuple, axis=1))
    overlap = tested_tuples & result_tuples
    
    assert len(overlap) == 0, f"Sample returned already tested points: {overlap}"


def test_sample_with_empty_conditions():
    """Test that sample works when no points have been tested yet (cold start)"""
    reference = pd.DataFrame(
        [(i, j) for i in range(5) for j in range(5)], 
        columns=['x1', 'x2']
    )
    empty_tested = pd.DataFrame(columns=['x1', 'x2'])
    
    result = sample(
        conditions=empty_tested,
        reference_conditions=reference,
        num_samples=3,
        epsilon=0.5,
        random_state=42
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert all(col in result.columns for col in ['x1', 'x2'])