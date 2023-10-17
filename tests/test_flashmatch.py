import numpy as np

from pfmatch.flashmatch_types import FlashMatch

# import pytest fixtures; do not remove
from tests.fixtures import rng


def test_flashmatch_filter_loss_matrix(rng):
    # Create a FlashMatch object with some fake data
    num_qclusters = int(rng.random()*50)+2
    num_flashes = int(rng.random()*50)+2
    flashmatch = FlashMatch(num_qclusters, num_flashes)
    flashmatch.loss_matrix = rng.random(size=(num_qclusters, num_flashes))
    
    # Set a loss threshold 10-90%
    loss_threshold = rng.random()*0.8 + 0.1

    # Filter the loss matrix
    *__, LM = flashmatch.filter_loss_matrix(loss_threshold)
    
    # check that all values <=loss_threshold are in the filtered loss matrix
    assert np.all([x in LM for x in flashmatch.loss_matrix[flashmatch.loss_matrix<=loss_threshold].ravel()])
    
    # filter none out
    *__, LM = flashmatch.filter_loss_matrix(1.5)
    assert np.allclose(LM, flashmatch.loss_matrix)
    
    # filter all out
    *__, LM = flashmatch.filter_loss_matrix(-np.inf)
    assert len(LM) == 0

        
"""TODO: tests for
local_match
global_match
bipartite_match
"""