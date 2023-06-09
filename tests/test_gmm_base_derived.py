# Jesse Windle <jesse@bayesfactor.net>, 2023

import pytest
import numpy as np


def test_loss_components(model_and_data):
    mod, data = model_and_data
    theta = data['theta']
    lc1 = mod.loss_components(theta)
    lce = mod.loss_components_expanded(theta)
    lc2 = np.mean(lce, axis=0)
    assert np.all(np.isclose(lc1, lc2))


