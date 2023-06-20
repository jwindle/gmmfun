# Jesse Windle <jesse@bayesfactor.net>, 2023

import pytest
import numpy as np


def test_similar_output(model_pair_and_data):
    mod0, mod1, data = model_pair_and_data
    mod0.opt_I(data['theta0'])
    mod0.update_both(10)
    mod1.opt_I(data['theta0'])
    mod1.update_both(10)
    # These shouldn't produce identical results, just close, unless we have lots of data
    assert np.all(np.isclose(mod1.theta, mod0.theta, atol=1e-1))
