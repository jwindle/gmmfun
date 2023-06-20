# Jesse Windle <jesse@bayesfactor.net>, 2023

import pytest
import numpy as np


def test_similar_output(all_models_and_data):
    models, data = all_models_and_data
    for name, mod in models.items():
        mod.opt_I(data['theta0'])
        mod.update_both(10)
    for name, mod in models.items():
        print(name)
        # These shouldn't produce identical results, just close, unless we have lots of data
        assert np.all(np.isclose(mod.theta, models['cgf_4'].theta, atol=1e-2))
