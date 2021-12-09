
from codacore.model import CodaCore
from codacore.datasets import simulate_hts

def test_codacore(cv_params=None, opt_params=None):
    """Test codacore"""
    seed = 0
    n = 1000
    p = 100
    x, y = simulate_hts(n, p, random_state=seed)
    x = x + 1

    model = CodaCore(random_state=seed, objective='binary_classification',
                     cv_params=cv_params, opt_params=opt_params)
    model.fit(x, y)

    assert 0 in model.get_numerator_parts(0)
    assert 1 in model.get_denominator_parts(0)

    x, y = simulate_hts(n, p, logratio='balance', random_state=seed)
    x = x + 1

    model = CodaCore(random_state=seed, objective='binary_classification',
                     cv_params=cv_params, opt_params=opt_params)
    model.fit(x, y)

    assert 3 in model.get_numerator_parts(0)
    assert 4 in model.get_denominator_parts(0)


    x, y = simulate_hts(n, p, logratio='amalgamation', random_state=seed)
    x = x + 1

    model = CodaCore(random_state=seed, objective='binary_classification', type='SLR',
                     cv_params=cv_params, opt_params=opt_params)
    model.fit(x, y)

    assert 0 in model.get_numerator_parts(0)
    assert 2 in model.get_denominator_parts(0)
    return None

if __name__ == '__main__':
    test_codacore()
    test_codacore(cv_params={'num_folds': 4})
    test_codacore(opt_params={'epochs': 20})
    test_codacore(cv_params={'num_folds': 4}, opt_params={'epochs': 111})
