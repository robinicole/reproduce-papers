"""Every registered model fits the toy problem and behaves as its design implies.

DML variants recover the effect under confounding; S-learners fit on-policy but attenuate the
effect; the monotone head is monotone; the DML layout beats the same trees used naively; the
fixed-effects estimator recovers a known elasticity. About three minutes on CPU.
"""
import numpy as np
import pytest

from registry import MODELS, Config, build

FAST = Config(head="add", loss="l2", epochs=20, effect_epochs=20, max_trees=150)


def fit_predict(name, toy, cfg=FAST):
    P, Wtr, Wt, eff = toy
    pred, psi = build(name, cfg, P.n_cat).fit(Wtr).predict(Wt)
    return np.abs(pred - Wt.y).mean(), np.abs(psi - eff).mean(), eff.mean(), pred, psi, Wt


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_fits(name, toy_add):
    """Finite, and on-policy accurate. Effect quality is model-specific and tested below: an
    S-learner attenuates it and sDML loses it entirely, both by design."""
    mae, _, _, pred, psi, _ = fit_predict(name, toy_add)
    assert np.isfinite(pred).all() and np.isfinite(psi).all()
    assert mae < 8, f"{name}: on-policy MAE {mae:.2f}"


def test_dml_recovers_the_effect_where_the_naive_learner_cannot(toy_add):
    _, dml_err, eff_mean, *_ = fit_predict("dml", toy_add)
    _, tf_err, *_ = fit_predict("tf", toy_add)
    assert dml_err < 0.5 * eff_mean
    assert dml_err < tf_err


def test_removing_the_treatment_model_breaks_the_effect(toy_add):
    _, dml_err, *_ = fit_predict("dml", toy_add)
    _, sdml_err, *_ = fit_predict("sdml", toy_add)
    assert sdml_err > dml_err


def test_dml_layout_beats_the_same_trees_used_naively(toy_add):
    _, s_err, *_ = fit_predict("lgbm", toy_add)
    _, direct_err, *_ = fit_predict("dml-lgbm", toy_add)
    _, ar_err, *_ = fit_predict("dml-lgbm-ar", toy_add)
    assert direct_err < s_err and ar_err < s_err


def test_monotone_head_is_monotone_in_discount(toy_add):
    P, Wtr, Wt, _ = toy_add
    m = build("mdl", FAST, P.n_cat).fit(Wtr)
    assert (m._demand(Wt.with_discount(0.5)) >= m._demand(Wt.with_discount(0.0)) - 1e-4).all()


def test_multiplicative_head_runs_end_to_end(toy_mult):
    cfg = Config(head="mult", loss="l1", epochs=6, effect_epochs=6, max_trees=100)
    for name in ["dml", "mdl-anchored", "lgbm", "dml-lgbm"]:
        _, _, _, pred, psi, _ = fit_predict(name, toy_mult, cfg)
        assert np.isfinite(pred).all() and (pred >= 0).all(), name
        assert psi.mean() < 0, name                       # demand falls with price on average
        if name != "lgbm":                                # constrained by construction; the S-learner is not
            assert (psi <= 0).all(), name


def test_runs_are_reproducible(toy_add):
    """Same seed, same numbers: network construction is seeded, not only training."""
    a = fit_predict("dml", toy_add, Config(head="add", loss="l2", epochs=2, effect_epochs=2))[3]
    b = fit_predict("dml", toy_add, Config(head="add", loss="l2", epochs=2, effect_epochs=2))[3]
    assert np.array_equal(a, b)


THREE = Config(head="mult", loss="l1", epochs=12, effect_epochs=12, max_trees=150)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_handles_three_treatments(name, toy_three):
    """Shapes and finiteness with a vector treatment: psi is (N, 3), forecasts are non-negative."""
    P, Wtr, Wt, _ = toy_three
    cfg = Config(**{**THREE.__dict__, "treatments": P.treatments, "epochs": 3, "effect_epochs": 3})
    pred, psi = build(name, cfg, P.n_cat).fit(Wtr).predict(Wt)
    assert pred.shape == Wt.y.shape and psi.shape == (len(Wt), 3)
    assert np.isfinite(pred).all() and np.isfinite(psi).all() and (pred >= 0).all()


def test_dml_separates_three_confounded_effects(toy_three):
    """With all three treatments residualized, DML gets every sign right and beats the naive
    transformer on each effect; the S-learner cannot tell the season from the treatments."""
    P, Wtr, Wt, true = toy_three
    cfg = Config(**{**THREE.__dict__, "treatments": P.treatments})
    _, psi_dml = build("dml", cfg, P.n_cat).fit(Wtr).predict(Wt)
    _, psi_tf = build("tf", cfg, P.n_cat).fit(Wtr).predict(Wt)
    signs = np.array([-1, -1, +1])
    assert (np.sign(psi_dml.mean(0)) == signs).all()
    err_dml, err_tf = np.abs(psi_dml - true).mean(0), np.abs(psi_tf - true).mean(0)
    assert (err_dml < np.abs(true).mean(0)).all(), err_dml          # not degenerate on any treatment
    assert err_dml.sum() < err_tf.sum(), (err_dml, err_tf)


def test_twfe_recovers_a_known_elasticity():
    import twfe
    rng = np.random.default_rng(0)
    n, T, true_eps = 800, 60, -2.0
    item = rng.normal(2.5, 0.5, n)[:, None]
    week = np.sin(2 * np.pi * np.arange(T) / 26)[None, :] * 0.4
    logp = 1.0 + 0.3 * rng.normal(0, 1, (n, 1)) - 0.5 * week + rng.normal(0, 0.15, (n, T))
    q = rng.poisson(np.exp(item + week + true_eps * logp)).astype(np.float64)
    eps, _, _ = twfe.fit(q, np.exp(logp), np.ones_like(q, bool))
    naive = np.polyfit(logp.ravel(), np.log1p(q.ravel()), 1)[0]
    assert abs(eps - true_eps) < 0.1 and abs(naive - true_eps) > abs(eps - true_eps)
