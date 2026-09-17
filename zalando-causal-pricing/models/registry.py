"""Every model in the study, by name: how to build it, how to label it, and whether it trains by
epochs. Runners, the report, the pipeline and the tests all read this and nothing else."""
from dataclasses import dataclass, field
from typing import Callable

from dml import DML, TFForecaster, TorchEffect, TorchNuisance
from lgbm import LGBMEffect, LGBMNuisance, LGBMSLearner
from mdl import MDLForecaster


@dataclass
class Config:
    """Training budget and problem shape shared by every model in a run."""
    head: str = "mult"          # 'mult' (constant elasticity, real data) or 'add' (linear simulator)
    loss: str = "l1"            # nuisance / S-learner loss
    epochs: int = 10            # transformer epochs (nuisance and S-learner nets)
    effect_epochs: int = 10     # effect-network epochs
    seed: int = 0
    lr: float = 1e-3
    max_trees: int = 2000       # LightGBM early-stopping cap; check the logged tree counts against it
    net: dict = field(default_factory=dict)


@dataclass
class Spec:
    display: str
    neural: bool                # trains by epochs (so results are labelled with the epoch count)
    make: Callable[[Config, list], object]


def _torch_dml(cross_fit=True, treatment_model=True):
    return lambda c, n_cat: DML(
        c.head,
        nuisance=lambda fold: TorchNuisance(n_cat, c.head, c.loss, c.epochs, c.lr, c.seed, c.net, treatment_model, fold),
        effect=lambda: TorchEffect(n_cat, c.head, c.effect_epochs, c.lr, c.seed, c.net),
        cross_fit=cross_fit)


def _lgbm_dml(autoregressive):
    return lambda c, n_cat: DML(
        c.head,
        nuisance=lambda fold: LGBMNuisance(c.head, c.seed, c.max_trees, autoregressive, fold),
        effect=lambda: LGBMEffect(c.head, c.seed, c.max_trees))


MODELS = {
    "dml": Spec("DML Forecaster (paper 1)", True, _torch_dml()),
    "dml-nocf": Spec("DML, no cross-fitting", True, _torch_dml(cross_fit=False)),
    "sdml": Spec("sDML (no treatment model)", True, _torch_dml(treatment_model=False)),
    "tf": Spec("TF, linear head S-learner (paper 1 ablation)", True,
               lambda c, n: TFForecaster(n, c.head, c.loss, c.epochs, c.lr, c.seed, c.net)),
    "mdl": Spec("Monotonic-demand transformer (paper 2)", True,
                lambda c, n: MDLForecaster(n, c.head, c.epochs, c.lr, c.seed, c.net)),
    "mdl-anchored": Spec("paper 2 model anchored to recent demand level (ablation)", True,
                         lambda c, n: MDLForecaster(n, c.head, c.epochs, c.lr, c.seed, c.net, anchor=True)),
    "lgbm": Spec("direct multi-horizon LightGBM (S-learner, paper 2 baseline)", False,
                 lambda c, n: LGBMSLearner(c.head, c.seed, c.max_trees)),
    "dml-lgbm": Spec("DML layout, direct multi-horizon LightGBM nuisances", False, _lgbm_dml(False)),
    "dml-lgbm-ar": Spec("DML layout, autoregressive LightGBM nuisances", False, _lgbm_dml(True)),
}
NAIVE = {"last4": "naive: mean of last 4 weeks"}  # computed by the M5 runner, not a model


def build(name, cfg, n_cat):
    return MODELS[name].make(cfg, n_cat)


def label(name, epochs):
    """Row label in tables: neural models carry their epoch count, the rest do not."""
    return f"{name} ({int(epochs)} ep)" if name in MODELS and MODELS[name].neural else name


def display(name):
    return MODELS[name].display if name in MODELS else NAIVE.get(name, name)
