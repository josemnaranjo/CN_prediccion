import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import batch_prediccion_jnc as b
from batch_prediccion_jnc import Config


def serie_sintetica(n=157, seed=0):
    rng = np.random.default_rng(seed)
    idx = list(pd.date_range("2023-10-02", periods=n, freq="7D").date)
    vals = 100 + 20 * np.sin(np.arange(n) / 6) + rng.normal(0, 8, n)
    return pd.Series(vals, index=idx)


def test_config_defaults():
    c = Config()
    assert (c.semanas_evaluacion, c.semanas_backtest, c.semanas_refit) == (12, None, 12)


def test_config_backtest_menor_que_evaluacion_falla():
    with pytest.raises(ValueError):
        Config(semanas_backtest=5)


def test_config_backtest_igual_a_evaluacion_ok():
    assert Config(semanas_backtest=12).semanas_backtest == 12
