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


def backtest_legacy(serie, cfg):
    """Copia del algoritmo anterior (ajuste único + append sin refit), como referencia."""
    n_test = cfg.semanas_evaluacion
    entrenamiento, prueba = serie.iloc[:-n_test], serie.iloc[-n_test:]
    res = b._ajustar(entrenamiento.iloc[-cfg.semanas_entrenamiento:].values.astype(float), cfg)
    B, valores, pred, bloques = cfg.bloque_acumulado, prueba.values.astype(float), [], []
    for i in range(0, n_test, B):
        tramo = valores[i:i + B]
        if len(tramo) == B:
            f = np.clip(np.asarray(res.forecast(B), dtype=float), 0, None)
            bloques.append((float(tramo.sum()), float(f.sum())))
        for real in tramo:
            pred.append(float(np.clip(res.forecast(1)[0], 0, None)))
            res = res.append([real], refit=False)
    return pd.Series(pred, index=prueba.index), bloques


def test_n_dibujo():
    c = Config()
    assert b._n_dibujo(157, c) == 79          # 157 - 78
    assert b._n_dibujo(85, c) == 12           # disponible 7 < 12 -> sube a evaluacion
    assert b._n_dibujo(42, c) == 12
    assert b._n_dibujo(157, Config(semanas_backtest=30)) == 30
    assert b._n_dibujo(157, Config(semanas_backtest=500)) == 79


def test_puntos_refit_se_cuentan_desde_el_final():
    c = Config()
    n = 157
    pts = b._puntos_refit(n, n - 79, c)
    assert pts[0] == n - 79                   # ajuste inicial
    assert (n - 12) in pts                    # reajuste al inicio de la evaluación
    assert not any(n - 12 < p < n for p in pts)   # ninguno dentro de la evaluación
    assert all((n - p) % 12 == 0 for p in pts[1:])


def test_ultimas_12_semanas_identicas_al_legacy():
    s, c = serie_sintetica(), Config()
    pred, bloques = b.backtest_arima(s, c)
    ref_pred, ref_bloques = backtest_legacy(s, c)
    assert len(pred) == 79
    pd.testing.assert_series_equal(pred.iloc[-12:], ref_pred)
    assert bloques == ref_bloques and len(bloques) == 3


def test_no_hay_filas_antes_del_primer_entrenamiento():
    s, c = serie_sintetica(), Config()
    pred, _ = b.backtest_arima(s, c)
    assert pred.index[0] >= s.index[c.semanas_entrenamiento]


def test_backtest_12_explicito_igual_al_legacy():
    s, c = serie_sintetica(), Config(semanas_backtest=12)
    pred, bloques = b.backtest_arima(s, c)
    ref_pred, ref_bloques = backtest_legacy(s, c)
    pd.testing.assert_series_equal(pred, ref_pred)
    assert bloques == ref_bloques


def test_serie_corta_dibuja_12():
    s = serie_sintetica(n=45)
    pred, bloques = b.backtest_arima(s, Config())
    assert len(pred) == 12 and len(bloques) == 3
    assert pred.notna().all()


def test_cantidad_de_ajustes(monkeypatch):
    llamadas = []
    orig = b._ajustar
    monkeypatch.setattr(b, "_ajustar", lambda *a, **k: (llamadas.append(1), orig(*a, **k))[1])
    b.backtest_arima(serie_sintetica(), Config())
    assert len(llamadas) <= 8                 # ~79/12 ≈ 7, nunca uno por semana
