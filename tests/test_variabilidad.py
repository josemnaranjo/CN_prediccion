import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAIZ = Path(__file__).resolve().parent.parent

COLUMNAS_009 = [
    ("cv_demanda", "NUMERIC(8,4)"),
    ("cv_stock", "NUMERIC(8,4)"),
    ("ratio_variabilidad", "NUMERIC(8,4)"),
    ("media_demanda", "NUMERIC(14,2)"),
    ("media_stock", "NUMERIC(14,2)"),
    ("tasa_quiebre", "NUMERIC(6,4)"),
    ("semanas_variabilidad", "SMALLINT"),
    ("prop_stock_reconstruido", "NUMERIC(6,4)"),
]


def test_migracion_009_forma():
    sql = (RAIZ / "sql" / "009_metricas_variabilidad.sql").read_text(encoding="utf-8")
    for col, tipo in COLUMNAS_009:
        assert re.search(rf"ADD COLUMN IF NOT EXISTS {col}\s+{re.escape(tipo)}\s*[,;]", sql)
    assert sql.count("COMMENT ON COLUMN predictions.metrica_sku.") == 8
    assert "NOT NULL" not in sql
    assert "DEFAULT" not in sql.upper()
    assert "VIEW" not in sql.upper()
    assert "CREATE TABLE" not in sql.upper()
    assert "coeficientes de variacion" in sql
    assert "NO es Var(stock)/Var(demanda)" in sql
    assert sql.strip().upper().rstrip(";").endswith("COMMIT")


def test_migracion_009_cada_comentario_explica_el_null():
    sql = (RAIZ / "sql" / "009_metricas_variabilidad.sql").read_text(encoding="utf-8")
    comentarios = re.findall(
        r"COMMENT ON COLUMN predictions\.metrica_sku\.(\w+) IS\s+'([^']*)'", sql)
    assert [c for c, _ in comentarios] == [c for c, _ in COLUMNAS_009]
    for col, texto in comentarios:
        assert "sin serie de stock" in texto, col
        assert "NULL = no calculable, no cero" in texto, col


# ---------------------------------------------------------------------------
# Task 1: Config
# ---------------------------------------------------------------------------
import pytest

import batch_prediccion_jnc as bp
from batch_prediccion_jnc import Config


def test_config_variabilidad():
    c = Config()
    assert c.semanas_variabilidad == 52
    assert c.min_semanas_variabilidad == 40


def test_config_variabilidad_invalida():
    with pytest.raises(ValueError):
        Config(min_semanas_variabilidad=53)
    with pytest.raises(ValueError):
        Config(min_semanas_variabilidad=1)


# ---------------------------------------------------------------------------
# Task 2: calcular_variabilidad
# ---------------------------------------------------------------------------
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

FIN = date(2026, 10, 26)


def _semanas(n):
    return [FIN - timedelta(weeks=n - 1 - i) for i in range(n)]


def _alterna(n, a, b):
    return np.tile([float(a), float(b)], n // 2 + 1)[:n]


def _series(demanda, stock, origen="reconstruido", kopr="A"):
    sem = _semanas(len(demanda))
    datos = pd.DataFrame({"kopr": kopr, "semana": sem,
                          "demanda": demanda, "demanda_cruda": demanda})
    hist = pd.DataFrame({"kopr": kopr, "semana": _semanas(len(stock)),
                         "stock_fin": stock, "origen": origen})
    return datos, hist


def _calc(datos, hist, cfg=None):
    return bp.calcular_variabilidad(datos, hist, cfg or Config())


def test_variabilidad_caso_base():
    d = _alterna(52, 10, 30)
    r = _calc(*_series(d, 2 * d))["A"]
    assert r["cv_demanda"] == 0.5049
    assert r["cv_stock"] == 0.5049
    assert r["ratio_variabilidad"] == 1.0
    assert r["media_demanda"] == 20.0
    assert r["media_stock"] == 40.0
    assert r["tasa_quiebre"] == 0.0
    assert r["semanas_variabilidad"] == 52
    assert r["prop_stock_reconstruido"] == 1.0
    assert tuple(r) == bp.COLUMNAS_VARIABILIDAD


def test_variabilidad_usa_demanda_cruda():
    d = _alterna(52, 10, 30)
    datos, hist = _series(np.full(52, 5.0), 2 * d)
    datos["demanda_cruda"] = d
    r = _calc(datos, hist)["A"]
    assert r["cv_demanda"] == 0.5049
    assert r["media_demanda"] == 20.0


def test_variabilidad_toma_las_ultimas_52():
    d = np.concatenate([np.full(8, 1e6), _alterna(52, 10, 30)])
    r = _calc(*_series(d, 2 * d))["A"]
    assert r["semanas_variabilidad"] == 52
    assert r["cv_demanda"] == 0.5049


def test_variabilidad_con_45_semanas_calcula():
    d = _alterna(45, 10, 30)
    r = _calc(*_series(d, 2 * d))["A"]
    assert r["semanas_variabilidad"] == 45
    assert r["cv_demanda"] is not None


def test_variabilidad_bajo_el_minimo_deja_todo_en_none():
    d = _alterna(39, 10, 30)
    r = _calc(*_series(d, 2 * d))["A"]
    assert tuple(r) == bp.COLUMNAS_VARIABILIDAD
    assert all(v is None for v in r.values())


def test_variabilidad_quiebre_y_reconstruido():
    d = _alterna(52, 10, 30)
    base = _calc(*_series(d, _alterna(52, 20, 60)))["A"]
    stock = _alterna(52, 20, 60)
    stock[:12] = -1.0
    stock[12] = 0.0
    origen = ["reconstruido"] * 26 + ["capturado"] * 26
    r = _calc(*_series(d, stock, origen=origen))["A"]
    assert r["tasa_quiebre"] == 0.25
    assert r["prop_stock_reconstruido"] == 0.5
    assert r["cv_stock"] != base["cv_stock"]   # los quiebres quedan dentro


def test_variabilidad_alinea_por_semana():
    d = _alterna(60, 10, 30)
    datos, hist = _series(d, 2 * d)
    ordenado = _calc(datos, hist)["A"]
    desordenado = _calc(datos, hist.sample(frac=1, random_state=0))["A"]
    assert desordenado == ordenado
    hueco = hist.drop(hist.index[[20, 30, 40]])   # 3 semanas dentro de la ventana
    assert _calc(datos, hueco)["A"]["semanas_variabilidad"] == 49


def test_variabilidad_sku_nuevo_sin_historia_comun():
    d = _alterna(60, 10, 30)
    datos, hist = _series(d, 2 * d)
    hist = hist.tail(30)                          # el stock cubre solo 30 semanas
    assert all(v is None for v in _calc(datos, hist)["A"].values())


def test_variabilidad_demanda_constante():
    r = _calc(*_series(np.full(52, 10.0), _alterna(52, 20, 60)))["A"]
    assert r["cv_demanda"] == 0.0
    assert r["ratio_variabilidad"] is None
    assert r["cv_stock"] is not None


def test_variabilidad_media_no_positiva():
    d = _alterna(52, 10, 30)
    r = _calc(*_series(d, -d))["A"]
    assert r["cv_stock"] is None and r["ratio_variabilidad"] is None
    assert r["media_stock"] == -20.0
    r = _calc(*_series(np.zeros(52), d))["A"]
    assert r["cv_demanda"] is None and r["ratio_variabilidad"] is None
    assert r["media_demanda"] == 0.0


def test_variabilidad_ratio_gigante_va_en_none():
    d = 1000.0 + _alterna(52, -0.1, 0.1)
    r = _calc(*_series(d, _alterna(52, 0, 40)))["A"]
    assert r["cv_demanda"] == 0.0001
    assert r["cv_stock"] > 1.0
    assert r["ratio_variabilidad"] is None


def test_variabilidad_sin_serie_de_stock():
    datos, _ = _series(_alterna(52, 10, 30), _alterna(52, 20, 60))
    vacio = pd.DataFrame(columns=["kopr", "semana", "stock_fin", "origen"])
    for hist in (None, vacio):
        r = _calc(datos, hist)
        assert set(r) == {"A"}
        assert tuple(r["A"]) == bp.COLUMNAS_VARIABILIDAD
        assert all(v is None for v in r["A"].values())


def test_variabilidad_tipos_de_python():
    d = _alterna(52, 10, 30)
    for v in _calc(*_series(d, 2 * d))["A"].values():
        assert type(v) in (float, int)
        assert not (isinstance(v, float) and math.isnan(v))


def test_variabilidad_varios_skus_independientes():
    d = _alterna(52, 10, 30)
    da, ha = _series(d, 2 * d, kopr="A")
    db, hb = _series(_alterna(30, 10, 30), _alterna(30, 20, 60), kopr="B")
    r = _calc(pd.concat([da, db], ignore_index=True), pd.concat([ha, hb], ignore_index=True))
    assert r["A"]["ratio_variabilidad"] == 1.0
    assert all(v is None for v in r["B"].values())
