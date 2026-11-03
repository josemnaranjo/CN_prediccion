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


# ---------------------------------------------------------------------------
# Task 3: persistencia y línea de log
# ---------------------------------------------------------------------------
class _FakeCursor:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeConn:
    committed = False
    def cursor(self): return _FakeCursor()
    def commit(self): self.committed = True


def _metrica(kopr, **extra):
    base = {"kopr": kopr, "semanas_backtest": 12, "semanas_para_mape": 12,
            "semanas_censuradas": 0, "mape_semanal": 20.0, "mae_semanal": 5.0,
            "mape_4sem": 15.0, "bloques_4sem": 3, "en_quiebre": False}
    return base | extra


MEDIDAS = {"cv_demanda": 0.5, "cv_stock": 0.7, "ratio_variabilidad": 1.4,
           "media_demanda": 20.0, "media_stock": 40.0, "tasa_quiebre": 0.06,
           "semanas_variabilidad": 52, "prop_stock_reconstruido": 1.0}


def test_guardar_metricas_persiste_la_variabilidad(monkeypatch):
    import psycopg2.extras
    capturado = {}
    monkeypatch.setattr(psycopg2.extras, "execute_values",
                        lambda cur, sql, filas, **k: capturado.update(sql=sql, filas=filas))
    conn = _FakeConn()
    bp.guardar_metricas(conn, 7, [_metrica("A", **MEDIDAS),
                                  _metrica("B"),                                  # sin claves
                                  _metrica("C", **(MEDIDAS | {"cv_stock": float("nan")}))])
    for col in bp.COLUMNAS_VARIABILIDAD:
        assert col in capturado["sql"]
    a, b, c = capturado["filas"]
    assert a[11:] == tuple(MEDIDAS[col] for col in bp.COLUMNAS_VARIABILIDAD)
    assert b[11:] == (None,) * 8
    assert c[12] is None                      # NaN -> NULL
    assert conn.committed


def test_resumen_variabilidad_linea(caplog):
    con_dato = [_metrica(f"S{i}", **(MEDIDAS | {"ratio_variabilidad": 1.35,
                                                "tasa_quiebre": 0.059})) for i in range(20)]
    sin_ratio = _metrica("S20", **(MEDIDAS | {"ratio_variabilidad": None,
                                              "tasa_quiebre": 0.059}))
    sin_dato = _metrica("S21", **dict.fromkeys(bp.COLUMNAS_VARIABILIDAD))
    with caplog.at_level("INFO"):
        bp.resumen_variabilidad(con_dato + [sin_ratio, sin_dato], Config())
    assert ("Variabilidad (52 semanas): 21 de 22 SKUs, ratio mediano 1.35, "
            "quiebre medio 5.9%") in caplog.text
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_resumen_variabilidad_sin_datos(caplog):
    metricas = [_metrica(f"S{i}") for i in range(22)]
    with caplog.at_level("INFO"):
        bp.resumen_variabilidad(metricas, Config())
    assert ("Variabilidad (52 semanas): 0 de 22 SKUs, ratio mediano n/d, "
            "quiebre medio n/d") in caplog.text


# ---------------------------------------------------------------------------
# Task 4: integración en main
# ---------------------------------------------------------------------------
from test_stock_historico import corrida  # noqa: E402,F401  (fixture)

SERIE = pd.DataFrame({"kopr": ["A"], "semana": [date(2026, 10, 26)],
                      "stock_fin": [1.0], "origen": ["reconstruido"]})


@pytest.fixture
def espiada(corrida, monkeypatch):
    """`corrida` con las métricas guardadas y la variabilidad registradas."""
    eventos, cfg = corrida
    guardado = {}
    monkeypatch.setattr(bp, "guardar_metricas",
                        lambda conn, bid, ms: (eventos.append("metricas"),
                                               guardado.update(metricas=ms)))
    monkeypatch.setattr(bp, "generar_stock_historico",
                        lambda conn, bid, ini, c: eventos.append(("historico", bid, ini)) or SERIE)

    def variab(datos, serie, c):
        eventos.append("variabilidad")
        guardado["serie"] = serie
        return {"A": dict(MEDIDAS)}
    monkeypatch.setattr(bp, "calcular_variabilidad", variab)
    return eventos, cfg, guardado


def _todas_nulas(metricas):
    return all(m.get(col) is None for m in metricas for col in bp.COLUMNAS_VARIABILIDAD)


def test_main_calcula_la_variabilidad_entre_historico_y_metricas(espiada):
    eventos, _, guardado = espiada
    assert bp.main([]) == 0
    assert eventos.index(("historico", 99, date(2025, 1, 6))) < eventos.index("variabilidad") \
        < eventos.index("metricas") < eventos.index(("cerrar", "ok"))
    assert guardado["serie"] is SERIE


def test_main_pasa_las_ocho_medidas_a_guardar_metricas(espiada):
    _, _, guardado = espiada
    assert bp.main([]) == 0
    (m,) = guardado["metricas"]
    assert m["kopr"] == "A"
    assert {col: m[col] for col in bp.COLUMNAS_VARIABILIDAD} == MEDIDAS


def test_main_historico_none_deja_las_ocho_en_none(corrida, monkeypatch):
    eventos, _ = corrida
    guardado = {}
    monkeypatch.setattr(bp, "generar_stock_historico", lambda *a, **k: None)
    monkeypatch.setattr(bp, "guardar_metricas",
                        lambda conn, bid, ms: guardado.update(metricas=ms))
    assert bp.main([]) == 0
    assert _todas_nulas(guardado["metricas"])
    assert ("cerrar", "ok") in eventos


def test_main_excepcion_en_variabilidad_no_cambia_el_estado(espiada, monkeypatch):
    eventos, _, guardado = espiada

    llamadas = []

    def falla(*a, **k):
        llamadas.append(1)
        raise RuntimeError("boom")
    monkeypatch.setattr(bp, "calcular_variabilidad", falla)
    assert bp.main([]) == 0
    assert llamadas == [1]
    assert "metricas" in eventos and _todas_nulas(guardado["metricas"])
    assert ("cerrar", "ok") in eventos and ("cerrar", "error") not in eventos


def test_main_excepcion_en_historico_sigue_guardando_metricas(espiada, monkeypatch):
    eventos, _, guardado = espiada

    def falla(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(bp, "generar_stock_historico", falla)
    assert bp.main([]) == 0
    assert eventos.index("rollback") < eventos.index("metricas") \
        < eventos.index(("cerrar", "ok"))
    assert guardado["serie"] is None


@pytest.mark.parametrize("argv, reconstruir", [([], False), (["--desde-csv", "x.csv"], True)])
def test_main_sin_historico_guarda_nulls(corrida, monkeypatch, argv, reconstruir):
    eventos, cfg = corrida
    cfg.reconstruir_stock = reconstruir
    guardado = {}
    monkeypatch.setattr(bp, "guardar_metricas",
                        lambda conn, bid, ms: guardado.update(metricas=ms))
    assert bp.main(argv) == 0
    assert _todas_nulas(guardado["metricas"])
    assert ("cerrar", "ok") in eventos


def test_variabilidad_cv_que_no_cabe_va_en_none():
    d = _alterna(52, 10, 30)
    r = _calc(*_series(d, _alterna(52, -100.0, 100.02)))["A"]   # media 0.01, CV ~ 10098
    assert r["media_stock"] == 0.01
    assert r["cv_stock"] is None
    assert r["ratio_variabilidad"] is None
    assert r["cv_demanda"] == 0.5049
