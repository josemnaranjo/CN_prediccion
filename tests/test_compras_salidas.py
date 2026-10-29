import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAIZ = Path(__file__).resolve().parent.parent


def test_migracion_008_forma():
    sql = (RAIZ / "sql" / "008_compras_salidas.sql").read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS compras NUMERIC(14,2)" in sql
    assert "ADD COLUMN IF NOT EXISTS salidas NUMERIC(14,2)" in sql
    assert "NOT NULL" not in sql
    assert "Excluye GRI y GAR." in sql
    assert "(FCV + FDV - NCV + GDI)" in sql
    assert "VIEW" not in sql.upper()
    assert "CREATE TABLE" not in sql.upper()


# ---------------------------------------------------------------------------
# Task 2: Config, consulta y lector
# ---------------------------------------------------------------------------
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

import batch_prediccion_jnc as bp
from batch_prediccion_jnc import Config


def test_config_tido_compra():
    c = Config()
    assert c.tido_compra == ("FCC",)
    assert c.tido_compra_neg == ("NCC",)
    assert Config(tido_compra=("GRI",)).tido_compra == ("GRI",)
    with pytest.raises(ValueError):
        Config(tido_compra=("FCV",))
    with pytest.raises(ValueError):
        Config(tido_compra_neg=("NCV",))


def _suma(sql, alias):
    return sql.split(f"AS {alias}")[0].rsplit("SUM(", 1)[1]


def test_query_movimientos_separa_compras_y_salidas():
    sql = bp._query_movimientos(Config(), date(2026, 1, 5), date(2026, 10, 12))
    compras = _suma(sql, "compras")
    assert "'FCC'" in compras and "'NCC'" in compras
    assert "'GRI'" not in compras and "'GAR'" not in compras
    salidas = _suma(sql, "salidas")
    for t in ("FCV", "FDV", "GDI", "NCV"):
        assert f"'{t}'" in salidas
    assert "CAPRCO2" not in sql


def test_leer_movimientos_trae_compras_y_salidas(monkeypatch):
    monkeypatch.setattr(bp.pd, "read_sql", lambda q, c: pd.DataFrame({
        "kopr": ["ABC  ", "DEF"],
        "semana": [pd.Timestamp("2026-10-05")] * 2,
        "flujo_neto": [Decimal("12.5"), Decimal("4")],
        "compras": [Decimal("1442"), None],
        "salidas": [Decimal("711"), None],
    }))
    df = bp.leer_movimientos_stock(None, Config(), date(2026, 1, 5), date(2026, 10, 5))
    assert df["compras"].dtype == float and df["salidas"].dtype == float
    assert list(df["compras"]) == [1442.0, 0.0]
    assert list(df["salidas"]) == [711.0, 0.0]


# ---------------------------------------------------------------------------
# Task 3: unir, persistir y orquestar
# ---------------------------------------------------------------------------
S1, S2, S3 = date(2026, 10, 5), date(2026, 10, 12), date(2026, 10, 19)


def _serie_base():
    return pd.DataFrame({
        "kopr": ["A"] * 3 + ["B"] * 3,
        "semana": [S1, S2, S3] * 2,
        "stock_fin": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
        "flujo_neto": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "origen": ["reconstruido", "capturado", "capturado"] * 2,
    })


def _movs(filas):
    return pd.DataFrame(filas, columns=["kopr", "semana", "flujo_neto", "compras", "salidas"])


def _fila(df, kopr, semana):
    return df[(df["kopr"] == kopr) & (df["semana"] == semana)].iloc[0]


def test_agregar_une_por_sku_y_semana():
    r = bp.agregar_compras_salidas(
        _serie_base(), _movs([("A", S2, 0.0, 1442.0, 711.0)]), S3)
    f = _fila(r, "A", S2)
    assert f["compras"] == 1442.0 and f["salidas"] == 711.0


def test_agregar_semana_sin_movimiento_es_cero():
    r = bp.agregar_compras_salidas(
        _serie_base(), _movs([("A", S2, 0.0, 1442.0, 711.0)]), S3)
    f = _fila(r, "A", S1)
    assert f["compras"] == 0.0 and f["salidas"] == 0.0


def test_agregar_compras_negativas_se_llevan_a_cero():
    r = bp.agregar_compras_salidas(
        _serie_base(), _movs([("A", S2, 0.0, -30.0, 5.0)]), S3)
    f = _fila(r, "A", S2)
    assert f["compras"] == 0.0 and f["salidas"] == 5.0


def test_agregar_posterior_a_hasta_es_nulo():
    r = bp.agregar_compras_salidas(_serie_base(), _movs([]), S2)
    f = _fila(r, "A", S3)
    assert pd.isna(f["compras"]) and pd.isna(f["salidas"])
    assert _fila(r, "A", S2)["compras"] == 0.0


def test_agregar_ignora_sku_ajeno():
    base = _serie_base()
    r = bp.agregar_compras_salidas(base, _movs([("Z", S2, 0.0, 9.0, 9.0)]), S3)
    assert len(r) == len(base)
    assert set(r["kopr"]) == {"A", "B"}


def test_agregar_no_altera_flujo_ni_stock():
    base = _serie_base()
    r = bp.agregar_compras_salidas(base, _movs([("A", S2, 0.0, 7.0, 7.0)]), S3)
    for col in ("kopr", "semana", "stock_fin", "flujo_neto", "origen"):
        assert list(r[col]) == list(base[col])


class _CurFake:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _ConnFake:
    def __init__(self): self.commits = 0
    def cursor(self): return _CurFake()
    def commit(self): self.commits += 1


def _guardar(monkeypatch, serie):
    llamadas = []
    import psycopg2.extras
    monkeypatch.setattr(psycopg2.extras, "execute_values",
                        lambda cur, sql, filas, **k: llamadas.append((sql, filas)))
    bp.guardar_stock_historico(_ConnFake(), 7, "B01", serie)
    return llamadas[0]


def _una_fila(**extra):
    return pd.DataFrame({"kopr": ["A"], "semana": [S1], "stock_fin": [100.12345],
                         "flujo_neto": [-5.5555], "origen": ["capturado"], **extra})


def test_guardar_incluye_compras_y_salidas(monkeypatch):
    sql, filas = _guardar(monkeypatch, _una_fila(compras=[1442.0], salidas=[711.004]))
    assert "compras" in sql and "salidas" in sql
    assert filas == [(7, "A", "B01", S1, 100.12, -5.56, "capturado", 1442.0, 711.0)]


def test_guardar_nan_y_columnas_ausentes_son_null(monkeypatch):
    _, filas = _guardar(monkeypatch, _una_fila(compras=[float("nan")], salidas=[float("nan")]))
    assert filas[0][-2:] == (None, None)
    _, filas = _guardar(monkeypatch, _una_fila())
    assert filas[0][-2:] == (None, None)


class _Cx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Pg:
    def rollback(self): pass


def _preparar_generar(monkeypatch, ancla_semana, cerrada, visto):
    ancla = bp.Ancla(semana=ancla_semana, capturado_en=None, antiguedad_dias=1,
                     stock={"A": 5.0})
    serie = pd.DataFrame({"kopr": ["A"], "semana": [ancla_semana], "stock_fin": [5.0],
                          "flujo_neto": [1.0], "origen": ["capturado"]})
    monkeypatch.setattr(bp, "elegir_ancla", lambda conn, c: ancla)
    monkeypatch.setattr(bp, "conectar_mssql", lambda: _Cx())
    monkeypatch.setattr(bp, "ultima_semana_cerrada", lambda *a, **k: cerrada)

    def mov(conn, c, desde, semana_ancla):
        visto["hasta"] = semana_ancla
        return "MOV"
    monkeypatch.setattr(bp, "leer_movimientos_stock", mov)
    monkeypatch.setattr(bp, "reconstruir_stock_historico", lambda *a: "REC")
    monkeypatch.setattr(bp, "calidad_stock_historico", lambda df: None)
    monkeypatch.setattr(bp, "leer_capturas", lambda conn, c: "CAP")
    monkeypatch.setattr(bp, "cierres_semanales", lambda cap, c: "CIE")
    monkeypatch.setattr(bp, "armar_serie_historica", lambda *a: serie)
    monkeypatch.setattr(bp, "agregar_compras_salidas",
                        lambda s, m, h: visto.update(agr=(s, m, h)) or "CON_COMPRAS")
    monkeypatch.setattr(bp, "guardar_stock_historico",
                        lambda conn, bid, bod, s: visto.update(guardado=s))
    return serie


def test_generar_lee_hasta_la_ultima_semana_cerrada(monkeypatch):
    visto = {}
    _preparar_generar(monkeypatch, S2, date(2026, 10, 26), visto)
    bp.generar_stock_historico(_Pg(), 42, date(2025, 1, 6), Config(skus=("A",)))
    assert visto["hasta"] == date(2026, 10, 26)
    visto = {}
    _preparar_generar(monkeypatch, S3, S2, visto)
    bp.generar_stock_historico(_Pg(), 42, date(2025, 1, 6), Config(skus=("A",)))
    assert visto["hasta"] == S3


def test_generar_agrega_compras_antes_de_guardar(monkeypatch):
    visto = {}
    serie = _preparar_generar(monkeypatch, S2, date(2026, 10, 26), visto)
    bp.generar_stock_historico(_Pg(), 42, date(2025, 1, 6), Config(skus=("A",)))
    assert visto["agr"] == (serie, "MOV", date(2026, 10, 26))
    assert visto["guardado"] == "CON_COMPRAS"
