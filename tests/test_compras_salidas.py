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
