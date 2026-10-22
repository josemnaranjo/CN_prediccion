import sys
from pathlib import Path

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import batch_prediccion_jnc as bp
from batch_prediccion_jnc import Config

RAIZ = Path(__file__).resolve().parent.parent


def test_config_defaults_stock_historico():
    c = Config()
    assert c.reconstruir_stock is True
    assert c.dias_minimos_ancla == 10
    assert c.tido_entrada == ("FCC", "GRI", "GAR")
    assert c.tido_salida == ("FCV", "FDV", "GDI")
    assert c.tido_entrada_neg == ("NCC",)
    assert c.tido_salida_neg == ("NCV",)
    assert c.zona_horaria == "America/Santiago"
    assert c.hora_madrugada_max == 6


def test_config_tido_stock_no_se_solapan():
    with pytest.raises(ValueError):
        Config(tido_entrada=("FCC", "FCV"))
    with pytest.raises(ValueError):
        Config(tido_entrada_neg=("NCV",))


def test_migracion_007_forma():
    sql = (RAIZ / "sql" / "007_stock_historico.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS predictions.stock_historico" in sql
    assert "PRIMARY KEY (batch_run_id, kopr, bodega, semana)" in sql
    assert "EXTRACT(ISODOW FROM semana) = 1" in sql
    assert "origen IN ('reconstruido', 'capturado')" in sql
    assert "REFERENCES predictions.batch_run(id)" in sql
    assert "stock_historico_semana_lunes" in sql
    assert "stock_historico_origen" in sql
    assert "INTEGER" not in sql.upper()


def test_signos_tido():
    assert bp.signos_tido(Config()) == {
        "FCC": 1, "GRI": 1, "GAR": 1, "NCV": 1,
        "FCV": -1, "FDV": -1, "GDI": -1, "NCC": -1,
    }


def test_query_movimientos_elementos_obligatorios():
    sql = bp._query_movimientos(Config(), date(2026, 1, 5), date(2026, 10, 12))
    assert "BOSULIDO = 'B01'" in sql
    assert "CAPRCO1" in sql
    assert "CAPRCO2" not in sql
    assert "'19000101'" in sql
    assert "% 7" in sql
    for t in ("FCC", "GRI", "GAR", "NCV", "FCV", "FDV", "GDI", "NCC"):
        assert f"'{t}'" in sql
    assert "FEEMDO >= '2026-01-05'" in sql
    assert "FEEMDO < '2026-10-12'" in sql


def test_leer_movimientos_stock_aplica_signo_y_tipos(monkeypatch):
    capturado = {}

    def falso_read_sql(query, conn):
        capturado["query"] = query
        return pd.DataFrame({
            "kopr": ["ABC  ", " DEF"],
            "semana": [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-09-28")],
            "flujo_neto": [Decimal("12.5"), Decimal("-3")],
        })

    monkeypatch.setattr(bp.pd, "read_sql", falso_read_sql)
    df = bp.leer_movimientos_stock(None, Config(), date(2026, 1, 5), date(2026, 10, 5))
    assert list(df["kopr"]) == ["ABC", "DEF"]
    assert df["semana"].iloc[0] == date(2026, 10, 5)
    assert type(df["semana"].iloc[0]) is date
    assert df["flujo_neto"].dtype == float
    assert "FEEMDO < '2026-10-12'" in capturado["query"]
