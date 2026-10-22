import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    assert "INTEGER" not in sql.upper()
