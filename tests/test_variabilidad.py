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
