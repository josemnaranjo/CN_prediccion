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
        assert f"ADD COLUMN IF NOT EXISTS {col}" in sql
        assert tipo in sql
    assert sql.count("COMMENT ON COLUMN predictions.metrica_sku.") == 8
    assert "NOT NULL" not in sql
    assert "DEFAULT" not in sql.upper()
    assert "VIEW" not in sql.upper()
    assert "CREATE TABLE" not in sql.upper()
    assert "coeficientes de variacion" in sql
    assert "NO es Var(stock)/Var(demanda)" in sql
    assert sql.strip().upper().rstrip(";").endswith("COMMIT")
