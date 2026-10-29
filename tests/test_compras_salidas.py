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
