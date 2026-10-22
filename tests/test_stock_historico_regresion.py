"""Regresión del stock reconstruido contra capturas reales (requiere ambas BD)."""
import os
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import batch_prediccion_jnc as bp
from batch_prediccion_jnc import Ancla, CONFIG

pytestmark = pytest.mark.skipif(
    not (os.environ.get("M32QUILLOTA_DATABASE_URL")
         and os.environ.get("POSTGRES_DATABASE_URL")),
    reason="requiere M32QUILLOTA_DATABASE_URL y POSTGRES_DATABASE_URL",
)

SEMANA_ANCLA = date(2026, 10, 5)       # cierre del 11 de octubre
SEMANA_COMPARADA = date(2026, 9, 28)   # cierre del 4 de octubre
INICIO = date(2026, 1, 5)
ALKA_MENTOL = "5131018"   # ALKA MENTOL 20x1x400gr.LEY
SELZ_QUESO = "5168471"    # SELZ MINI QUESO BS 1X40X35 LEY


def test_stock_reconstruido_vs_cierres_reales():
    conn_pg = bp.conectar_postgres()
    conn_ms = bp.conectar_mssql()
    try:
        cierres = bp.cierres_semanales(bp.leer_capturas(conn_pg, CONFIG), CONFIG)
        real_ancla = cierres[cierres["semana"] == SEMANA_ANCLA]
        assert not real_ancla.empty, "no hay cierre real para la semana ancla"
        ancla = Ancla(
            semana=SEMANA_ANCLA,
            capturado_en=real_ancla["capturado_en"].max().to_pydatetime(),
            antiguedad_dias=0,
            stock={str(k): float(v)
                   for k, v in zip(real_ancla["kopr"], real_ancla["stock_fin"])},
        )
        mov = bp.leer_movimientos_stock(conn_ms, CONFIG, INICIO, SEMANA_ANCLA)
    finally:
        conn_pg.close()
        conn_ms.close()

    rec = bp.reconstruir_stock_historico(ancla, mov, CONFIG, INICIO)
    rec = rec[rec["semana"] == SEMANA_COMPARADA].set_index("kopr")["stock_fin"]
    real = (cierres[cierres["semana"] == SEMANA_COMPARADA]
            .set_index("kopr")["stock_fin"].astype(float))
    comunes = [k for k in bp.SKUS_TESIS if k in rec.index and k in real.index]
    assert comunes, "ningún SKU con reconstrucción y cierre real comparables"
    err = (rec.loc[comunes] - real.loc[comunes]).abs()

    assert err[ALKA_MENTOL] <= 30
    # El spec dice SELZ QUESO +-50 "al cierre del 11 de octubre", pero ese punto
    # es exacto por construcción (es la ancla); se asierta en el cierre del 4-oct.
    assert err[SELZ_QUESO] <= 50
    assert err.median() < 15
