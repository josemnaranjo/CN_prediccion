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


# ---------------------------------------------------------------------------
# Task 3: cierres semanales y ancla
# ---------------------------------------------------------------------------
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = "America/Santiago"


def _cap(filas):
    """filas: (kopr, 'YYYY-MM-DD HH:MM', valor)."""
    return pd.DataFrame({
        "kopr": [f[0] for f in filas],
        "stock_fisico": [float(f[2]) for f in filas],
        "capturado_en": [pd.Timestamp(f[1], tz=TZ) for f in filas],
    })


def test_cierre_es_el_lunes_de_madrugada():
    caps = _cap([("A", "2026-10-11 00:05", 90), ("A", "2026-10-12 00:05", 100)])
    c = bp.cierres_semanales(caps, Config())
    assert len(c) == 1
    assert c.iloc[0]["semana"] == date(2026, 10, 5)
    assert c.iloc[0]["stock_fin"] == 100


def test_captura_de_la_tarde_no_cierra_nada():
    caps = _cap([("A", "2026-10-12 00:05", 100), ("A", "2026-10-12 15:30", 999),
                 ("A", "2026-10-19 15:30", 777)])
    c = bp.cierres_semanales(caps, Config())
    assert list(c["semana"]) == [date(2026, 10, 5)]
    assert c.iloc[0]["stock_fin"] == 100


def test_varias_madrugadas_gana_la_mas_temprana():
    caps = _cap([("A", "2026-10-12 00:40", 110), ("A", "2026-10-12 00:05", 100)])
    c = bp.cierres_semanales(caps, Config())
    assert len(c) == 1
    assert c.iloc[0]["stock_fin"] == 100


def _cierres_dos_lunes():
    caps = _cap([("A", "2026-10-12 00:05", 100), ("A", "2026-10-19 00:05", 80)])
    return bp.cierres_semanales(caps, Config())


def test_ancla_exige_antiguedad_minima():
    c = _cierres_dos_lunes()
    z = ZoneInfo(TZ)
    a = bp.elegir_ancla_de(c, Config(), datetime(2026, 10, 22, 12, 0, tzinfo=z))
    assert a.semana == date(2026, 10, 5)
    assert a.antiguedad_dias == 10
    assert bp.elegir_ancla_de(c, Config(), datetime(2026, 10, 21, 12, 0, tzinfo=z)) is None


def test_ancla_loguea_la_eleccion(caplog):
    c = _cierres_dos_lunes()
    with caplog.at_level(logging.INFO, logger="batch_jnc"):
        bp.elegir_ancla_de(c, Config(), datetime(2026, 10, 22, 12, 0, tzinfo=ZoneInfo(TZ)))
    assert "2026-10-05" in caplog.text
    assert "10" in caplog.text


def test_ancla_none_registra_la_razon(caplog):
    vacio = bp.cierres_semanales(_cap([("A", "2026-10-12 15:30", 1)]), Config())
    with caplog.at_level(logging.INFO, logger="batch_jnc"):
        a = bp.elegir_ancla_de(vacio, Config(), datetime(2026, 10, 22, tzinfo=ZoneInfo(TZ)))
    assert a is None
    assert "no se reconstruye" in caplog.text


def test_ancla_trae_el_stock_de_todos_los_skus():
    caps = _cap([("A", "2026-10-12 00:05", 100), ("B", "2026-10-12 00:06", 50)])
    c = bp.cierres_semanales(caps, Config())
    a = bp.elegir_ancla_de(c, Config(), datetime(2026, 10, 30, tzinfo=ZoneInfo(TZ)))
    assert a.stock == {"A": 100.0, "B": 50.0}


def test_elegir_ancla_lee_postgres():
    visto = {}

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None):
            visto["sql"], visto["params"] = sql, params
        def fetchall(self):
            return [("A", Decimal("100.00"), datetime(2026, 10, 12, 0, 5, tzinfo=ZoneInfo(TZ)))]

    class Conn:
        def cursor(self): return Cur()

    cfg = Config()
    a = bp.elegir_ancla(Conn(), cfg, ahora=datetime(2026, 10, 30, tzinfo=ZoneInfo(TZ)))
    assert "stock_snapshot" in visto["sql"]
    assert "bodega" in visto["sql"]
    assert cfg.bodega in tuple(visto["params"])
    assert a.stock == {"A": 100.0}


# --- Reconstrucción hacia atrás y control de calidad -------------------------

def _anc(semana, stock):
    return bp.Ancla(semana=semana, capturado_en=datetime(2026, 10, 12, 0, 5, tzinfo=ZoneInfo(TZ)),
                    antiguedad_dias=10, stock=stock)


def _mov(filas):
    return pd.DataFrame(filas, columns=["kopr", "semana", "flujo_neto"])


def _rec(ancla, mov, inicio):
    return bp.reconstruir_stock_historico(ancla, mov, Config(), inicio)


def test_reconstruccion_camina_hacia_atras():
    mov = _mov([("A", date(2026, 10, 5), 30.0), ("A", date(2026, 9, 28), -20.0)])
    df = _rec(_anc(date(2026, 10, 5), {"A": 100.0}), mov, date(2026, 9, 21))
    df = df.sort_values("semana").reset_index(drop=True)
    assert list(df["semana"]) == [date(2026, 9, 21), date(2026, 9, 28), date(2026, 10, 5)]
    assert list(df["stock_fin"]) == [90.0, 70.0, 100.0]
    assert list(df["flujo_neto"]) == [0.0, -20.0, 30.0]


def test_reconstruccion_grilla_completa_sin_hoyos():
    mov = _mov([])
    df = _rec(_anc(date(2026, 10, 5), {"A": 100.0}), mov, date(2026, 9, 7))
    assert len(df) == 5
    assert (df["flujo_neto"] == 0).all()
    assert (df["stock_fin"] == 100).all()


def test_reconstruccion_no_recorta_negativos():
    mov = _mov([("A", date(2026, 10, 5), 50.0)])
    df = _rec(_anc(date(2026, 10, 5), {"A": 10.0}), mov, date(2026, 9, 28))
    previa = df[df["semana"] == date(2026, 9, 28)]["stock_fin"].iloc[0]
    assert previa == -40.0


def test_reconstruccion_omite_sku_sin_captura_y_avisa(caplog):
    mov = _mov([("B", date(2026, 10, 5), 5.0)])
    with caplog.at_level(logging.WARNING):
        df = _rec(_anc(date(2026, 10, 5), {"A": 100.0}), mov, date(2026, 9, 28))
    assert set(df["kopr"]) == {"A"}
    assert "B" in caplog.text


def test_reconstruccion_ignora_movimientos_posteriores_al_ancla():
    base = _mov([("A", date(2026, 10, 5), 30.0)])
    con = _mov([("A", date(2026, 10, 5), 30.0), ("A", date(2026, 10, 12), 999.0)])
    a = _rec(_anc(date(2026, 10, 5), {"A": 100.0}), base, date(2026, 9, 28))
    b = _rec(_anc(date(2026, 10, 5), {"A": 100.0}), con, date(2026, 9, 28))
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def _grilla_q(n_skus, n_sem, negativas):
    filas = []
    for s in range(n_skus):
        for w in range(n_sem):
            filas.append((f"S{s}", date(2026, 1, 5) + pd.Timedelta(weeks=w), 10.0))
    df = pd.DataFrame(filas, columns=["kopr", "semana", "stock_fin"])
    for i, v in negativas:
        df.loc[i, "stock_fin"] = v
    return df


def test_calidad_cuenta_negativas(caplog):
    df = _grilla_q(2, 10, [(3, -7.0)])
    with caplog.at_level(logging.INFO):
        r = bp.calidad_stock_historico(df)
    assert r == {"semanas": 20, "negativas": 1, "pct": 5.0}
    assert "S0" in caplog.text
    assert "-7" in caplog.text
    assert str(df.loc[3, "semana"]) in caplog.text


def test_calidad_avisa_fuera_de_rango(caplog):
    with caplog.at_level(logging.WARNING):
        bp.calidad_stock_historico(_grilla_q(1, 10, []))
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        bp.calidad_stock_historico(_grilla_q(1, 25, [(0, -1.0)]))
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


# --- Task 5: armado y persistencia de la serie ---------------------------------

def _recon_3sem():
    return pd.DataFrame({"kopr": ["A"] * 3,
                         "semana": [date(2026, 9, 21), date(2026, 9, 28), date(2026, 10, 5)],
                         "stock_fin": [70.0, 90.0, 100.0],
                         "flujo_neto": [5.0, 20.0, 10.0]})


def _cie(filas):
    return pd.DataFrame([{"kopr": k, "semana": s, "stock_fin": v,
                          "capturado_en": pd.Timestamp("2026-10-20", tz="UTC")}
                         for k, s, v in filas])


def _serie(cierres_filas, recon=None):
    return bp.armar_serie_historica(_recon_3sem() if recon is None else recon,
                                    _cie(cierres_filas), _anc(date(2026, 10, 5), {"A": 100.0}))


def test_armar_serie_marca_origen():
    s = _serie([("A", date(2026, 10, 12), 130.0), ("A", date(2026, 10, 19), 125.0)])
    assert list(s["origen"]) == ["reconstruido", "reconstruido", "capturado",
                                 "capturado", "capturado"]
    assert s.loc[s["semana"] == date(2026, 10, 5), "stock_fin"].iloc[0] == 100.0
    assert s.loc[s["semana"] == date(2026, 10, 5), "flujo_neto"].iloc[0] == 10.0


def test_armar_serie_flujo_de_capturadas_es_la_diferencia():
    s = _serie([("A", date(2026, 10, 12), 130.0), ("A", date(2026, 10, 19), 125.0)])
    assert list(s["flujo_neto"].iloc[-2:]) == [30.0, -5.0]


def test_armar_serie_cierre_sin_semana_previa_flujo_cero():
    s = _serie([("A", date(2026, 10, 26), 140.0)])
    assert s.iloc[-1]["flujo_neto"] == 0


def test_armar_serie_sin_semanas_repetidas_ni_cierres_previos():
    s = _serie([("A", date(2026, 10, 5), 999.0), ("A", date(2026, 9, 28), 999.0),
                ("A", date(2026, 10, 12), 130.0)])
    assert not s.duplicated(["kopr", "semana"]).any()
    assert len(s) == 4
    assert 999.0 not in set(s["stock_fin"])
    assert s[s["semana"] == date(2026, 9, 28)]["origen"].iloc[0] == "reconstruido"


def test_armar_serie_todas_las_semanas_son_lunes():
    s = _serie([("A", date(2026, 10, 12), 130.0)])
    assert all(x.weekday() == 0 for x in s["semana"])


class _CurFake:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _ConnFake:
    def __init__(self): self.commits = 0
    def cursor(self): return _CurFake()
    def commit(self): self.commits += 1


def test_guardar_stock_historico(monkeypatch):
    llamadas = []
    import psycopg2.extras
    monkeypatch.setattr(psycopg2.extras, "execute_values",
                        lambda cur, sql, filas, **k: llamadas.append((sql, filas, k)))
    serie = pd.DataFrame({"kopr": ["A"], "semana": [date(2026, 10, 5)],
                          "stock_fin": [100.12345], "flujo_neto": [-5.5555],
                          "origen": ["capturado"]})
    conn = _ConnFake()
    bp.guardar_stock_historico(conn, 7, "B01", serie)
    sql, filas, k = llamadas[0]
    assert "predictions.stock_historico" in sql
    assert filas == [(7, "A", "B01", date(2026, 10, 5), 100.12, -5.56, "capturado")]
    assert k["page_size"] == 1000
    assert conn.commits == 1


def test_guardar_stock_historico_vacio_no_hace_nada(monkeypatch):
    llamadas = []
    import psycopg2.extras
    monkeypatch.setattr(psycopg2.extras, "execute_values",
                        lambda *a, **k: llamadas.append(1))
    conn = _ConnFake()
    bp.guardar_stock_historico(conn, 7, "B01", pd.DataFrame(
        columns=["kopr", "semana", "stock_fin", "flujo_neto", "origen"]))
    assert llamadas == [] and conn.commits == 0
