import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import batch_prediccion_jnc as b
from batch_prediccion_jnc import Config

HASTA = date(2026, 10, 12)  # un lunes: última semana cerrada de los tests


def semanas(n, hasta=HASTA):
    return [hasta - timedelta(weeks=n - 1 - i) for i in range(n)]


def df_ventas(kopr, valores, hasta=HASTA):
    return pd.DataFrame({"kopr": kopr, "semana": semanas(len(valores), hasta),
                         "demanda": [float(v) for v in valores]})


def df_bonif(kopr, valores, hasta=HASTA):
    return pd.DataFrame({"kopr": kopr, "semana": semanas(len(valores), hasta),
                         "bonificado": [float(v) for v in valores]})


CFG = Config(skus=("A", "B", "5131012"))


def test_config_defaults_bonificacion():
    c = Config()
    assert c.semanas_factor_bonif == 13
    assert c.tido_bonificacion == ("GDI",)
    assert c.skus_sin_factor == ("5131012",)
    assert c.factor_bonif_maximo == 0.60
    assert c.semanas_factor_largo == 52
    assert c.umbral_deriva_factor == 0.10


def test_config_tido_bonificacion_no_puede_solapar_venta():
    with pytest.raises(ValueError):
        Config(tido_bonificacion=("FCV",))


def test_ultima_semana_cerrada():
    assert b.ultima_semana_cerrada(date(2026, 10, 21)) == date(2026, 10, 12)  # miércoles
    assert b.ultima_semana_cerrada(date(2026, 10, 19)) == date(2026, 10, 12)  # lunes


def test_factor_es_agregado_no_promedio_de_factores():
    # 6 semanas de 100 y 7 de 200 (total 2000); bonificado 10 por semana (total 130).
    ventas = df_ventas("A", [100] * 6 + [200] * 7)
    bonif = df_bonif("A", [10] * 13)
    r = b.calcular_factores_bonificacion(ventas, bonif, HASTA, CFG)
    assert r["A"]["factor"] == pytest.approx(130 / 2000)   # 0.065, no 0.0731


def test_solo_cuenta_la_ventana_de_13_semanas():
    ventas = df_ventas("A", [100] * 20)
    bonif = df_bonif("A", [1000] * 7 + [10] * 13)   # lo anterior a la ventana no cuenta
    r = b.calcular_factores_bonificacion(ventas, bonif, HASTA, CFG)
    assert r["A"]["factor"] == pytest.approx(130 / 1300)


def test_semanas_posteriores_a_la_ultima_cerrada_no_cuentan():
    ventas = df_ventas("A", [100] * 14)
    bonif = df_bonif("A", [10] * 13 + [9999])  # la 14.ª queda después de HASTA - 1 semana
    r = b.calcular_factores_bonificacion(ventas, bonif, HASTA - timedelta(weeks=1), CFG)
    assert r["A"]["factor"] == pytest.approx(0.10)


def test_factor_largo_usa_52_semanas():
    ventas = df_ventas("A", [100] * 52)
    bonif = df_bonif("A", [0] * 39 + [20] * 13)
    r = b.calcular_factores_bonificacion(ventas, bonif, HASTA, CFG)
    assert r["A"]["factor"] == pytest.approx(0.20)
    assert r["A"]["factor_largo"] == pytest.approx(260 / 5200)


def test_venta_cero_da_none():
    r = b.calcular_factores_bonificacion(
        df_ventas("A", [0] * 13), df_bonif("A", [10] * 13), HASTA, CFG)
    assert r["A"]["factor"] is None


def test_venta_neta_negativa_da_none():
    r = b.calcular_factores_bonificacion(
        df_ventas("A", [-50] * 13), df_bonif("A", [10] * 13), HASTA, CFG)
    assert r["A"]["factor"] is None


def test_factor_no_almacenable_da_none_y_advierte(caplog):
    caplog.set_level(logging.WARNING)
    r = b.calcular_factores_bonificacion(
        df_ventas("A", [0.01] * 13), df_bonif("A", [100] * 13), HASTA, CFG)
    assert r["A"]["factor"] is None
    assert "A" in caplog.text


def test_sku_excluido_da_none_aunque_haya_datos():
    r = b.calcular_factores_bonificacion(
        df_ventas("5131012", [100] * 13), df_bonif("5131012", [30] * 13), HASTA, CFG)
    assert r["5131012"] == {"factor": None, "factor_largo": None}


def test_sin_bonificado_en_la_fuente_devuelve_vacio():
    assert b.calcular_factores_bonificacion(df_ventas("A", [100] * 13), None, HASTA, CFG) == {}


def test_sin_filas_gdi_el_factor_es_cero():
    vacio = pd.DataFrame(columns=["kopr", "semana", "bonificado"])
    r = b.calcular_factores_bonificacion(df_ventas("A", [100] * 13), vacio, HASTA, CFG)
    assert r["A"]["factor"] == 0.0


def test_advertir_sobre_maximo_se_usa_igual(caplog):
    caplog.set_level(logging.INFO)
    f = {"A": {"factor": 0.65, "factor_largo": 0.64}, "B": {"factor": 0.20, "factor_largo": 0.21}}
    b.advertir_factores(f, CFG)
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1 and "A" in avisos[0].getMessage()
    assert f["A"]["factor"] == 0.65   # no se recorta


def test_advertir_deriva_registra_los_dos_valores(caplog):
    caplog.set_level(logging.INFO)
    b.advertir_factores({"A": {"factor": 0.475, "factor_largo": 0.176}}, CFG)
    msg = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "47.5" in msg and "17.6" in msg


def test_advertir_sin_novedad_no_emite_warning(caplog):
    caplog.set_level(logging.INFO)
    b.advertir_factores({"A": {"factor": 0.20, "factor_largo": 0.22}, "B": {"factor": None, "factor_largo": None}}, CFG)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_query_incluye_gdi_sin_cambiar_el_filtro_de_venta():
    q = b._query_ventas(Config())
    assert "'GDI'" in q
    assert "IN ('FCV','FDV','NCV','GDI')" in q          # WHERE de ambos tipos
    assert "DDO.CAPRCO1" in q
    assert "KOBO" not in q and "BOSULIDO" not in q       # mismo filtro que la venta: sin bodega
    assert "GROUP BY RTRIM(DDO.KOPRCT), CONVERT(DATE, EDO.FEEMDO)" in q


def test_separar_descarta_de_ventas_los_dias_sin_venta():
    df = pd.DataFrame({
        "kopr": ["A", "A", "B"],
        "fecha": pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-06"]),
        "unidades": [10.0, 0.0, 5.0],
        "bonificado": [2.0, 3.0, 0.0],
        "lineas_venta": [1, 0, 2],          # A/06-oct es un día solo con guías GDI
    })
    ventas, bonif = b.separar_ventas_y_bonificado(df)
    assert list(ventas["kopr"]) == ["A", "B"] and list(ventas["unidades"]) == [10.0, 5.0]
    assert list(ventas.columns) == ["kopr", "fecha", "unidades"]
    assert list(bonif["kopr"]) == ["A", "A"] and list(bonif["bonificado"]) == [2.0, 3.0]
    assert list(bonif.columns) == ["kopr", "fecha", "bonificado"]


def test_semanalizar_bonificado_agrupa_por_lunes():
    bonif = pd.DataFrame({"kopr": ["A", "A", "A"],
                          "fecha": pd.to_datetime(["2026-10-05", "2026-10-11", "2026-10-12"]),
                          "bonificado": [1.0, 2.0, 4.0]})
    r = b.semanalizar_bonificado(bonif).set_index("semana")["bonificado"]
    assert r[date(2026, 10, 5)] == 3.0 and r[date(2026, 10, 12)] == 4.0


def test_semanalizar_bonificado_vacio():
    r = b.semanalizar_bonificado(pd.DataFrame(columns=["kopr", "fecha", "bonificado"]))
    assert list(r.columns) == ["kopr", "semana", "bonificado"] and r.empty


class _FakeCursor:
    def __init__(self, store): self.store = store
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, *a, **k): self.store["executed"] = a


class _FakeConn:
    def __init__(self): self.store = {}
    def cursor(self): return _FakeCursor(self.store)
    def commit(self): self.store["commit"] = True


def _metrica(kopr, **extra):
    base = {"kopr": kopr, "semanas_backtest": 12, "semanas_para_mape": 12,
            "semanas_censuradas": 0, "mape_semanal": 20.0, "mae_semanal": 5.0,
            "mape_4sem": 15.0, "bloques_4sem": 3, "en_quiebre": False}
    return base | extra


def test_guardar_metricas_persiste_el_factor(monkeypatch):
    capturado = {}
    import psycopg2.extras
    monkeypatch.setattr(psycopg2.extras, "execute_values",
                        lambda cur, sql, filas, **k: capturado.update(sql=sql, filas=filas))
    conn = _FakeConn()
    b.guardar_metricas(conn, 7, [_metrica("A", factor_bonificacion=0.26123456),
                                 _metrica("B", factor_bonificacion=None),
                                 _metrica("C")])          # sin la clave: NULL
    assert "factor_bonificacion" in capturado["sql"]
    assert [f[-1] for f in capturado["filas"]] == [0.2612, None, None]
    assert conn.store["commit"] is True


def _resultado(kopr, demanda):
    return {"kopr": kopr, "proyeccion": pd.DataFrame({"unidades": [demanda] * 4})}


def test_resumen_cobertura_con_factor(caplog):
    caplog.set_level(logging.INFO)
    stock = pd.DataFrame({"kopr": ["5131018"], "bodega": ["B01"], "stock_fisico": [3355.0]})
    b.resumen_cobertura([_resultado("5131018", 564.0)], stock, Config(),
                        factores={"5131018": {"factor": 0.261, "factor_largo": 0.26}})
    assert "mediana 4.7 semanas" in caplog.text


def test_resumen_cobertura_sin_factor_igual_a_hoy(caplog):
    caplog.set_level(logging.INFO)
    stock = pd.DataFrame({"kopr": ["5131018"], "bodega": ["B01"], "stock_fisico": [3355.0]})
    b.resumen_cobertura([_resultado("5131018", 564.0)], stock, Config())
    assert "mediana 5.9 semanas" in caplog.text   # 3355 / 564 = 5.95 -> %.1f da 5.9
