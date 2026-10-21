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
