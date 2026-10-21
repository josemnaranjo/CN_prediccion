# Cobertura con producto bonificado (job + base de datos) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el job calcule y guarde por SKU el factor de bonificación (GDI / venta neta, últimas 13 semanas) y que `v_cobertura_actual` use la salida amplificada, sin tocar el modelo ni sus métricas.

**Architecture:** El factor es un dato paralelo a la serie de demanda. La consulta de Softland que ya existe trae un segundo agregado (GDI) en la misma pasada; en Python se separa en dos tablas (venta, idéntica a la de hoy, y bonificado). Una función pura calcula el factor agregado por ventana, se guarda en `metrica_sku.factor_bonificacion` y la migración 006 recrea la vista de cobertura con `COALESCE(factor, 0)`.

**Tech Stack:** Python 3.12, pandas, psycopg2, pytest, PostgreSQL, SQL Server (Softland, solo lectura).

**Spec:** `/Users/josemarianaranjo/Desktop/dashboard_CN/SPEC_cobertura_bonificado.md` (secciones 1 y 2 y criterios de Job/Base de datos; backend y frontend van en otras sesiones).

## Rulings previos a la ejecución (el spec es la autoridad; estos son huecos que el spec no resuelve)

1. **Migración `006`, no `005`.** El spec la llama `sql/005_factor_bonificacion.sql`, pero `005_evaluacion_en_vista.sql` ya existe y está aplicada en dev. Se crea `sql/006_factor_bonificacion.sql`. Costo si está mal: renombrar un archivo.
2. **"Mismo filtro de bodega".** La consulta de ventas actual **no filtra por bodega** (solo `TIDO`, fecha y SKU; la bodega solo se usa en la consulta de stock). Para que numerador y denominador estén en la misma base, GDI se consulta con los mismos filtros que la venta, es decir, sin filtro de bodega. Costo si está mal: factor inflado por guías a otras bodegas; lo detecta el criterio "ALKA MENTOL entre 0,24 y 0,28" al correr contra Softland real.
3. **Factor no almacenable.** `NUMERIC(6,4)` admite hasta 99,9999. Un factor ≥ 99,9999 o negativo se guarda como `NULL` con advertencia (si no, el INSERT tumba la persistencia de la corrida completa). Venta neta de la ventana ≤ 0 → `NULL`.
4. **Constantes extra de `Config`.** El spec lista cuatro; la advertencia de deriva necesita además `semanas_factor_largo = 52` y `umbral_deriva_factor = 0.10`, ambos del texto del spec.
5. **Cobertura mínima/máxima de la vista** también se amplifican con el factor (el spec de backend lo pide y dice que "la corrección llega sola" si leen la vista).
6. **`resumen_cobertura`** (solo log) pasa a usar el factor, porque su docstring dice ser "el mismo cálculo que la vista" y el log no debe contradecirla.
7. **Rama:** `feat/cobertura-bonificado`, creada desde `feat/backtest-extendido` (aún sin mergear a `main`), porque ahí viven `tests/`, `requirements-dev.txt` y la migración 005. Si se mergea primero el backtest, esta rama queda encima sin conflicto relevante.

## Global Constraints

- Factor = `GDI / (FCV + FDV − NCV)`, todo `MAEDDO.CAPRCO1`, **agregado por SKU en la ventana y recién entonces dividido** (no promedio de factores semanales).
- Ventana: `semanas_factor_bonif: int = 13` últimas semanas **completas** (misma definición de semana cerrada que `completar_grilla`).
- Config (valores verbatim del spec): `semanas_factor_bonif: int = 13`, `tido_bonificacion: tuple = ('GDI',)`, `skus_sin_factor: tuple = ('5131012',)`, `factor_bonif_maximo: float = 0.60`.
- `5131012` → factor `NULL` sin calcularlo. Venta de la ventana cero → `NULL`.
- Factor sobre `factor_bonif_maximo`: **se usa igual, no se recorta**, solo advierte. Factor de 13 semanas que difiere en más de 10 puntos porcentuales del de 52: advertir con los dos valores.
- GRI queda fuera. No hace falta una consulta aparte ni una conexión nueva.
- **El job NO debe tocar:** serie de entrenamiento (sigue siendo venta facturada), winsorización, detección de censura, MAPE, MAPE a 4 semanas, cobertura empírica del intervalo, filas de `serie_semanal` (`real`, `backtest`, `proyeccion`).
- Persistencia: `metrica_sku`, una fila por SKU y corrida.
- SQL: `DROP VIEW` + `CREATE VIEW` (no `CREATE OR REPLACE`) para `v_cobertura_actual`; script idempotente; `COALESCE(factor, 0)` de modo que con todos los factores `NULL` la cobertura sea idéntica al decimal a la de hoy.
- La vista expone además: `factor_bonificacion`, `salida_semanal = demanda_semanal * (1 + COALESCE(factor, 0))`; `cobertura_semanas = stock_fisico / salida_semanal`.
- Cifras de `--dry-run --desde-csv` sobre `dataset_cluster_AXAY_24-08-2026.csv` deben ser **idénticas** antes y después (Winsorización, MAPE semanal, MAPE 4 semanas, SKUs ≤ 20 %, SKUs sin stock).

## Review Focus

- Venta neta de la ventana cero o negativa (notas de crédito superan la venta): factor `NULL`, sin `ZeroDivisionError` ni factor negativo (Tarea 1).
- Factor ≥ 99,9999 (venta casi nula con bonificado): `NULL` + advertencia en vez de reventar el INSERT de toda la corrida (Tarea 1).
- Día con solo guías GDI y ninguna venta: no debe aparecer en la tabla de ventas, o cambiaría el inicio de la grilla y las 157 semanas de historia (Tarea 2).
- `--desde-csv` (no trae bonificado): todos los factores `NULL`, métricas idénticas, sin errores (Tarea 3).
- Orden de despliegue: el job nuevo hace `INSERT` con la columna `factor_bonificacion`; si corre antes que la migración 006, la persistencia falla. Queda escrito en la Tarea 3 y en el cierre (Tarea 4).

---

## File Structure

- Modify: `batch_prediccion_jnc.py` — `Config`, helpers de factor, extracción, `main`, `guardar_metricas`, `resumen_cobertura`.
- Create: `sql/006_factor_bonificacion.sql`
- Create: `tests/test_bonificacion.py`

---

### Task 1: Config y cálculo del factor (funciones puras)

**Files:**
- Modify: `batch_prediccion_jnc.py` (`Config` ~línea 71-125; funciones nuevas tras `completar_grilla`; `completar_grilla` usa el helper)
- Create: `tests/test_bonificacion.py`

**Interfaces:**
- Produces:
  - `Config.semanas_factor_bonif: int = 13`, `semanas_factor_largo: int = 52`, `tido_bonificacion: tuple[str, ...] = ("GDI",)`, `skus_sin_factor: tuple[str, ...] = ("5131012",)`, `factor_bonif_maximo: float = 0.60`, `umbral_deriva_factor: float = 0.10`.
  - `ultima_semana_cerrada(hoy: date | None = None) -> date`
  - `calcular_factores_bonificacion(ventas: pd.DataFrame, bonificado: pd.DataFrame | None, ultima_cerrada: date, cfg: Config = CONFIG) -> dict[str, dict[str, float | None]]` — `ventas` con columnas `kopr, semana, demanda`; `bonificado` con `kopr, semana, bonificado`; devuelve `{kopr: {"factor": f13|None, "factor_largo": f52|None}}` para todos los `cfg.skus`; `{}` si `bonificado is None`.
  - `advertir_factores(factores: dict, cfg: Config = CONFIG) -> None`

- [ ] **Step 1: Tests que fallan** (`tests/test_bonificacion.py`)

```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `python3 -m pytest tests/test_bonificacion.py -q`
Expected: FAIL (`semanas_factor_bonif` / `ultima_semana_cerrada` no existen).

- [ ] **Step 3: Config.** Agregar al final de `Config`, antes de `nombre_modelo` (cada campo con su comentario, mismo estilo que el resto):

```python
    # --- Bonificación (GDI) -------------------------------------------------
    # Mercadería bonificada ("compras 4, te regalo 1"): sale de la bodega pero no
    # aparece en ninguna factura. El factor es un dato paralelo a la demanda: no
    # entra al entrenamiento ni a las métricas.
    semanas_factor_bonif: int = 13   # ventana del factor, en semanas cerradas
    semanas_factor_largo: int = 52   # solo para advertir deriva de la mecánica comercial
    tido_bonificacion: tuple[str, ...] = ("GDI",)
    skus_sin_factor: tuple[str, ...] = ("5131012",)  # unidad de bonificado ≠ unidad de venta
    factor_bonif_maximo: float = 0.60    # sobre esto, solo advierte
    umbral_deriva_factor: float = 0.10   # |factor13 - factor52| sobre esto, advierte
```

y en `__post_init__` (junto a la otra validación):

```python
        if set(self.tido_bonificacion) & set(TIPOS_VENTA):
            raise ValueError(
                f"tido_bonificacion {self.tido_bonificacion} no puede incluir "
                f"tipos de venta {TIPOS_VENTA}: se contaría doble")
```

- [ ] **Step 4: Helpers.** En `completar_grilla` reemplazar `ultima_cerrada = lunes_de(date.today()) - timedelta(weeks=1)` por `ultima_cerrada = ultima_semana_cerrada()`. Justo antes de `completar_grilla` agregar:

```python
def ultima_semana_cerrada(hoy: date | None = None) -> date:
    """Lunes de la última semana ISO completa (la semana en curso no cuenta)."""
    return lunes_de(hoy or date.today()) - timedelta(weeks=1)
```

Después de `completar_grilla` agregar la sección:

```python
# ============================================================================
# Factor de bonificación
# ============================================================================

FACTOR_MAX_ALMACENABLE = 99.9999  # límite de NUMERIC(6,4)


def _suma_ventana(df: pd.DataFrame | None, col: str, hasta: date, semanas: int) -> dict:
    if df is None or df.empty:
        return {}
    desde = hasta - timedelta(weeks=semanas - 1)
    d = df[(df["semana"] >= desde) & (df["semana"] <= hasta)]
    return d.groupby("kopr")[col].sum().to_dict()


def _razon_bonificacion(kopr: str, bonif: float, venta: float, semanas: int) -> float | None:
    if venta <= 0:
        return None
    f = bonif / venta
    if not 0 <= f < FACTOR_MAX_ALMACENABLE:
        log.warning("  %s: factor de bonificación %.4f fuera de rango en %d semanas "
                    "(bonificado %.0f, venta neta %.0f). Se deja sin factor.",
                    kopr, f, semanas, bonif, venta)
        return None
    return float(f)


def calcular_factores_bonificacion(ventas: pd.DataFrame, bonificado: pd.DataFrame | None,
                                   ultima_cerrada: date,
                                   cfg: Config = CONFIG) -> dict[str, dict[str, float | None]]:
    """Bonificado sobre venta neta, agregado por SKU en la ventana y luego dividido.

    No es el promedio de los factores semanales: semana a semana el factor es
    errático porque la entrega del bonificado no cae en la semana de la factura.
    Al agregar ese ruido de calendario desaparece, y como la tarjeta proyecta
    cuatro semanas, el agregado es la escala que corresponde.

    `ventas` es la venta neta semanal cruda (sin winsorizar). Con
    `bonificado=None` (no hay fuente, p. ej. --desde-csv) devuelve {}.
    """
    if bonificado is None:
        return {}

    resultado: dict[str, dict[str, float | None]] = {}
    ventanas = {"factor": cfg.semanas_factor_bonif, "factor_largo": cfg.semanas_factor_largo}
    sumas = {clave: (_suma_ventana(ventas, "demanda", ultima_cerrada, n),
                     _suma_ventana(bonificado, "bonificado", ultima_cerrada, n))
             for clave, n in ventanas.items()}

    for kopr in cfg.skus:
        if kopr in cfg.skus_sin_factor:
            resultado[kopr] = {"factor": None, "factor_largo": None}
            continue
        resultado[kopr] = {
            clave: _razon_bonificacion(kopr, bon.get(kopr, 0.0), ven.get(kopr, 0.0), n)
            for clave, n in ventanas.items()
            for ven, bon in [sumas[clave]]
        }
    return resultado


def advertir_factores(factores: dict, cfg: Config = CONFIG) -> None:
    """Avisa de factores atípicos: reflejan decisiones comerciales que cambian sin aviso."""
    if not factores:
        return
    con_factor = sum(1 for v in factores.values() if v["factor"] is not None)
    log.info("Factor de bonificación (%d semanas): %d de %d SKUs con factor",
             cfg.semanas_factor_bonif, con_factor, len(factores))
    for kopr, v in sorted(factores.items()):
        f, largo = v["factor"], v["factor_largo"]
        if f is None:
            continue
        if f > cfg.factor_bonif_maximo:
            log.warning("  %s: factor de bonificación %.1f%% sobre el máximo de %.0f%%. "
                        "Se usa igual; verificar que no sea un error de datos.",
                        kopr, 100 * f, 100 * cfg.factor_bonif_maximo)
        if largo is not None and abs(f - largo) > cfg.umbral_deriva_factor:
            log.warning("  %s: el factor de %d semanas (%.1f%%) difiere del de %d semanas "
                        "(%.1f%%) en más de %.0f puntos: la mecánica comercial cambió.",
                        kopr, cfg.semanas_factor_bonif, 100 * f,
                        cfg.semanas_factor_largo, 100 * largo,
                        100 * cfg.umbral_deriva_factor)
```

- [ ] **Step 5: Verificar** `python3 -m pytest tests/ -q` → todos PASS (incluye los 20 del backtest).

- [ ] **Step 6: Commit**

```bash
git add batch_prediccion_jnc.py tests/test_bonificacion.py
git commit -m "feat: factor de bonificación (config y cálculo agregado por ventana)"
```

(Cerrar el mensaje con `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.)

---

### Task 2: Extracción de GDI en la misma consulta

**Files:**
- Modify: `batch_prediccion_jnc.py` (`leer_ventas_mssql` ~línea 233; helpers nuevos junto a `semanalizar`)
- Test: `tests/test_bonificacion.py`

**Interfaces:**
- Consumes: `Config.tido_bonificacion`, `TIPOS_VENTA`, `TIPOS_NEGATIVOS`.
- Produces:
  - `_query_ventas(cfg: Config) -> str`
  - `separar_ventas_y_bonificado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` — entrada con `kopr, fecha, unidades, bonificado, lineas_venta`; salida `(ventas[kopr, fecha, unidades], bonificado[kopr, fecha, bonificado])`
  - `semanalizar_bonificado(bonificado: pd.DataFrame) -> pd.DataFrame` (`kopr, semana, bonificado`)
  - `leer_ventas_mssql(conn, cfg) -> tuple[pd.DataFrame, pd.DataFrame]` (**cambia el retorno**: `(ventas, bonificado)`)

- [ ] **Step 1: Tests que fallan**

```python
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
```

- [ ] **Step 2: Verificar que fallan** — `python3 -m pytest tests/test_bonificacion.py -q` → FAIL (`_query_ventas` no existe).

- [ ] **Step 3: Implementar.** Reemplazar `leer_ventas_mssql` por:

```python
def _query_ventas(cfg: Config) -> str:
    venta = _lista_sql(TIPOS_VENTA)
    negativos = _lista_sql(TIPOS_NEGATIVOS)
    bonif = _lista_sql(cfg.tido_bonificacion)
    return f"""
    SELECT
      RTRIM(DDO.KOPRCT) AS kopr,
      CONVERT(DATE, EDO.FEEMDO) AS fecha,
      SUM(CASE WHEN EDO.TIDO IN ({venta})
               THEN DDO.CAPRCO1 * CASE WHEN EDO.TIDO IN ({negativos}) THEN -1 ELSE 1 END
               ELSE 0 END) AS unidades,
      SUM(CASE WHEN EDO.TIDO IN ({bonif}) THEN DDO.CAPRCO1 ELSE 0 END) AS bonificado,
      SUM(CASE WHEN EDO.TIDO IN ({venta}) THEN 1 ELSE 0 END) AS lineas_venta
    FROM dbo.MAEEDO AS EDO WITH (NOLOCK)
    INNER JOIN dbo.MAEDDO AS DDO WITH (NOLOCK) ON EDO.IDMAEEDO = DDO.IDMAEEDO
    WHERE EDO.TIDO IN ({_lista_sql(TIPOS_VENTA + tuple(cfg.tido_bonificacion))})
      AND EDO.FEEMDO >= DATEADD(MONTH, -{cfg.meses_historia}, GETDATE())
      AND EDO.FEEMDO <  DATEADD(DAY, 1, GETDATE())
      AND RTRIM(DDO.KOPRCT) IN ({_lista_sql(cfg.skus)})
    GROUP BY RTRIM(DDO.KOPRCT), CONVERT(DATE, EDO.FEEMDO)
    """


def separar_ventas_y_bonificado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte el resultado de `_query_ventas` en la venta y el bonificado.

    La venta conserva solo los días con al menos una línea de venta, que son
    exactamente las filas que devolvía la consulta antes de agregar GDI. Un
    día con solo guías de bonificado no debe aparecer ahí: ampliaría la grilla
    de semanas y movería la historia sobre la que se midió el modelo.
    """
    ventas = (df.loc[df["lineas_venta"] > 0, ["kopr", "fecha", "unidades"]]
                .reset_index(drop=True))
    bonificado = (df.loc[df["bonificado"] != 0, ["kopr", "fecha", "bonificado"]]
                    .reset_index(drop=True))
    return ventas, bonificado


def leer_ventas_mssql(conn, cfg: Config = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ventas diarias por SKU y bonificado diario, desde Softland, en una sola pasada.

    La parte de venta es la consulta validada en la reconciliación del
    29-09-2026 contra el dataset de entrenamiento. Cualquier cambio en los tipos
    de documento, en la columna de cantidad o en la clave de producto rompe la
    equivalencia con los datos sobre los que se midió el error del modelo; por
    eso el bonificado solo se *agrega* como segundo agregado y la venta que se
    devuelve es la misma de antes.
    """
    df = pd.read_sql(_query_ventas(cfg), conn)
    df["kopr"] = df["kopr"].astype(str).str.strip()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["unidades"] = df["unidades"].astype(float)
    df["bonificado"] = df["bonificado"].astype(float)
    return separar_ventas_y_bonificado(df)
```

Junto a `semanalizar` agregar:

```python
def semanalizar_bonificado(bonificado: pd.DataFrame) -> pd.DataFrame:
    """Agrega el bonificado diario a semanas ISO (lunes)."""
    if bonificado.empty:
        return pd.DataFrame(columns=["kopr", "semana", "bonificado"])
    d = bonificado.copy()
    d["semana"] = d["fecha"].map(lunes_de)
    return d.groupby(["kopr", "semana"], as_index=False)["bonificado"].sum()
```

(`main` aún llama `leer_ventas_mssql` con el retorno antiguo; se actualiza en la Tarea 3. No correr el job entre tareas.)

- [ ] **Step 4: Verificar** `python3 -m pytest tests/ -q` → PASS.

- [ ] **Step 5: Commit** `git commit -m "feat: extraer GDI en la misma consulta de ventas"` (con la línea Co-Authored-By).

---

### Task 3: Integración en `main`, persistencia y log de cobertura

**Files:**
- Modify: `batch_prediccion_jnc.py` (`main`, `guardar_metricas`, `resumen_cobertura`)
- Test: `tests/test_bonificacion.py`

**Interfaces:**
- Consumes: todo lo de las Tareas 1 y 2.
- Produces: `metricas["factor_bonificacion"]: float | None` por SKU (agregado en `main`, **no** en `procesar_sku`); `guardar_metricas` escribe `factor_bonificacion` (redondeado a 4 decimales); `resumen_cobertura(resultados, stock, cfg=CONFIG, factores=None)`.

- [ ] **Step 1: Capturar la línea base** (antes de cualquier cambio en `main`, desde el HEAD de la Tarea 2):

```bash
CSV="/Users/josemarianaranjo/Desktop/MSI/SEMINARIO DE GRADO/1/0.entrenamiento_base_8-09-2026/dataset_cluster_AXAY_24-08-2026.csv"
python3 batch_prediccion_jnc.py --desde-csv "$CSV" --dry-run 2>&1 | grep -E "Historia|Winsorización|MAPE|SKUs sin stock|Modelados" > /tmp/bonif_base.log; cat /tmp/bonif_base.log
```

(Si `main` ya no corre porque `leer_ventas_mssql` cambió de retorno, no importa: el modo CSV no lo llama.) Guardar el archivo en el scratchpad, no en el repo.

- [ ] **Step 2: Tests que fallan**

```python
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
```

- [ ] **Step 3: Verificar que fallan** — `python3 -m pytest tests/test_bonificacion.py -q` → FAIL (`factores` no existe, SQL sin factor).

- [ ] **Step 4: `guardar_metricas`.** Reemplazar su cuerpo:

```python
def guardar_metricas(conn, batch_id: int, metricas: list[dict]) -> None:
    from psycopg2.extras import execute_values

    if not metricas:
        return

    def _factor(m):
        f = _nulo(m.get("factor_bonificacion"))
        return None if f is None else round(float(f), 4)

    filas = [(batch_id, m["kopr"], m["semanas_backtest"], m["semanas_para_mape"],
              m["semanas_censuradas"], _nulo(m["mape_semanal"]),
              _nulo(m["mae_semanal"]), _nulo(m["mape_4sem"]),
              m["bloques_4sem"], m["en_quiebre"], _factor(m)) for m in metricas]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.metrica_sku
                (batch_run_id, kopr, semanas_backtest, semanas_para_mape,
                 semanas_censuradas, mape_semanal, mae_semanal, mape_4sem,
                 bloques_4sem, en_quiebre, factor_bonificacion)
            VALUES %s
        """, filas)
    conn.commit()
```

- [ ] **Step 5: `resumen_cobertura`.** Firma `(resultados, stock, cfg: Config = CONFIG, factores: dict | None = None)`; actualizar el docstring ("…incluyendo la salida por bonificado cuando hay factor"). En el bucle de filas:

```python
        f = ((factores or {}).get(r.kopr) or {}).get("factor") or 0.0
        filas.append((r.kopr, float(r.stock_fisico) / (d * (1 + f))))
```

- [ ] **Step 6: `main`.**
  1. En la rama MSSQL: `diarias, bonificado = leer_ventas_mssql(cx, cfg)`; ampliar el `log.info` con `len(bonificado)` filas de bonificado; después de `semanal = semanalizar(diarias)` agregar `bonif_sem = semanalizar_bonificado(bonificado)`.
  2. En la rama CSV, tras leer: `bonif_sem = None` y `log.info("Sin datos de bonificado en el modo --desde-csv: el factor queda NULL")`.
  3. Después de `datos = completar_grilla(...)` y su chequeo de vacío, y antes de `# --- 2. Tratamiento`, **sin modificar `datos`**:

```python
    factores = calcular_factores_bonificacion(semanal, bonif_sem, ultima_semana_cerrada(), cfg)
    advertir_factores(factores, cfg)
```

  4. Tras el lazo de modelado y antes de `resumen(...)`:

```python
    for r in resultados:
        r["metricas"]["factor_bonificacion"] = (factores.get(r["kopr"]) or {}).get("factor")
```

  5. `resumen_cobertura(resultados, stock, cfg, factores)`.
  6. **No tocar** `procesar_sku`, `armar_filas_serie`, winsorización ni censura.

- [ ] **Step 7: Verificar** `python3 -m pytest tests/ -q` → PASS.

- [ ] **Step 8: Regresión de cifras** (criterio principal):

```bash
python3 batch_prediccion_jnc.py --desde-csv "$CSV" --dry-run 2>&1 | grep -E "Historia|Winsorización|MAPE|SKUs sin stock|Modelados" > /tmp/bonif_nuevo.log
diff /tmp/bonif_base.log /tmp/bonif_nuevo.log && echo IDENTICO
```
Expected: `IDENTICO` (salvo la marca de tiempo de cada línea: comparar con `cut -c10-` si las horas difieren). Además la salida debe incluir la línea `Sin datos de bonificado…` y no fallar.

- [ ] **Step 9: Commit** `git commit -m "feat: guardar factor de bonificación y usarlo en el log de cobertura"` (con Co-Authored-By).

---

### Task 4: Migración 006

**Files:**
- Create: `sql/006_factor_bonificacion.sql`
- Test: `tests/test_bonificacion.py`

**Interfaces:**
- Produces: columna `predictions.metrica_sku.factor_bonificacion NUMERIC(6,4)`; vista `predictions.v_cobertura_actual` con las mismas columnas de hoy, en el mismo orden y con el mismo cálculo cuando el factor es `NULL`, más `factor_bonificacion` y `salida_semanal` **al final**.

- [ ] **Step 1: Test que falla** (análisis estático del archivo; no hay base disponible en el test)

```python
SQL_006 = Path(__file__).resolve().parent.parent / "sql" / "006_factor_bonificacion.sql"
COLUMNAS_004 = ["kopr", "nombre", "cluster_abcxyz", "semana_corte", "bodega", "stock_fisico",
                "demanda_semanal", "demanda_inf", "demanda_sup", "cobertura_semanas",
                "cobertura_min", "cobertura_max", "en_quiebre", "mape_semanal", "mape_4sem",
                "batch_run_id"]


def test_migracion_006_forma():
    sql = SQL_006.read_text()
    assert "ADD COLUMN IF NOT EXISTS factor_bonificacion NUMERIC(6,4)" in sql
    assert "DROP VIEW IF EXISTS predictions.v_cobertura_actual" in sql
    assert "CREATE VIEW predictions.v_cobertura_actual" in sql
    assert "CREATE OR REPLACE VIEW" not in sql
    assert sql.count("COALESCE(m.factor_bonificacion, 0)") >= 3
    assert sql.strip().upper().rstrip(";").endswith("COMMIT")


def test_migracion_006_conserva_columnas_de_004_y_agrega_al_final():
    import re
    sql = SQL_006.read_text()
    cuerpo = sql.split("CREATE VIEW predictions.v_cobertura_actual", 1)[1]
    seleccion = cuerpo.split("FROM corrida AS c", 1)[0].rsplit("SELECT", 1)[-1]
    nombres = re.findall(r"(?:AS\s+(\w+)|\b\w+\.(\w+))\s*,?\s*(?:\n|$)", seleccion)
    nombres = [a or b for a, b in nombres]
    assert nombres[:len(COLUMNAS_004)] == COLUMNAS_004
    assert nombres[len(COLUMNAS_004):] == ["factor_bonificacion", "salida_semanal"]
```

- [ ] **Step 2: Verificar que falla** — archivo inexistente.

- [ ] **Step 3: Crear `sql/006_factor_bonificacion.sql`.** Encabezado de comentario al estilo de `004` (motor PostgreSQL, fecha 21-10-2026; explicar que la bodega se vacía con venta facturada más bonificado GDI; que el factor viene del job; que la vista se recrea con `DROP`+`CREATE` porque `CREATE OR REPLACE` solo admite agregar columnas al final; "Correr después de 005. Idempotente."). Cuerpo:

```sql
BEGIN;

ALTER TABLE predictions.metrica_sku
  ADD COLUMN IF NOT EXISTS factor_bonificacion NUMERIC(6,4);

COMMENT ON COLUMN predictions.metrica_sku.factor_bonificacion IS
  'Bonificado (GDI) sobre venta neta, agregado en las ultimas 13 semanas. '
  'NULL cuando no se pudo calcular o el SKU esta excluido.';

DROP VIEW IF EXISTS predictions.v_cobertura_actual;

CREATE VIEW predictions.v_cobertura_actual AS
WITH corrida AS (
    SELECT id, semana_corte FROM predictions.batch_run
    WHERE status = 'ok' ORDER BY finished_at DESC LIMIT 1
),
proyectado AS (
    SELECT s.kopr,
           AVG(s.unidades)     AS demanda_semanal,
           AVG(s.unidades_inf) AS demanda_inf,
           AVG(s.unidades_sup) AS demanda_sup,
           COUNT(*)            AS semanas_proyectadas
    FROM predictions.serie_semanal AS s
    JOIN corrida AS c ON c.id = s.batch_run_id
    WHERE s.tipo = 'proyeccion'
    GROUP BY s.kopr
)
SELECT
    k.kopr,
    k.nombre,
    k.cluster_abcxyz,
    c.semana_corte,
    st.bodega,
    st.stock_fisico,
    ROUND(p.demanda_semanal, 1)                        AS demanda_semanal,
    ROUND(p.demanda_inf, 1)                            AS demanda_inf,
    ROUND(p.demanda_sup, 1)                            AS demanda_sup,
    CASE WHEN p.demanda_semanal > 0
         THEN ROUND(st.stock_fisico / (p.demanda_semanal * (1 + COALESCE(m.factor_bonificacion, 0))), 1) END AS cobertura_semanas,
    CASE WHEN p.demanda_sup > 0
         THEN ROUND(st.stock_fisico / (p.demanda_sup * (1 + COALESCE(m.factor_bonificacion, 0))), 1) END     AS cobertura_min,
    CASE WHEN p.demanda_inf > 0
         THEN ROUND(st.stock_fisico / (p.demanda_inf * (1 + COALESCE(m.factor_bonificacion, 0))), 1) END     AS cobertura_max,
    COALESCE(m.en_quiebre, FALSE)                      AS en_quiebre,
    m.mape_semanal,
    m.mape_4sem,
    c.id AS batch_run_id,
    m.factor_bonificacion                              AS factor_bonificacion,
    ROUND(p.demanda_semanal * (1 + COALESCE(m.factor_bonificacion, 0)), 1) AS salida_semanal
FROM corrida AS c
JOIN predictions.stock_snapshot AS st ON st.batch_run_id = c.id
JOIN predictions.sku            AS k  ON k.kopr = st.kopr
LEFT JOIN proyectado            AS p  ON p.kopr = st.kopr
LEFT JOIN predictions.metrica_sku AS m ON m.batch_run_id = c.id AND m.kopr = st.kopr
WHERE k.activo;

COMMENT ON VIEW predictions.v_cobertura_actual IS
    'Una fila por SKU con el stock y para cuántas semanas alcanza. La cobertura usa la salida de bodega (venta facturada + bonificado): salida_semanal = demanda_semanal * (1 + factor_bonificacion). Con factor NULL equivale a la demanda sola. demanda_semanal sigue siendo venta facturada. Ordenar por cobertura_semanas deja arriba los productos en riesgo: es el ORDER BY del selector de producto.';

COMMIT;
```

(El `CASE WHEN demanda > 0` se conserva de la vista 004 a propósito: así los `NULL` de cobertura no cambian; con factor ≥ 0 la salida es > 0 exactamente cuando la demanda lo es. `cobertura_semanas` se calcula con `p.demanda_semanal` sin redondear, igual que antes; `salida_semanal` redondeada es solo informativa.)

- [ ] **Step 4: Verificar** `python3 -m pytest tests/ -q` → PASS.

- [ ] **Step 5: Commit** `git add sql/006_factor_bonificacion.sql tests/test_bonificacion.py && git commit -m "feat: migración 006, factor de bonificación y cobertura con salida amplificada"` (con Co-Authored-By).

---

## Fuera de este plan (necesita tu aprobación; escribe en dev o requiere Softland)

**Orden de despliegue:** aplicar `006` en la base **antes** de desplegar el job nuevo; si no, `guardar_metricas` falla en el `INSERT` y la corrida queda en `error`.

**V1 — Equivalencia de la vista (escribe DDL en dev).** Con la migración aún no aplicada, en una sola sesión: `CREATE TEMP TABLE cob_antes AS SELECT * FROM predictions.v_cobertura_actual;` → aplicar `006` → `SELECT count(*) FROM (SELECT <columnas de 004> FROM predictions.v_cobertura_actual EXCEPT SELECT <mismas> FROM cob_antes) x;` debe dar 0 (todos los factores son `NULL` porque la columna es nueva). Aplicar `006` una segunda vez para comprobar idempotencia.

**V2 — Corrida real (necesita `M32QUILLOTA_DATABASE_URL`).** `--dry-run` sin CSV contra Softland: `metrica_sku.factor_bonificacion` con valor en 21 SKUs y `NULL` en 5131012; ALKA MENTOL (5131018) entre 0,24 y 0,28; el log advierte por 5141403 (BIGTIME ULTRA REDBERRY); MAPE y SKUs bajo el umbral idénticos a la corrida anterior; mismo número de filas en `serie_semanal`. Si el factor sale muy distinto, revisar el ruling 2 (filtro de bodega).

## Self-Review

- **Cobertura del spec (secciones 1 y 2):** Config (T1), extracción en la misma consulta sin tocar GRI (T2), cálculo agregado + `NULL` por venta cero y por SKU excluido + las dos advertencias (T1), "lo que no hay que tocar" (T3 step 8 y ningún cambio en `procesar_sku`/winsorización/censura), persistencia en `metrica_sku` (T3), migración con `DROP`+`CREATE` y columnas nuevas (T4), criterios de aceptación del job y de la base (V1/V2 y T3 step 8).
- **Placeholders:** ninguno; el único dato externo es la ruta del CSV (hallada por el revisor previo) y las URLs de las bases (V2).
- **Consistencia de tipos:** `calcular_factores_bonificacion` devuelve `{kopr: {"factor", "factor_largo"}}` y esa forma la consumen `advertir_factores`, `main` y `resumen_cobertura`; `leer_ventas_mssql` devuelve `(ventas, bonificado)` y `main` desempaqueta igual; `ultima_semana_cerrada` es la misma en `completar_grilla` y en el factor.
