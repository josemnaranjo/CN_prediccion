# Backtest extendido Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que la línea del modelo cubra todo el período visible (~105 semanas por SKU) sin mover ni un decimal las métricas de la tesis.

**Architecture:** Se separan la *ventana de evaluación* (12 semanas, de ahí salen las métricas) y la *ventana de dibujo* (todo lo posible). El backtest reestima el ARIMA cada 12 semanas sobre ventana móvil de 78; los puntos de reajuste se cuentan **hacia atrás desde el final**, de modo que hay un reajuste exacto al inicio de la ventana de evaluación y ninguno dentro, lo que hace el cálculo de las últimas 12 semanas idéntico al actual.

**Tech Stack:** Python 3, pandas, numpy, statsmodels (SARIMAX), PostgreSQL, pytest (nuevo, solo dev).

**Spec:** `/Users/josemarianaranjo/Desktop/dashboard_CN/SPEC_backtest_extendido.md`

## Alcance de este plan

Este repo (`prediccion_stock_repo`) contiene el job y las migraciones SQL: **Tareas 1–5**.
El backend (`GET /prediccion/skus/:kopr/serie`) y el frontend (gráfico) **no están en este repo ni se encontraron en `dashboard_nestjs/src` ni `dashboard_reactjs/src`** (solo existen los `BRIEF_*.md`). Se cubren en la **Tarea 6** como contrato, para un plan/sesión aparte una vez ubicado ese código.

## Global Constraints

- `semanas_evaluacion = 12` — "NO cambiar: define las cifras reportadas en la tesis".
- `semanas_refit` debe ser igual a `semanas_evaluacion` para que las métricas sean comparables con el estudio.
- No tocar: tabla `serie_semanal` y su unicidad, `armar_filas_serie`, `proyectar_arima`, winsorización, detección de quiebres, lectura de Softland y de stock.
- No agregar un tipo nuevo de fila; la ventana de evaluación se distingue por fecha.
- No cambiar el esquema; si se registra el largo de la línea dibujada, es una clave nueva en `metricas` que NO se persiste.
- `metricas["semanas_backtest"]` significa "semanas sobre las que se calcularon las métricas" = `semanas_evaluacion`.
- `CREATE OR REPLACE VIEW` solo admite agregar columnas **al final**.
- Cifras que deben mantenerse exactas en `--dry-run` sobre `dataset_cluster_AXAY_24-08-2026.csv`: `111 semanas capadas (3.21%), 38657 unidades retiradas (2.76%)`; `MAPE semanal — mediana 25.18%`; `MAPE 4 semanas — mediana 17.58% | SKUs <= 20%: 15 de 22`; `SKUs sin stock al cierre: 5131012`; MAPE agrupado 28,16 %.

## Review Focus

- Serie con longitud justo en el mínimo (`min_semanas_sku + semanas_evaluacion` = 42): la ventana de dibujo no puede empezar antes de `min_semanas_sku`; no debe romper (Tarea 2).
- Serie corta (< 90 semanas, `disponible < 12`): debe dibujar exactamente 12 y comportarse como hoy (Tarea 2).
- `semanas_backtest` explícito menor que `semanas_evaluacion`: error claro al construir `Config` (Tarea 1).
- `semanas_backtest = 12` explícito: resultado idéntico al código actual (Tarea 2).
- SKU con semanas censuradas fuera de la ventana de evaluación: no deben afectar métricas ni `en_quiebre` (Tarea 3).
- Vista recreada dos veces (migración idempotente): no debe fallar (Tarea 4).

---

## File Structure

- Modify: `batch_prediccion_jnc.py` — `Config`, `backtest_arima`, `procesar_sku`, `resumen`.
- Create: `sql/005_evaluacion_en_vista.sql`
- Create: `tests/test_backtest.py` — tests con series sintéticas.
- Create: `requirements-dev.txt` — `pytest`.

---

### Task 1: Config con ventanas separadas

**Files:**
- Modify: `batch_prediccion_jnc.py:74-76` y `:111-113`
- Create: `requirements-dev.txt`, `tests/test_backtest.py`

**Interfaces:**
- Produces: `Config.semanas_evaluacion: int = 12`, `Config.semanas_backtest: int | None = None`, `Config.semanas_refit: int = 12`. `Config(semanas_backtest=5)` lanza `ValueError`.

- [ ] **Step 1: Instalar pytest**

```bash
echo "pytest>=8.0" > requirements-dev.txt && pip install -r requirements-dev.txt
```

- [ ] **Step 2: Test que falla** (`tests/test_backtest.py`)

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import batch_prediccion_jnc as b
from batch_prediccion_jnc import Config


def serie_sintetica(n=157, seed=0):
    rng = np.random.default_rng(seed)
    idx = list(pd.date_range("2023-10-02", periods=n, freq="7D").date)
    vals = 100 + 20 * np.sin(np.arange(n) / 6) + rng.normal(0, 8, n)
    return pd.Series(vals, index=idx)


def test_config_defaults():
    c = Config()
    assert (c.semanas_evaluacion, c.semanas_backtest, c.semanas_refit) == (12, None, 12)


def test_config_backtest_menor_que_evaluacion_falla():
    with pytest.raises(ValueError):
        Config(semanas_backtest=5)


def test_config_backtest_igual_a_evaluacion_ok():
    assert Config(semanas_backtest=12).semanas_backtest == 12
```

- [ ] **Step 3: Verificar que falla**

Run: `pytest tests/test_backtest.py -v` → FAIL (`semanas_evaluacion` no existe).

- [ ] **Step 4: Implementar.** Reemplazar las líneas 74-76:

```python
    semanas_entrenamiento: int = 78   # 18 meses, igual que en el estudio
    semanas_evaluacion: int = 12      # ventana de las métricas. NO cambiar: define
                                      # las cifras reportadas en la tesis
    semanas_backtest: int | None = None  # ventana dibujada; None = todo lo posible
    semanas_refit: int = 12           # cada cuántas semanas se reestima el modelo;
                                      # debe ser igual a semanas_evaluacion para que
                                      # las métricas sigan comparables con el estudio
    min_semanas_sku: int = 30         # mínimo para intentar modelar un SKU
```

y en `__post_init__`, antes de `p, d, q`:

```python
        if (self.semanas_backtest is not None
                and self.semanas_backtest < self.semanas_evaluacion):
            raise ValueError(
                f"semanas_backtest ({self.semanas_backtest}) debe ser >= "
                f"semanas_evaluacion ({self.semanas_evaluacion})")
```

- [ ] **Step 5: Verificar** `pytest tests/test_backtest.py -v` → PASS (el resto del código aún usa `cfg.semanas_backtest`; con `None` fallaría en `backtest_arima`, se arregla en la Tarea 2; no correr el job entre tareas).

- [ ] **Step 6: Commit**

```bash
git add batch_prediccion_jnc.py tests requirements-dev.txt
git commit -m "feat: separar ventana de evaluación y de dibujo en Config"
```

---

### Task 2: backtest_arima con reajuste periódico

**Files:**
- Modify: `batch_prediccion_jnc.py:465-508` (`backtest_arima`)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `Config` de la Tarea 1.
- Produces: `_n_dibujo(n_serie: int, cfg: Config) -> int`; `_puntos_refit(n_serie: int, inicio: int, cfg: Config) -> list[int]` (índices posicionales ascendentes, el primero siempre `inicio`); `backtest_arima(serie, cfg) -> tuple[pd.Series, list[tuple[float, float]]]` (firma igual; la serie ahora tiene `n_dibujo` puntos, los bloques son 3 con B=4 y eval=12).

- [ ] **Step 1: Tests que fallan** (agregar a `tests/test_backtest.py`)

```python
def backtest_legacy(serie, cfg):
    """Copia del algoritmo anterior (ajuste único + append sin refit), como referencia."""
    n_test = cfg.semanas_evaluacion
    entrenamiento, prueba = serie.iloc[:-n_test], serie.iloc[-n_test:]
    res = b._ajustar(entrenamiento.iloc[-cfg.semanas_entrenamiento:].values.astype(float), cfg)
    B, valores, pred, bloques = cfg.bloque_acumulado, prueba.values.astype(float), [], []
    for i in range(0, n_test, B):
        tramo = valores[i:i + B]
        if len(tramo) == B:
            f = np.clip(np.asarray(res.forecast(B), dtype=float), 0, None)
            bloques.append((float(tramo.sum()), float(f.sum())))
        for real in tramo:
            pred.append(float(np.clip(res.forecast(1)[0], 0, None)))
            res = res.append([real], refit=False)
    return pd.Series(pred, index=prueba.index), bloques


def test_n_dibujo():
    c = Config()
    assert b._n_dibujo(157, c) == 79          # 157 - 78
    assert b._n_dibujo(85, c) == 12           # disponible 7 < 12 -> sube a evaluacion
    assert b._n_dibujo(42, c) == 12
    assert b._n_dibujo(157, Config(semanas_backtest=30)) == 30
    assert b._n_dibujo(157, Config(semanas_backtest=500)) == 79


def test_puntos_refit_se_cuentan_desde_el_final():
    c = Config()
    n = 157
    pts = b._puntos_refit(n, n - 79, c)
    assert pts[0] == n - 79                   # ajuste inicial
    assert (n - 12) in pts                    # reajuste al inicio de la evaluación
    assert not any(n - 12 < p < n for p in pts)   # ninguno dentro de la evaluación
    assert all((n - p) % 12 == 0 for p in pts[1:])


def test_ultimas_12_semanas_identicas_al_legacy():
    s, c = serie_sintetica(), Config()
    pred, bloques = b.backtest_arima(s, c)
    ref_pred, ref_bloques = backtest_legacy(s, c)
    assert len(pred) == 79
    pd.testing.assert_series_equal(pred.iloc[-12:], ref_pred)
    assert bloques == ref_bloques and len(bloques) == 3


def test_no_hay_filas_antes_del_primer_entrenamiento():
    s, c = serie_sintetica(), Config()
    pred, _ = b.backtest_arima(s, c)
    assert pred.index[0] >= s.index[c.semanas_entrenamiento]


def test_backtest_12_explicito_igual_al_legacy():
    s, c = serie_sintetica(), Config(semanas_backtest=12)
    pred, bloques = b.backtest_arima(s, c)
    ref_pred, ref_bloques = backtest_legacy(s, c)
    pd.testing.assert_series_equal(pred, ref_pred)
    assert bloques == ref_bloques


def test_serie_corta_dibuja_12():
    s = serie_sintetica(n=45)
    pred, bloques = b.backtest_arima(s, Config())
    assert len(pred) == 12 and len(bloques) == 3
    assert pred.notna().all()


def test_cantidad_de_ajustes(monkeypatch):
    llamadas = []
    orig = b._ajustar
    monkeypatch.setattr(b, "_ajustar", lambda *a, **k: (llamadas.append(1), orig(*a, **k))[1])
    b.backtest_arima(serie_sintetica(), Config())
    assert len(llamadas) <= 8                 # ~79/12 ≈ 7, nunca uno por semana
```

- [ ] **Step 2: Verificar que fallan**

Run: `pytest tests/test_backtest.py -v` → FAIL (`_n_dibujo` no existe / `semanas_backtest=None`).

- [ ] **Step 3: Implementar.** Reemplazar `backtest_arima` completa (conservar su docstring ampliado) y agregar los helpers justo encima:

```python
def _n_dibujo(n_serie: int, cfg: Config) -> int:
    """Semanas de backtest que se dibujan para una serie de largo `n_serie`."""
    disponible = n_serie - cfg.semanas_entrenamiento
    n = disponible if cfg.semanas_backtest is None else min(cfg.semanas_backtest, disponible)
    n = max(n, cfg.semanas_evaluacion)   # nunca menos que la ventana de evaluación
    return min(n, n_serie - 1)           # nunca toda la serie


def _puntos_refit(n_serie: int, inicio: int, cfg: Config) -> list[int]:
    """Posiciones donde se reestima el modelo, contadas hacia atrás desde el final.

    Contar desde el final (y no desde el inicio) garantiza un reajuste justo al
    comienzo de la ventana de evaluación y ninguno dentro de ella. Siempre
    incluye `inicio`, que necesita un primer ajuste.
    """
    pts = {inicio}
    p = n_serie - cfg.semanas_refit
    while p > inicio:
        pts.add(p)
        p -= cfg.semanas_refit
    return sorted(pts)


def backtest_arima(serie: pd.Series,
                   cfg: Config = CONFIG) -> tuple[pd.Series, list[tuple[float, float]]]:
    """Backtest con reajuste periódico sobre ventana móvil, en dos horizontes.

    La predicción a un paso se genera para toda la ventana de dibujo (ver
    `_n_dibujo`). El modelo se reestima cada `semanas_refit` semanas sobre las
    `semanas_entrenamiento` previas (solo datos anteriores); entre reajustes se
    avanza con `append(..., refit=False)`. Cada predicción usa solo información
    anterior a la semana que estima.

    Los bloques acumulados de `bloque_acumulado` semanas se calculan solo dentro
    de la ventana de evaluación (las últimas `semanas_evaluacion`), alineados a
    su inicio, cada uno pronosticado de una sola vez desde su comienzo, a ciegas.
    Sumar predicciones a un paso daría un error mucho menor y engañoso.
    """
    N = len(serie)
    n = _n_dibujo(N, cfg)
    inicio = N - n
    if inicio < cfg.min_semanas_sku:
        return pd.Series(np.nan, index=serie.index[inicio:]), []

    eval_ini = N - cfg.semanas_evaluacion
    refits = set(_puntos_refit(N, inicio, cfg))
    B = cfg.bloque_acumulado
    valores = serie.values.astype(float)

    pred_1paso: list[float] = []
    bloques: list[tuple[float, float]] = []
    res = None
    for i in range(inicio, N):
        if i in refits:
            res = _ajustar(valores[max(0, i - cfg.semanas_entrenamiento):i], cfg)
        if i >= eval_ini and (i - eval_ini) % B == 0 and i + B <= N:
            # Pronóstico de todo el bloque desde su inicio, a ciegas.
            f = np.clip(np.asarray(res.forecast(B), dtype=float), 0, None)
            bloques.append((float(valores[i:i + B].sum()), float(f.sum())))
        pred_1paso.append(float(np.clip(res.forecast(1)[0], 0, None)))
        res = res.append([valores[i]], refit=False)

    return pd.Series(pred_1paso, index=serie.index[inicio:]), bloques
```

- [ ] **Step 4: Verificar** `pytest tests/test_backtest.py -v` → todos PASS. Si `test_ultimas_12_semanas_identicas_al_legacy` falla, los puntos de refit están mal alineados (deben contarse desde el final).

- [ ] **Step 5: Commit**

```bash
git add batch_prediccion_jnc.py tests/test_backtest.py
git commit -m "feat: backtest con reajuste periódico sobre ventana móvil"
```

---

### Task 3: procesar_sku mide solo la ventana de evaluación

**Files:**
- Modify: `batch_prediccion_jnc.py:570-615` (`procesar_sku`)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `backtest_arima` de la Tarea 2.
- Produces: `procesar_sku(...)["backtest"]` con ~79 filas; `["metricas"]` con las mismas claves de hoy, calculadas sobre las últimas `semanas_evaluacion`, más la clave nueva `semanas_dibujo` (no se persiste).

- [ ] **Step 1: Tests que fallan**

```python
def df_sku(n=157, censurar_antes=0):
    s = serie_sintetica(n)
    cens = np.zeros(n, bool)
    if censurar_antes:
        cens[10:10 + censurar_antes] = True       # lejos de la evaluación
    return pd.DataFrame({"semana": list(s.index), "demanda": s.values, "censurada": cens})


def test_metricas_solo_sobre_evaluacion():
    r = b.procesar_sku("X", df_sku(), Config())
    assert len(r["backtest"]) == 79
    m = r["metricas"]
    assert m["semanas_backtest"] == 12
    assert m["semanas_dibujo"] == 79
    assert m["bloques_4sem"] == 3
    ref_pred, _ = backtest_legacy(serie_sintetica(), Config())
    reales = serie_sintetica().loc[ref_pred.index]
    assert m["mape_semanal"] == pytest.approx(b.mape(reales.values, ref_pred.values))


def test_censura_antigua_no_afecta_metricas_ni_quiebre():
    base = b.procesar_sku("X", df_sku(), Config())["metricas"]
    con = b.procesar_sku("X", df_sku(censurar_antes=5), Config())["metricas"]
    assert con["mape_semanal"] == base["mape_semanal"]
    assert con["semanas_censuradas"] == 0 and con["en_quiebre"] is False


def test_guard_de_largo_minimo():
    assert b.procesar_sku("X", df_sku(n=41), Config()) is None
    assert b.procesar_sku("X", df_sku(n=42), Config()) is not None
```

- [ ] **Step 2: Verificar que fallan** — `pytest tests/test_backtest.py -v` → FAIL (`semanas_dibujo`, longitudes).

- [ ] **Step 3: Implementar** en `procesar_sku`:

```python
    minimo = cfg.min_semanas_sku + cfg.semanas_evaluacion
```

y reemplazar desde `pred_backtest, bloques_pron = ...` hasta `en_quiebre`:

```python
    pred_backtest, bloques_pron = backtest_arima(serie, cfg)

    # Las métricas salen solo de la ventana de evaluación; el resto del
    # backtest es únicamente para dibujar la línea del modelo.
    eval_idx = pred_backtest.index[-cfg.semanas_evaluacion:]
    reales_eval = serie.loc[eval_idx]
    pred_eval = pred_backtest.loc[eval_idx]
    cens_eval = censurada.loc[eval_idx].to_numpy(bool)

    # Un bloque acumulado se descarta completo si alguna de sus semanas está
    # censurada: su suma tampoco representa la demanda del período.
    B = cfg.bloque_acumulado
    bloques_limpios = [b for i, b in enumerate(bloques_pron)
                       if not cens_eval[i * B:(i + 1) * B].any()]

    m_sem = mape(reales_eval.values, pred_eval.values, excluir=cens_eval)
    m_acum, bloques = mape_bloques(bloques_limpios)
    mae = float(np.abs(reales_eval.values - pred_eval.values).mean())

    en_quiebre = bool(cens_eval[-1]) if len(cens_eval) else False
```

(conservar el comentario original sobre `en_quiebre`). En el dict de retorno: `reales`→`reales_eval`, `cens_test`→`cens_eval`, y

```python
            "semanas_backtest": int(len(pred_eval)),   # semanas de las métricas
            "semanas_dibujo": int(len(pred_backtest)), # solo log; no se persiste
```

- [ ] **Step 4: Verificar** `pytest tests/test_backtest.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add batch_prediccion_jnc.py tests/test_backtest.py
git commit -m "feat: métricas solo sobre la ventana de evaluación"
```

---

### Task 4: Log del resumen y migración 005

**Files:**
- Modify: `batch_prediccion_jnc.py:767` (`resumen`)
- Create: `sql/005_evaluacion_en_vista.sql`

- [ ] **Step 1: Resumen.** Reemplazar la línea del log:

```python
    log.info("Resumen del backtest (%d semanas evaluadas por SKU; línea dibujada de %d a %d semanas):",
             cfg.semanas_evaluacion, int(m["semanas_dibujo"].min()), int(m["semanas_dibujo"].max()))
```

- [ ] **Step 2: Migración.** Copiar el `SELECT` completo de `sql/003_censura_quiebres.sql` (`FROM`/`JOIN`/`WHERE` idénticos) en `sql/005_evaluacion_en_vista.sql`, con `CREATE OR REPLACE VIEW predictions.v_serie_actual AS` y **una única columna nueva al final**, tras `r.id AS batch_run_id`:

```sql
    r.id AS batch_run_id,
    m.semanas_backtest AS semanas_evaluacion
```

Encabezado de comentario al estilo de 004 (motor, fecha 19-10-2026, "Correr después de 004. Idempotente.") y al final:

```sql
COMMENT ON VIEW predictions.v_serie_actual IS
    'Serie vigente para el gráfico, con las métricas del SKU. Filtrar por kopr y ordenar por semana; `semana_corte` marca dónde termina lo real. `semanas_evaluacion` indica cuántas de las últimas filas de backtest son la ventana de evaluación. Si `sku_en_quiebre` es verdadero no habrá filas de tipo proyeccion.';
```

- [ ] **Step 3: Verificar sintaxis y que las columnas previas no cambian de orden**

Run: `diff <(grep -c . sql/003_censura_quiebres.sql >/dev/null; sed -n '/SELECT/,/batch_run_id/p' sql/003_censura_quiebres.sql) <(sed -n '/SELECT/,/batch_run_id/p' sql/005_evaluacion_en_vista.sql)` → sin diferencias. Luego aplicar dos veces en la base de desarrollo (Tarea 5) para comprobar idempotencia.

- [ ] **Step 4: Commit**

```bash
git add batch_prediccion_jnc.py sql/005_evaluacion_en_vista.sql
git commit -m "feat: resumen con ventanas separadas y migración 005"
```

---

### Task 5: Verificación de aceptación del job

**Files:** ninguno (solo verificación).

- [ ] **Step 1: Ubicar el CSV** `dataset_cluster_AXAY_24-08-2026.csv` (carpeta `datasets_entrenamiento/raw/`; no está en el árbol de `dashboard_CN`, pedirlo al usuario si no aparece con `find / -name "dataset_cluster_AXAY*" 2>/dev/null`).

- [ ] **Step 2: Correr con medición de tiempo**

```bash
time python batch_prediccion_jnc.py --desde-csv "<ruta>/dataset_cluster_AXAY_24-08-2026.csv" --dry-run 2>&1 | tee /tmp/dry_nuevo.log
```

Expected, exactamente: `111 semanas capadas (3.21%), 38657 unidades retiradas (2.76%)`; `MAPE semanal — mediana 25.18%`; `MAPE 4 semanas — mediana 17.58% | SKUs <= 20%: 15 de 22`; `SKUs sin stock al cierre: 5131012`. Tiempo: ~10–20 s más que antes. Si una cifra cambia, revisar `_puntos_refit` (alineación desde el final).

- [ ] **Step 3: Comparar contra la versión anterior** (`git stash`/`git worktree` en `a59f461`, mismo comando) y confirmar con `diff` que las líneas de métricas son idénticas.

- [ ] **Step 4: MAPE agrupado 28,16 %** — con un script en el scratchpad que llame `procesar_sku` por SKU y calcule `mape` sobre las últimas 12 filas de todos los `backtest` concatenados; debe dar 28,16. Verificar además filas por SKU entre 70 y 105 y `bloques_4sem == 3` en todos.

- [ ] **Step 5: Base de desarrollo.** Pedir confirmación al usuario antes de escribir: aplicar `sql/001..005` si falta alguna (005 dos veces), correr el job sin `--dry-run`, y verificar:

```sql
SELECT count(*) FROM predictions.serie_semanal s JOIN predictions.v_serie_actual v
  ON v.batch_run_id = s.batch_run_id AND v.kopr = s.kopr AND v.semana = s.semana
 WHERE s.tipo = 'backtest';   -- ~2.000 (antes 264)
SELECT DISTINCT semanas_evaluacion FROM predictions.v_serie_actual;  -- 12
```

- [ ] **Step 6: Notas para el informe** (agregar al README.md del repo): el backtest usa orígenes móviles con reajuste cada 12 semanas sobre ventana de 78; cierra el pendiente de evaluación con orígenes móviles. Commit `docs: nota de backtest con orígenes móviles`.

---

### Task 6: Backend y frontend (contrato; código fuera de este repo)

Primero ubicar el código del endpoint `GET /prediccion/skus/:kopr/serie` y de la vista del gráfico (los `BRIEF_backend_prediccion.md` y `BRIEF_frontend_prediccion.md` los describen; hoy no están en `dashboard_nestjs/src` ni `dashboard_reactjs/src`). Después, plan propio con este contrato:

- **Backend:** campo de nivel superior `semanaInicioEvaluacion: string | null` (YYYY-MM-DD, lunes) = `semana` de la fila `tipo='backtest'` que está `semanas_evaluacion` posiciones desde el final, ordenando por semana; `null` si no hay backtest. Leer `semanas_evaluacion` de `v_serie_actual`. Test: SKU 5131018 devuelve esa fecha; SKU sin backtest devuelve `null`.
- **Frontend:** línea del modelo partida en esa fecha, **compartiendo el punto de unión** (sin corte): backtest histórico naranja al 45 % de opacidad 1,5 px; ventana de evaluación naranja pleno 2 px; proyección naranja pleno 2 px punteado. Leyenda: "Venta real", "Modelo", "Modelo — 12 semanas evaluadas". Banda de fondo tenue opcional rotulada "ventana de evaluación" en ambos paneles. Nota al pie: mantener "Error semanal medio del modelo: X % sobre las 12 semanas de validación" y añadir que el tramo destacado es ese período. `null` no rompe la vista. Verificar en tema claro y oscuro; el fixture `fixture_5131018.json` debe incluir el campo nuevo.

---

## Self-Review

- **Cobertura de la spec:** Config (T1), backtest_arima con refit desde el final y bloques solo en evaluación (T2), procesar_sku/guard/métricas (T3), log (T4), migración 005 (T4), criterios de aceptación y pasos posteriores (T5), backend/frontend (T6, como contrato por no estar el código en este repo).
- **Placeholders:** el único dato externo es la ruta del CSV (T5.1), que se debe localizar; la migración 005 se arma copiando el SELECT de 003 con la columna extra.
- **Consistencia de tipos:** `_n_dibujo`, `_puntos_refit`, `semanas_dibujo`, `semanas_evaluacion` se usan igual en todas las tareas.
