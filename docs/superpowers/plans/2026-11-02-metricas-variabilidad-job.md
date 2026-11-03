# Métricas de variabilidad — Job Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que cada corrida calcule la variabilidad del stock y de la demanda cruda por SKU (52 semanas cerradas) y la guarde en las ocho columnas de `metrica_sku` creadas por la migración 009.

**Architecture:** Una función pura `calcular_variabilidad(datos, stock_hist, cfg)` hace un `join` por semana entre la demanda cruda y el stock reconstruido y devuelve un dict por SKU. `main` la llama después de reconstruir el stock y antes de `guardar_metricas`, dentro de un `try/except`; `guardar_metricas` escribe las ocho columnas (NULL cuando no hay dato). Todo en `batch_prediccion_jnc.py`, que concentra las funciones del job.

**Tech Stack:** Python 3.12, pandas, numpy, psycopg2, pytest, PostgreSQL.

**Spec:** Pegado en la conversación del 2026-11-02 ("Spec — métricas de variabilidad del inventario"); no existe como archivo en el repo. Este plan implementa su sección 3 (Job). La sección 2 (migración 009) está en `docs/superpowers/plans/2026-11-02-metricas-variabilidad-migracion.md`, ya aplicada en la BD dev.

## Global Constraints

- **El modelo no se toca.** MAPE semanal, MAPE a 4 semanas, filas de `serie_semanal` y la reconstrucción del stock quedan idénticos a la corrida anterior.
- Todo por SKU, sobre las **últimas 52 semanas cerradas**: `semanas_variabilidad: int = 52`, `min_semanas_variabilidad: int = 40` en `Config`.
- Las ocho medidas: `cv_demanda`, `cv_stock`, `ratio_variabilidad`, `media_demanda`, `media_stock`, `tasa_quiebre`, `semanas_variabilidad`, `prop_stock_reconstruido`.
- `cv_demanda` = desv. estándar / media de la demanda **cruda** (`demanda_cruda`, no `demanda` winsorizada). `cv_stock` sobre `stock_fin`. `ratio_variabilidad = cv_stock / cv_demanda`: cociente de coeficientes de variación, **no** de varianzas.
- Las semanas en quiebre **quedan dentro**, en ambos lados; no se excluyen ni se corrigen.
- `tasa_quiebre` = proporción de semanas con `stock_fin <= 0` (medición directa, no la marca `censurada`).
- `prop_stock_reconstruido` = proporción de la ventana con `origen = 'reconstruido'`.
- Con menos de 40 semanas con dato: las **ocho** columnas en `None`.
- Media de demanda o de stock `<= 0`: ese CV en `None` y el cociente también. `cv_demanda == 0`: cociente `None`, nunca infinito.
- Las dos series se alinean **por semana con un join**, sin asumir mismo orden ni mismo largo.
- Sin serie de stock en la corrida: las ocho en `None` y el job sigue.
- Una excepción en el bloque de variabilidad **no cambia el estado de la corrida**.
- Log: **una** línea, `Variabilidad (52 semanas): 21 de 22 SKUs, ratio mediano 1.35, quiebre medio 5.9%`. Sin advertencias por valores altos.

## Rulings previos a la ejecución (huecos del spec; costo si están mal: bajo)

1. **Orden en `main`.** El spec pide variabilidad "después de la reconstrucción del stock y antes de guardar las métricas", pero hoy `main` guarda las métricas primero y `generar_stock_historico` (que reconstruye **y** guarda, devolviendo un `int`) corre después. Decisión: `generar_stock_historico` pasa a devolver la serie (`pd.DataFrame | None`) y su bloque en `main` se mueve antes de `guardar_metricas`. Costo: si `guardar_metricas` falla, el histórico de esa corrida ya quedó guardado bajo una corrida en `error`; las vistas solo leen corridas `ok`, así que es inocuo.
2. **Desviación estándar muestral** (`ddof=1`, el default de pandas). El spec no la fija. El cociente de CVs no cambia con esa elección porque ambos lados usan el mismo n; solo mueve cada CV ~1 %.
3. **Redondeo a la precisión de la columna** dentro de `calcular_variabilidad` (CV y tasas a 4 decimales, medias a 2) y `ratio_variabilidad` calculado desde los CV ya redondeados, para que el criterio "el ratio iguala `cv_stock / cv_demanda`" se cumpla sobre lo guardado.
4. **Ratio que no cabe en `NUMERIC(8,4)`** (≥ 10000): va en `None`, no revienta el `INSERT` de toda la corrida. Lo señaló la revisión de la migración.
5. **Ventana**: las 52 semanas que terminan en `datos["semana"].max()` (la última semana cerrada), ambos extremos incluidos. `semanas_variabilidad` cuenta las semanas presentes en **ambas** series dentro de esa ventana.
6. **Las medias se reportan aunque el CV sea `None`** por media `<= 0`; solo se anulan con el mínimo de 40 semanas.

## Review Focus

1. SKU nuevo o stock que empieza después que la demanda: la ventana se recorta al cruce de ambas series; con menos de 40 semanas comunes, ocho `None` (Task 2).
2. Series con distinto orden o distinto largo (stock desordenado, demanda con semanas que el stock no tiene): el resultado no cambia (Task 2).
3. Demanda constante (`cv_demanda == 0`) o media de stock negativa (stock reconstruido puede serlo): `None`, nunca `inf` ni `NaN` hacia la base (Task 2).
4. Ratio gigante por `cv_demanda` casi cero: `None`, la corrida no falla al insertar (Task 2 y Task 3).
5. Sin serie de stock (`--desde-csv`, `reconstruir_stock = False`, sin ancla, o falla del histórico) o falla del propio cálculo: la corrida cierra `ok` y las métricas se guardan con las ocho en NULL (Task 4).

---

## File Structure

- `batch_prediccion_jnc.py` — `Config` (Task 1); `COLUMNAS_VARIABILIDAD`, `calcular_variabilidad`, `resumen_variabilidad` en una sección nueva "Variabilidad", después de `calidad_stock_historico` y antes de `armar_filas_serie` (Tasks 2–3); `guardar_metricas` (Task 3); `generar_stock_historico` y `main` (Task 4).
- `tests/test_variabilidad.py` — se amplía (ya contiene el test de la migración 009); mismo estilo que `tests/test_bonificacion.py` y `tests/test_stock_historico.py` (`import batch_prediccion_jnc as bp`, `Config`, fixtures con `monkeypatch`).
- `tests/test_bonificacion.py`, `tests/test_stock_historico.py` — ajustes puntuales por los cambios de Task 3 y 4.

---

### Task 1: Config

**Files:**
- Modify: `batch_prediccion_jnc.py` (`Config`, bloque nuevo "Variabilidad" antes de "Operación"; `__post_init__`)
- Test: `tests/test_variabilidad.py`

**Interfaces:**
- Produces: `Config.semanas_variabilidad: int = 52`, `Config.min_semanas_variabilidad: int = 40`.

- [ ] **Step 1: Write the failing tests** `test_config_variabilidad` (`Config().semanas_variabilidad == 52`, `Config().min_semanas_variabilidad == 40`) y `test_config_variabilidad_invalida` (`pytest.raises(ValueError)` para `Config(min_semanas_variabilidad=53)`, es decir mayor que `semanas_variabilidad`, y para `Config(min_semanas_variabilidad=1)`).
- [ ] **Step 2: Run** `pytest tests/test_variabilidad.py -k config -v` — Expected: FAIL (`AttributeError` / `DID NOT RAISE`).
- [ ] **Step 3: Implement** los dos campos con comentario al estilo del bloque "Bonificación", y en `__post_init__` un `ValueError` si `min_semanas_variabilidad < 2` o `> semanas_variabilidad`.
- [ ] **Step 4: Run** el mismo comando — Expected: PASS.
- [ ] **Step 5: Commit** `feat: configuración de la ventana de variabilidad`.

---

### Task 2: `calcular_variabilidad`

**Files:**
- Modify: `batch_prediccion_jnc.py`
- Test: `tests/test_variabilidad.py`

**Interfaces:**
- Consumes: `Config.semanas_variabilidad`, `Config.min_semanas_variabilidad` (Task 1). `datos`: columnas `kopr` (str), `semana` (date), `demanda_cruda`. `stock_hist`: columnas `kopr`, `semana`, `stock_fin`, `origen` (salida de `armar_serie_historica`), o `None`.
- Produces: `COLUMNAS_VARIABILIDAD: tuple[str, ...]` con las ocho medidas en el orden de la migración, y `calcular_variabilidad(datos: pd.DataFrame, stock_hist: pd.DataFrame | None, cfg: Config = CONFIG) -> dict[str, dict]`: una clave por cada `kopr` de `datos`, valor = dict con exactamente las ocho claves; valores `float`/`int` de Python o `None` (nunca `NaN`, nunca `numpy`).

- [ ] **Step 1: Write the failing tests.** Helper local `_series(n, demanda, stock, origen="reconstruido")` que arma `datos` y `stock_hist` de un SKU `"A"` con `n` semanas consecutivas terminando en `date(2026, 10, 26)`. Tests, con estos valores:
  - `test_variabilidad_caso_base`: demanda alterna 10/30 (52 semanas), stock = 2 × demanda → `cv_demanda == cv_stock == 0.5049`, `ratio_variabilidad == 1.0`, `media_demanda == 20.0`, `media_stock == 40.0`, `tasa_quiebre == 0.0`, `semanas_variabilidad == 52`, `prop_stock_reconstruido == 1.0`.
  - `test_variabilidad_usa_demanda_cruda`: `datos` con `demanda` ≠ `demanda_cruda`; el resultado sale de `demanda_cruda`.
  - `test_variabilidad_toma_las_ultimas_52`: 60 semanas → `semanas_variabilidad == 52` y las 8 primeras semanas (con un valor extremo) no influyen.
  - `test_variabilidad_con_45_semanas_calcula` → `semanas_variabilidad == 45`; `test_variabilidad_bajo_el_minimo_deja_todo_en_none` con 39 semanas → las ocho `None`.
  - `test_variabilidad_quiebre_y_reconstruido`: 13 de 52 semanas con `stock_fin <= 0` (una exactamente en 0) → `tasa_quiebre == 0.25`; 26 de 52 con `origen="reconstruido"` y 26 `"capturado"` → `prop_stock_reconstruido == 0.5`. Las semanas en quiebre entran al CV (el `cv_stock` difiere del de la misma serie sin ceros).
  - `test_variabilidad_alinea_por_semana`: `stock_hist` desordenado (`sample(frac=1)`) y sin las 5 semanas más antiguas, `datos` con 60 → mismo resultado que con ambos ordenados y recortados a las 55 semanas comunes que caen en la ventana; `semanas_variabilidad == 52`.
  - `test_variabilidad_sku_nuevo_sin_historia_comun`: demanda de un SKU con 30 semanas comunes → ocho `None`.
  - `test_variabilidad_demanda_constante`: `cv_demanda == 0.0`, `ratio_variabilidad is None`, el resto calculado.
  - `test_variabilidad_media_no_positiva`: stock con media `<= 0` → `cv_stock is None` y `ratio_variabilidad is None`, `media_stock` negativa reportada; igual para demanda con media 0.
  - `test_variabilidad_ratio_gigante_va_en_none`: `cv_demanda` ≈ 0.0001 y `cv_stock` ≈ 1 → `ratio_variabilidad is None`.
  - `test_variabilidad_sin_serie_de_stock`: `stock_hist=None` y `stock_hist` vacío → un dict por SKU de `datos` con las ocho claves en `None`.
  - `test_variabilidad_tipos_de_python`: ningún valor es `numpy.generic` ni `NaN`.
  - `test_variabilidad_varios_skus_independientes`: dos SKUs, uno sin historia suficiente; el otro no se ve afectado.
- [ ] **Step 2: Run** `pytest tests/test_variabilidad.py -k variabilidad -v` — Expected: FAIL (`AttributeError: ... calcular_variabilidad`).
- [ ] **Step 3: Implement `calcular_variabilidad`** en una sección "Variabilidad". Por SKU: `merge` interno de demanda y stock por `["kopr","semana"]`, filtrado a `fin - 7*(semanas_variabilidad - 1)` días `<= semana <= fin` con `fin = datos["semana"].max()`; si quedan menos de `min_semanas_variabilidad` filas → ocho `None`. CV con `std(ddof=1)/mean`, `None` si la media es `<= 0`. Redondeo y guardas según los rulings 3, 4 y 6 (ratio desde los CV redondeados, `None` si `cv_demanda == 0` o ratio `>= 10000`). Convertir a `float`/`int` de Python.
- [ ] **Step 4: Run** el mismo comando — Expected: PASS.
- [ ] **Step 5: Commit** `feat: calcular_variabilidad por SKU sobre 52 semanas`.

---

### Task 3: Persistencia y línea de log

**Files:**
- Modify: `batch_prediccion_jnc.py` (`guardar_metricas`; `resumen_variabilidad` nueva, junto a `calcular_variabilidad`)
- Modify: `tests/test_bonificacion.py` (`test_guardar_metricas_persiste_el_factor`: leer el factor por posición `[10]` en vez de `[-1]`, porque las columnas nuevas pasan a ir al final)
- Test: `tests/test_variabilidad.py`

**Interfaces:**
- Consumes: `COLUMNAS_VARIABILIDAD` (Task 2). Un dict de métricas puede o no traer esas claves.
- Produces: `guardar_metricas` inserta además las ocho columnas, tomando `m.get(col)` (ausente → NULL), con `None` para `NaN`. `resumen_variabilidad(metricas: list[dict], cfg: Config = CONFIG) -> None` escribe por `log.info` la línea del spec.

- [ ] **Step 1: Write the failing tests** con el patrón de `test_guardar_metricas_persiste_el_factor` (`execute_values` capturado, `_FakeConn`):
  - `test_guardar_metricas_persiste_la_variabilidad`: dos métricas, una con las ocho medidas y otra sin ninguna clave → el SQL contiene las ocho columnas y las filas llevan los valores en el orden de `COLUMNAS_VARIABILIDAD` y ocho `None` la segunda.
  - `test_resumen_variabilidad_linea`: 22 métricas, 21 con `semanas_variabilidad` no nulo, ratio mediano 1.35 y `tasa_quiebre` media 0.059 → `caplog` contiene exactamente `Variabilidad (52 semanas): 21 de 22 SKUs, ratio mediano 1.35, quiebre medio 5.9%` y ningún `WARNING`.
  - `test_resumen_variabilidad_sin_datos`: todas en `None` → `0 de 22 SKUs, ratio mediano n/d, quiebre medio n/d`.
- [ ] **Step 2: Run** `pytest tests/test_variabilidad.py tests/test_bonificacion.py -k "metricas or resumen_variabilidad" -v` — Expected: FAIL.
- [ ] **Step 3: Implement** la ampliación del `INSERT` (nueve a diecinueve columnas; nombres de la migración 009) y `resumen_variabilidad` (SKUs con dato = `semanas_variabilidad is not None`; mediana sobre los ratios no nulos; quiebre medio sobre los SKUs con dato; la semana de la etiqueta sale de `cfg.semanas_variabilidad`).
- [ ] **Step 4: Run** el mismo comando más `pytest tests/test_bonificacion.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: guardar la variabilidad en metrica_sku y registrar su resumen`.

---

### Task 4: Integración en `main`

**Files:**
- Modify: `batch_prediccion_jnc.py` (`generar_stock_historico`, `main`)
- Modify: `tests/test_stock_historico.py` (los tests de `generar_stock_historico` y de `main` que dependen del retorno `int` y del orden; el fixture `corrida` hace que la función falsa devuelva un `DataFrame`)
- Test: `tests/test_variabilidad.py`

**Interfaces:**
- Consumes: `calcular_variabilidad`, `resumen_variabilidad`, `COLUMNAS_VARIABILIDAD` (Tasks 2–3).
- Produces: `generar_stock_historico(conn_pg, batch_id, inicio, cfg) -> pd.DataFrame | None`: la serie guardada, o `None` si no hay ancla (antes devolvía `int`; ruling 1).

- [ ] **Step 1: Write the failing tests** (usar el fixture `corrida` de `tests/test_stock_historico.py`, importándolo en `tests/test_variabilidad.py` o moviéndolo a `tests/conftest.py` si hace falta compartirlo):
  - `test_main_calcula_la_variabilidad_entre_historico_y_metricas`: el orden de eventos es `historico` < `variabilidad` < `metricas` < `cerrar ok`, y la serie que devuelve el histórico es la que recibe `calcular_variabilidad`.
  - `test_main_pasa_las_ocho_medidas_a_guardar_metricas`: las métricas que recibe `guardar_metricas` incluyen las ocho claves del SKU.
  - `test_main_sin_serie_de_stock_guarda_nulls_y_cierra_ok`: histórico devuelve `None` → métricas con las ocho en `None`, `("cerrar","ok")`.
  - `test_main_excepcion_en_variabilidad_no_cambia_el_estado`: `calcular_variabilidad` lanza → `("cerrar","ok")`, no `"error"`, y `guardar_metricas` igual se llama.
  - `test_main_excepcion_en_historico_sigue_guardando_metricas`: el histórico lanza → `rollback` antes de las métricas, métricas con ocho `None`, cierre `ok` (reemplaza la expectativa de orden de `test_main_excepcion_en_historico_no_cambia_el_estado`).
  - `test_main_con_reconstruir_stock_false_y_desde_csv_guardan_nulls`: sin histórico, métricas con las ocho en `None`.
  - Ajustar `test_generar_encadena_los_pasos` (`== 2` → `len(resultado) == 2`) y `test_generar_sin_ancla_no_guarda_y_no_falla` (`== 0` → `is None`); `test_main_reconstruye_entre_metricas_y_cierre` pasa a `test_main_reconstruye_antes_de_las_metricas` (orden `historico` < `metricas` < `cerrar`).
- [ ] **Step 2: Run** `pytest tests/test_variabilidad.py tests/test_stock_historico.py -v` — Expected: FAIL en los tests nuevos y en los ajustados.
- [ ] **Step 3: Implement.** `generar_stock_historico` devuelve `serie` (o `None` sin ancla). En `main`: mover el bloque del histórico (con su `try/except` y su `rollback`/reconexión intactos) para que corra antes de `guardar_metricas`, asignando `serie_stock = None` por defecto y cuando falla o no aplica; después, `try: variab = calcular_variabilidad(datos, serie_stock, cfg)` y sumar a cada `r["metricas"]` las ocho claves de `variab[kopr]` (si falla: `log.exception("Variabilidad falló; las métricas quedan sin ella")` y no se suma nada); llamar `resumen_variabilidad` sobre las métricas y luego `guardar_metricas`. El bloque no toca la conexión PostgreSQL.
- [ ] **Step 4: Run** `pytest -q` — Expected: toda la suite en verde (hoy 98 pasan, 1 omitido, más los tests nuevos).
- [ ] **Step 5: Commit** `feat: la corrida calcula y guarda la variabilidad del inventario`.

---

### Task 5: Verificación contra datos reales (requiere confirmación)

**Files:** ninguno.

Esta tarea corre el job contra Softland y escribe una corrida nueva en la BD dev; **se ejecuta solo con visto bueno del usuario** y con las credenciales de Softland en `.env`. Lo que verifica es el bloque "Criterios de aceptación" del spec.

- [ ] **Step 1: Antes de correr, anotar de la última corrida `ok`** el `mape_semanal`, `mape_4sem` por SKU y `count(*)` de `serie_semanal`.
- [ ] **Step 2: Correr** `python batch_prediccion_jnc.py` — Expected: termina con código 0, aparece la línea `Variabilidad (52 semanas): …` y ninguna advertencia nueva.
- [ ] **Step 3: Consultar la corrida nueva** y comprobar: 21 o 22 SKUs con las ocho columnas pobladas; `ratio_variabilidad` = `round(cv_stock/cv_demanda, 4)` en todas las filas; `semanas_variabilidad = 52` donde hay historia completa y las ocho en NULL donde quedó bajo 40; `tasa_quiebre` entre 0 y 1; `prop_stock_reconstruido` cerca de 1; rangos de referencia del spec (`cv_demanda` 0,30–0,67; `cv_stock` 0,49–1,02; ratio 0,91–1,83 con mediana ≈ 1,35; quiebre medio ≈ 6 %). Un ratio en decenas o centenas indica cociente de varianzas.
- [ ] **Step 4: Comparar con la corrida anterior**: MAPE semanal, MAPE a 4 semanas y filas de `serie_semanal` idénticos. Si algo difiere, se detiene el despliegue.

---

## Self-review

- **Cobertura del spec (sección 3 y criterios):** config → Task 1; función pura, join por semana, ocho medidas, casos borde, mínimo de 40 → Task 2; persistencia y línea de log sin advertencias → Task 3; posición en `main`, sin serie de stock, excepción que no cambia el estado → Task 4; criterios de aceptación y rangos de referencia → Task 5. "Lo que no cambia" está en Global Constraints y se comprueba en Task 5 (Step 4) y con la suite completa en Task 4. Lo que el spec deja fuera (exposición, agregado de portafolio, umbrales) no tiene tarea.
- **Consistencia de nombres:** `COLUMNAS_VARIABILIDAD`, `calcular_variabilidad`, `resumen_variabilidad` y `generar_stock_historico` se escriben igual en Interfaces, tests y pasos.
- **Proporción:** código solo en firmas y valores del spec; los cuerpos los decide quien implementa.
