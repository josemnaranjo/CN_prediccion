# Reconstrucción del inventario histórico (job + base de datos) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el job reconstruya hacia atrás el stock semanal por SKU (bodega B01) desde una captura ancla de `stock_snapshot`, lo complete con los cierres reales posteriores y lo guarde en `predictions.stock_historico`, sin tocar el modelo ni sus métricas.

**Architecture:** Producto paralelo a la predicción. Una consulta aparte a Softland trae el flujo neto semanal; una función pura camina hacia atrás desde el ancla (`stock(t) = stock(t+1) − flujo(t+1)`); un bloque en `main`, envuelto en `try/except`, orquesta ancla → movimientos → reconstrucción → cierres reales → guardado. Todo en `batch_prediccion_jnc.py`, que ya concentra las funciones del job.

**Tech Stack:** Python 3.12, pandas, psycopg2, pymssql, pytest, PostgreSQL, SQL Server (Softland, solo lectura).

**Spec:** Pegado en la conversación del 2026-10-22 ("Spec — reconstrucción del inventario histórico en el job"); no existe como archivo en el repo. Este plan copia los valores exactos que necesita.

## Global Constraints

- Modelo, target, entrenamiento, métricas y `serie_semanal`: **no se tocan**. `_query_ventas` y `leer_ventas_mssql` **no se modifican**.
- Cantidad: `CAPRCO1`. **No usar `CAPRCO2`.**
- Bodega: filtro explícito `BOSULIDO = cfg.bodega` (B01). Solo B01.
- Lunes ISO en SQL: `DATEADD(DAY, -(DATEDIFF(DAY, '19000101', FEEMDO) % 7), CONVERT(DATE, FEEMDO))`.
- Tipos de documento en `Config`, no incrustados en el SQL: `tido_entrada=("FCC","GRI","GAR")` (GAR ya viene con signo), `tido_salida=("FCV","FDV","GDI")`, `tido_entrada_neg=("NCC",)`, `tido_salida_neg=("NCV",)`.
- `reconstruir_stock: bool = True`; `dias_minimos_ancla: int = 10`. Con `reconstruir_stock = False` el job se comporta exactamente como hoy.
- **No recortar negativos a cero.** Grilla completa: una fila por SKU y semana, flujo 0 donde no hubo movimiento.
- Una captura tomada a las 00:0x del día D refleja el cierre del día D−1; la que cierra la semana ISO es la del **lunes** de madrugada. Se calcula, no se asume.
- Nada de la reconstrucción puede tumbar la corrida ni cambiar su estado.
- Referencia de calidad: 132 de 3.608 semanas negativas (3,7 %), concentradas en HOLANDA BOCADITOS (25) y TABLETA BOB (23).

## Rulings previos a la ejecución (huecos del spec; costo si están mal: bajo)

1. **`batch_run_id BIGINT` + `ON DELETE CASCADE`** en la migración 007, no `INTEGER` sin cascade como dice el DDL del spec: `batch_run.id` es `BIGSERIAL` y todas las tablas hijas del repo (`stock_snapshot`, `serie_semanal`) usan `BIGINT … ON DELETE CASCADE`. El resto del DDL es el del spec.
2. **Config extra** (el spec no la lista): `zona_horaria = "America/Santiago"` y `hora_madrugada_max = 6`. `capturado_en` es `TIMESTAMPTZ`; "lunes de madrugada" solo tiene sentido en hora local, y "madrugada" necesita un umbral. Las capturas de media tarde (corridas manuales) quedan fuera.
3. **Grilla desde `inicio`**: el primer lunes de la serie de ventas de la corrida (`inicio` de `main`), para que el panel alinee con la demanda (~163 semanas). Los movimientos se leen desde ese lunes.
4. **`flujo_neto` de las semanas `capturado` posteriores al ancla** = diferencia contra el cierre de la semana anterior de la serie; 0 si esa semana no existe. La ancla conserva el flujo leído de Softland. (La ventana de movimientos termina en el ancla, así que no hay otro flujo disponible.)
5. **Semanas previas al ancla con captura real** (30-sep-2026 en adelante) igual se marcan `reconstruido`, porque el criterio de aceptación pide "anteriores = reconstruido". Solo la ancla y las posteriores son `capturado`.
6. **Firmas con parámetros extra**: `leer_movimientos_stock(conn, cfg, desde, semana_ancla)`, `reconstruir_stock_historico(ancla, movimientos, cfg, inicio)` y `elegir_ancla(conn_pg, cfg, ahora=None)`. El spec da la firma abreviada; falta la ventana, el inicio de grilla y un reloj inyectable para testear.
7. **Punto ambiguo de la prueba de regresión**: el spec fija "SELZ QUESO, cierre del 11 de octubre, ±50", pero con ancla en ese cierre el punto es exacto por construcción. La Task 7 asierta ese ±50 sobre el cierre del **4 de octubre** (el único punto no exacto medido en ese tramo). **Confirmar con el usuario** si se refería a otra fecha.

## Review Focus

1. Un SKU sin fila en la captura ancla: se omite con warning, no aborta ni inventa un stock (Task 4).
2. Semanas sin ningún movimiento en todo el SKU (SKU nuevo o sin documentos): sale la grilla completa con stock constante, sin hoyos (Task 4).
3. Dos capturas el mismo lunes (00:05 y 15:30): gana la de madrugada; la de la tarde no cierra nada (Task 3).
4. Captura de domingo con sábado movido: no cierra semana; solo el lunes (Task 3).
5. Falla de Postgres a mitad del bloque (transacción abortada): se hace `rollback()` antes de `cerrar_corrida`, o el cierre `ok` también falla (Task 6).

---

## File Structure

- `sql/007_stock_historico.sql` — nueva tabla (Task 1).
- `batch_prediccion_jnc.py` — `Config` + funciones nuevas en una sección "Inventario histórico" colocada después de `guardar_stock`/antes de `armar_filas_serie`, salvo lectura de Softland (junto a `leer_stock_mssql`) y el bloque en `main` (Tasks 1–6).
- `tests/test_stock_historico.py` — tests unitarios de las Tasks 1–6 (pytest, mismo estilo que `tests/test_bonificacion.py`: `import batch_prediccion_jnc as b`, `Config`, `_FakeConn`).
- `tests/test_stock_historico_regresion.py` — prueba de regresión contra datos reales, se salta sin credenciales (Task 7).

---

### Task 1: Config y migración 007

**Files:**
- Modify: `batch_prediccion_jnc.py` (`Config`, bloque "Stock", y `__post_init__`)
- Create: `sql/007_stock_historico.sql`
- Test: `tests/test_stock_historico.py`

**Interfaces:**
- Produces: `Config.reconstruir_stock: bool`, `dias_minimos_ancla: int`, `tido_entrada`, `tido_salida`, `tido_entrada_neg`, `tido_salida_neg: tuple[str, ...]`, `zona_horaria: str`, `hora_madrugada_max: int`.

- [ ] **Step 1: Write the failing tests** en `tests/test_stock_historico.py`
  - `test_config_defaults_stock_historico`: `Config()` tiene `reconstruir_stock is True`, `dias_minimos_ancla == 10`, `tido_entrada == ("FCC","GRI","GAR")`, `tido_salida == ("FCV","FDV","GDI")`, `tido_entrada_neg == ("NCC",)`, `tido_salida_neg == ("NCV",)`, `zona_horaria == "America/Santiago"`, `hora_madrugada_max == 6`.
  - `test_config_tido_stock_no_se_solapan`: `Config(tido_entrada=("FCC","FCV"))` lanza `ValueError` (FCV ya está en salida). Los cuatro tuples deben ser disjuntos entre sí (solapar con `TIPOS_VENTA` es legítimo y no se valida).
  - `test_migracion_007_forma`: leer `sql/007_stock_historico.sql`; contiene `CREATE TABLE IF NOT EXISTS predictions.stock_historico`, `PRIMARY KEY (batch_run_id, kopr, bodega, semana)`, `EXTRACT(ISODOW FROM semana) = 1`, `origen IN ('reconstruido', 'capturado')`, `REFERENCES predictions.batch_run(id)`; no contiene `INTEGER`.
- [ ] **Step 2: Run** `pytest tests/test_stock_historico.py -v` → FAIL (atributos y archivo inexistentes).
- [ ] **Step 3: Add the Config fields** con los defaults de Global Constraints, comentados en el estilo de los bloques existentes, más la validación de disjunción en `__post_init__` (mensaje en español como las otras).
- [ ] **Step 4: Write `sql/007_stock_historico.sql`**: encabezado con el estilo de 004/006 (motor, fecha 22-10-2026, por qué, "Correr después de 006. Idempotente", y "Orden de despliegue: aplicar ANTES de desplegar el job nuevo; si no, el bloque falla, se registra y la corrida sigue sin serie"). DDL del spec con los ajustes del Ruling 1, `COMMENT ON COLUMN … origen` explicando los dos valores, e índice `(kopr, semana)`. Terminar en `COMMIT` solo si 006 abre `BEGIN` (copiar su convención).
- [ ] **Step 5: Run** `pytest tests/test_stock_historico.py -v` → PASS.
- [ ] **Step 6: Commit** `feat: config y migración 007 de stock histórico`

---

### Task 2: Lectura de movimientos desde Softland

**Files:**
- Modify: `batch_prediccion_jnc.py` (después de `leer_stock_mssql`)
- Test: `tests/test_stock_historico.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `_lista_sql`, `lunes_de`.
- Produces:
  - `signos_tido(cfg: Config) -> dict[str, int]` (entrada y salida_neg → `+1`; salida y entrada_neg → `−1`).
  - `_query_movimientos(cfg: Config, desde: date, hasta_excl: date) -> str`
  - `leer_movimientos_stock(conn, cfg: Config, desde: date, semana_ancla: date) -> pd.DataFrame` con columnas `kopr: str`, `semana: date` (lunes), `flujo_neto: float`. `hasta_excl = semana_ancla + 7 días`.

- [ ] **Step 1: Write the failing tests**
  - `test_signos_tido`: `signos_tido(Config()) == {"FCC":1,"GRI":1,"GAR":1,"NCV":1,"FCV":-1,"FDV":-1,"GDI":-1,"NCC":-1}`.
  - `test_query_movimientos_elementos_obligatorios`: el SQL contiene `BOSULIDO = 'B01'`, `CAPRCO1`, `'19000101'`, `% 7`, los ocho TIDO, `FEEMDO >= '2026-01-05'` y `FEEMDO < '2026-10-12'` para `desde=date(2026,1,5)`, `semana_ancla=date(2026,10,5)`; **no** contiene `CAPRCO2`.
  - `test_leer_movimientos_stock_aplica_signo_y_tipos`: monkeypatch `pd.read_sql` para devolver `kopr` con espacios, `semana` como `Timestamp`, `flujo_neto` como `Decimal`; el resultado tiene `kopr` sin espacios, `semana` como `date` y `flujo_neto` float.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** el SQL con un solo `SUM(CASE …)` generado desde `signos_tido` (un `WHEN TIDO IN (+) THEN CAPRCO1 WHEN TIDO IN (−) THEN -CAPRCO1 END`), `JOIN MAEEDO/MAEDDO` con `WITH (NOLOCK)` como `_query_ventas`, `WHERE` con los ocho TIDO, bodega, SKUs de `cfg.skus` y la ventana; `GROUP BY` kopr y el lunes. Docstring que explique por qué NO se agrega a `_query_ventas` (frase del spec: reconciliación unidad por unidad). Fechas como literal ISO `'YYYY-MM-DD'`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: leer movimientos de stock desde Softland`

---

### Task 3: Cierres semanales reales y elección del ancla

**Files:**
- Modify: `batch_prediccion_jnc.py` (sección nueva "Inventario histórico")
- Test: `tests/test_stock_historico.py`

**Interfaces:**
- Consumes: `Config`.
- Produces:
  - `@dataclass(frozen=True) class Ancla: semana: date; capturado_en: datetime; antiguedad_dias: int; stock: dict[str, float]` (`semana` = lunes de la semana que la captura cierra; `stock` = kopr → stock al cierre).
  - `cierres_semanales(capturas: pd.DataFrame, cfg: Config) -> pd.DataFrame`. Entrada: columnas `kopr, stock_fisico, capturado_en` (tz-aware). Salida: `kopr, semana, stock_fin, capturado_en`, una fila por (kopr, semana).
  - `elegir_ancla_de(cierres: pd.DataFrame, cfg: Config, ahora: datetime) -> Ancla | None` (pura; loguea la razón cuando devuelve `None`).
  - `leer_capturas(conn_pg, cfg: Config) -> pd.DataFrame` (lee `predictions.stock_snapshot` filtrando `bodega = cfg.bodega`; devuelve `kopr, stock_fisico, capturado_en`).
  - `elegir_ancla(conn_pg, cfg: Config, ahora: datetime | None = None) -> Ancla | None` = `leer_capturas` → `cierres_semanales` → `elegir_ancla_de`; `ahora` por defecto `datetime.now(ZoneInfo(cfg.zona_horaria))`.

- [ ] **Step 1: Write the failing tests** (capturas construidas con `pd.Timestamp("2026-10-12 00:05", tz="America/Santiago")`, un SKU `A`)
  - `test_cierre_es_el_lunes_de_madrugada`: capturas `2026-10-11 00:05` (domingo) y `2026-10-12 00:05` (lunes) → una sola fila, `semana == date(2026,10,5)`, valor del lunes.
  - `test_captura_de_la_tarde_no_cierra_nada`: lunes 12 a las 00:05 (valor 100) y a las 15:30 (valor 999) → `stock_fin == 100`; un lunes solo con captura de las 15:30 → no aparece.
  - `test_varias_madrugadas_gana_la_mas_temprana`: 00:05 (100) y 00:40 (110) el mismo lunes → 100.
  - `test_ancla_exige_antiguedad_minima`: cierres con capturas del lunes 12 y del lunes 19 de octubre, `ahora = 2026-10-22 12:00` → ancla `semana == date(2026,10,5)`, `antiguedad_dias == 10`; con `ahora = 2026-10-21 12:00` → `None` (9 días).
  - `test_ancla_none_registra_la_razon`: sin cierres elegibles → `None` y `caplog` contiene "no se reconstruye".
  - `test_ancla_trae_el_stock_de_todos_los_skus`: dos SKUs en la misma captura → `ancla.stock == {"A": 100.0, "B": 50.0}`.
  - `test_elegir_ancla_lee_postgres`: `conn_pg` falso cuyo cursor devuelve filas; verifica que el SQL filtra por `bodega` y consulta `stock_snapshot`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** `cierres_semanales`: convertir `capturado_en` a `cfg.zona_horaria`; conservar `weekday()==0` y `hour < cfg.hora_madrugada_max`; por (kopr, fecha local) la de menor hora; `semana = fecha − 7 días`. `elegir_ancla_de`: filtrar `capturado_en <= ahora − dias_minimos_ancla`; tomar la `semana` máxima; `antiguedad_dias = (ahora − capturado_en).days`. Log INFO con la semana, la `capturado_en` y la antigüedad cuando hay ancla (criterio de aceptación).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: elegir ancla de stock con cierre semanal calculado`

---

### Task 4: Reconstrucción pura y control de calidad

**Files:**
- Modify: `batch_prediccion_jnc.py` (sección "Inventario histórico")
- Test: `tests/test_stock_historico.py`

**Interfaces:**
- Consumes: `Ancla` (Task 3), `rango_semanas`, `movimientos` con el esquema de Task 2.
- Produces:
  - `reconstruir_stock_historico(ancla: Ancla, movimientos: pd.DataFrame, cfg: Config, inicio: date) -> pd.DataFrame` con `kopr, semana, stock_fin, flujo_neto`; una fila por SKU de `ancla.stock` y lunes de `inicio` a `ancla.semana`.
  - `calidad_stock_historico(df: pd.DataFrame) -> dict` y registra el control en el log (INFO por SKU con negativas: cantidad, peor valor, fecha de la última; INFO del total con porcentaje; WARNING si el porcentaje queda fuera de 3 %–5 %). Devuelve `{"semanas": int, "negativas": int, "pct": float}`.

- [ ] **Step 1: Write the failing tests**
  - `test_reconstruccion_camina_hacia_atras`: ancla `semana=2026-10-05`, `stock={"A": 100.0}`; movimientos `A`: 2026-10-05 → `+30`, 2026-09-28 → `−20`; `inicio=2026-09-21` → `stock_fin` por semana `[ (09-21, 90.0), (09-28, 70.0)...` calcular a mano: fin(10-05)=100; fin(09-28)=100−30=70; fin(09-21)=70−(−20)=90. Verificar los tres valores y que `flujo_neto` de cada semana es el leído.
  - `test_reconstruccion_grilla_completa_sin_hoyos`: movimientos vacíos para `A` → una fila por lunes entre `inicio` y ancla, todas con `flujo_neto == 0` y `stock_fin == 100`.
  - `test_reconstruccion_no_recorta_negativos`: ancla `stock={"A": 10.0}`, flujo `+50` en la ancla → `stock_fin` de la semana previa == `-40.0`.
  - `test_reconstruccion_omite_sku_sin_captura_y_avisa`: movimientos de `B` sin `B` en `ancla.stock` → sin filas de `B`, `caplog` con warning que nombra `B`. (Review Focus 1)
  - `test_reconstruccion_ignora_movimientos_posteriores_al_ancla`: un movimiento en la semana siguiente a la ancla no altera ningún valor.
  - `test_calidad_cuenta_negativas`: DataFrame con 2 SKUs y 10 semanas, 1 negativa → `pct == 5.0`; el log contiene el SKU, el valor peor y la fecha de la última.
  - `test_calidad_avisa_fuera_de_rango`: 0 negativas → warning; 4 % → sin warning.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** con grilla por producto cartesiano SKU × `rango_semanas(inicio, ancla.semana)`, `merge` del flujo (relleno 0) y `stock_fin` por SKU como `ancla − (suma acumulada inversa del flujo posterior)`: `fin(t) = ancla_stock − Σ flujo(s)` para `s` en `(t, ancla.semana]`. Sin `clip`. El porcentaje se calcula sobre todas las filas.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: reconstruir stock semanal hacia atrás con control de calidad`

---

### Task 5: Armado de la serie final y persistencia

**Files:**
- Modify: `batch_prediccion_jnc.py` (después de `guardar_stock`)
- Test: `tests/test_stock_historico.py`

**Interfaces:**
- Consumes: `reconstruir_stock_historico` (Task 4), `cierres_semanales` (Task 3), `Ancla`.
- Produces:
  - `armar_serie_historica(reconstruido: pd.DataFrame, cierres: pd.DataFrame, ancla: Ancla) -> pd.DataFrame` con `kopr, semana, stock_fin, flujo_neto, origen`: la ancla → `capturado` (stock = captura, flujo = el leído); anteriores → `reconstruido`; posteriores → `capturado` desde `cierres` (`semana > ancla.semana`), con `flujo_neto` según Ruling 4.
  - `guardar_stock_historico(conn, batch_id: int, bodega: str, serie: pd.DataFrame) -> None` (`execute_values`, `page_size=1000`, `commit`; no hace nada con DataFrame vacío; redondea a 2 decimales).

- [ ] **Step 1: Write the failing tests**
  - `test_armar_serie_marca_origen`: reconstruido 3 semanas (09-21, 09-28, 10-05=ancla) + cierres con `semana` 10-12 y 10-19 → orígenes `[reconstruido, reconstruido, capturado, capturado, capturado]`; en 10-05 `stock_fin` es el de la ancla.
  - `test_armar_serie_flujo_de_capturadas_es_la_diferencia`: ancla 100, cierre 10-12 = 130, cierre 10-19 = 125 → `flujo_neto` 30 y −5.
  - `test_armar_serie_cierre_sin_semana_previa_flujo_cero`: cierre en 10-26 con 10-19 ausente → `flujo_neto == 0`.
  - `test_armar_serie_sin_semanas_repetidas_ni_cierres_previos`: un cierre con `semana <= ancla.semana` no entra ni duplica.
  - `test_armar_serie_todas_las_semanas_son_lunes`: `all(s.weekday()==0)`.
  - `test_guardar_stock_historico`: con `_FakeConn` y `execute_values` monkeypatcheado (como en `test_guardar_metricas_persiste_el_factor`): el SQL menciona `predictions.stock_historico`, la fila es `(batch_id, kopr, "B01", semana, stock, flujo, origen)` y hay `commit`; con DataFrame vacío no hay `commit` ni `execute`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** ambas funciones; ordenar la salida por `kopr, semana`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: armar y guardar la serie histórica de stock`

---

### Task 6: Orquestación en `main`

**Files:**
- Modify: `batch_prediccion_jnc.py` (`main`, función nueva antes de `main`)
- Test: `tests/test_stock_historico.py`

**Interfaces:**
- Consumes: todo lo anterior, `conectar_mssql`, `abrir_corrida`/`cerrar_corrida`.
- Produces: `generar_stock_historico(conn_pg, batch_id: int, inicio: date, cfg: Config) -> int` (devuelve filas guardadas; devuelve 0 y loguea si no hay ancla). Pasos: `elegir_ancla` → conexión nueva `conectar_mssql()` → `leer_movimientos_stock(…, inicio, ancla.semana)` → `reconstruir_stock_historico` → `calidad_stock_historico` → `cierres_semanales(leer_capturas(…))` → `armar_serie_historica` → `guardar_stock_historico`.

- [ ] **Step 1: Write the failing tests**
  - `test_generar_sin_ancla_no_guarda_y_no_falla` (monkeypatch `elegir_ancla` → `None`): devuelve 0, no llama `leer_movimientos_stock` ni `guardar_stock_historico`.
  - `test_generar_encadena_los_pasos` (monkeypatch de las funciones con dobles): verifica que `leer_movimientos_stock` recibe `desde=inicio` y `semana_ancla=ancla.semana`, y que `guardar_stock_historico` recibe el `batch_id` y `cfg.bodega`.
  - `test_main_con_reconstruir_stock_false_no_llama_nada` y `test_main_excepcion_en_historico_no_cambia_el_estado`: monkeypatch de lecturas, modelado y `guardar_*` para correr `main([])` completo con dobles; en el segundo, `generar_stock_historico` lanza `RuntimeError` → `cerrar_corrida` se invoca con `"ok"`, `conn.rollback()` fue llamado **antes** de `cerrar_corrida`, retorno `0`. (Review Focus 5)
  - `test_main_desde_csv_no_reconstruye`: con `--desde-csv` se loguea el motivo y no se llama `generar_stock_historico`.
  - `test_main_dry_run_no_reconstruye`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** En `main`, **después de `guardar_metricas` y antes de `cerrar_corrida(... "ok" ...)`**, dentro del `try` existente pero con su propio `try/except Exception` interno: `log.exception`, `conn.rollback()`, continuar. Condición: `cfg.reconstruir_stock and not args.csv`; en `--desde-csv` loguear que no hay Softland. El bloque no altera `ok`/`fallidos`. Extraer `inicio` de lo ya calculado en `main`.
- [ ] **Step 4: Run** `pytest -v` completo → PASS (incluye `test_backtest.py` y `test_bonificacion.py` sin cambios).
- [ ] **Step 5: Commit** `feat: orquestar la reconstrucción de stock sin tumbar la corrida`

---

### Task 7: Regresión contra datos reales y verificación de aceptación

**Files:**
- Create: `tests/test_stock_historico_regresion.py`

**Interfaces:**
- Consumes: variables `M32QUILLOTA_DATABASE_URL` y `POSTGRES_DATABASE_URL`; `conectar_mssql`, `conectar_postgres`, `leer_capturas`, `cierres_semanales`, `leer_movimientos_stock`, `reconstruir_stock_historico`, `Ancla`.

- [ ] **Step 1: Write the test.** `pytest.mark.skipif` si faltan las dos variables. Construir un `Ancla` con `semana=date(2026,10,5)` (cierre del 11 de octubre) y `stock` tomado del cierre de `stock_snapshot` correspondiente; reconstruir desde `date(2026,1,5)`; comparar `stock_fin` de la semana `2026-09-28` (cierre del 4 de octubre) contra el cierre real de `cierres_semanales`. Asertar: ALKA MENTOL `abs(error) <= 30`; SELZ QUESO `abs(error) <= 50` (ver Ruling 7); mediana del error absoluto de los 22 SKUs `< 15`.
- [ ] **Step 2: Run** `pytest tests/test_stock_historico_regresion.py -v` con credenciales → PASS. Sin credenciales → SKIPPED (verificar que no rompe la suite).
- [ ] **Step 3: Confirmar que el modelo no se movió.** Con credenciales, correr `python batch_prediccion_jnc.py --dry-run` en `main` y en esta rama; comparar las líneas de MAPE semanal, MAPE a 4 semanas y `Historia: … (N semanas, … SKUs)`. Deben ser idénticas. Complemento: `git diff main -- batch_prediccion_jnc.py` solo agrega código (ninguna línea de `_query_ventas`, `procesar_sku`, `backtest_arima`, `armar_filas_serie` cambia).
- [ ] **Step 4: Verificar la base (solo tras aplicar `007` en dev y confirmarlo con el usuario).** Corrida real y consultas: 22 SKUs × ~163 semanas sin repetidas (`GROUP BY kopr, semana HAVING COUNT(*)>1` vacío); `bool_and(EXTRACT(ISODOW FROM semana)=1)`; en la semana ancla `stock_fin` = captura y `origen='capturado'`; anteriores `reconstruido`, posteriores `capturado`; porcentaje de negativas entre 3 % y 5 %, con HOLANDA BOCADITOS y TABLETA BOB a la cabeza; el log nombra la captura ancla y su antigüedad (≥ 10 días); `batch_run.status = 'ok'`.
- [ ] **Step 5: Commit** `test: regresión del stock reconstruido contra capturas reales`

---

## Self-review

- **Cobertura del spec:** §1 → T1; §2 → T2; §3 → T3; §4 → T4; §5 → T1 + T5; §6 → T6; criterios de aceptación → T4–T7; regresión → T7; "fuera de alcance" → ninguna tarea toca modelo, métricas de amplificación, otras bodegas, backend/frontend.
- **Consistencia de tipos:** `Ancla.semana/stock`, `cierres_semanales` (`kopr, semana, stock_fin, capturado_en`) y `reconstruir_stock_historico` (`kopr, semana, stock_fin, flujo_neto`) se usan con los mismos nombres en T3–T6.
- **Proporción:** tareas con firmas y aserciones; sin cuerpos de función salvo la fórmula de reconstrucción.
