# Métricas de variabilidad — Migración 009 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar a `predictions.metrica_sku` las ocho columnas nullable donde el job guardará la variabilidad del stock y de la demanda por SKU.

**Architecture:** Una migración SQL idempotente (`ALTER TABLE … ADD COLUMN IF NOT EXISTS` + `COMMENT ON COLUMN`), mismo patrón que `006` y `008`. Un test de forma lee el archivo y asierta su contenido, igual que `tests/test_compras_salidas.py`.

**Tech Stack:** PostgreSQL, pytest.

**Spec:** Pegado en la conversación del 2026-11-02 ("Spec — métricas de variabilidad del inventario"); no existe como archivo en el repo. Este plan cubre solo la sección 2 (Base de datos). La sección 3 (función `calcular_variabilidad`, `Config`, integración en `main`, log) es un plan aparte, posterior.

## Global Constraints

- Migración `sql/009_metricas_variabilidad.sql`, correr después de 008, idempotente (debe correr dos veces sin error).
- Columnas **nullable**, sin `NOT NULL` ni `DEFAULT`; no se tocan vistas; no se crea tabla.
- `COMMENT ON COLUMN` en cada una de las ocho columnas.
- El comentario de `ratio_variabilidad` debe decir que es el cociente de coeficientes de variación y **no** Var(stock)/Var(demanda).
- Tipos exactos del spec:
  `cv_demanda NUMERIC(8,4)`, `cv_stock NUMERIC(8,4)`, `ratio_variabilidad NUMERIC(8,4)`, `media_demanda NUMERIC(14,2)`, `media_stock NUMERIC(14,2)`, `tasa_quiebre NUMERIC(6,4)`, `semanas_variabilidad SMALLINT`, `prop_stock_reconstruido NUMERIC(6,4)`.
- Cabecera y estilo del archivo como `008_compras_salidas.sql` (bloque de comentario, `BEGIN;`/`COMMIT;`, acentos en comentarios, SQL sin acentos en los `COMMENT`).
- El modelo, el MAPE y `serie_semanal` no se tocan.

## Review Focus

1. Re-ejecución: el script aplicado dos veces seguidas no falla (Task 1, verificación contra Postgres si hay uno disponible; si no, el test asierta `IF NOT EXISTS` en las ocho columnas).
2. Orden de despliegue: aplicar antes del job nuevo, o el `INSERT` en `metrica_sku` fallará; queda documentado en la cabecera del archivo (Task 1).
3. `NULL` significa "no calculable" (menos de 40 semanas, sin serie de stock), no cero; el comentario de cada columna lo dice (Task 1).
4. Lectores de `metrica_sku` existentes (`v_cobertura_actual`, etc.) siguen intactos: el test asierta que no hay `VIEW` ni `CREATE TABLE` (Task 1).

---

## File Structure

- `sql/009_metricas_variabilidad.sql` — la migración (crear).
- `tests/test_variabilidad.py` — test de forma de la migración; los tests de la función `calcular_variabilidad` se agregarán aquí en el plan del job (crear).

---

### Task 1: Migración 009

**Files:**
- Create: `sql/009_metricas_variabilidad.sql`
- Test: `tests/test_variabilidad.py`

**Interfaces:**
- Consumes: tabla `predictions.metrica_sku` (migración 001, extendida por 003 y 006).
- Produces: columnas `cv_demanda`, `cv_stock`, `ratio_variabilidad`, `media_demanda`, `media_stock`, `tasa_quiebre`, `semanas_variabilidad`, `prop_stock_reconstruido` en `predictions.metrica_sku`, que el `INSERT` del job llenará en el plan siguiente.

- [ ] **Step 1: Write the failing test** in `tests/test_variabilidad.py`

Mismo arranque que `tests/test_compras_salidas.py` (`sys.path.insert`, `RAIZ`). Test `test_migracion_009_forma` que lee `sql/009_metricas_variabilidad.sql` y asierta:
  - cada una de las ocho líneas `ADD COLUMN IF NOT EXISTS <col> <tipo>` con los tipos exactos de Global Constraints;
  - `sql.count("COMMENT ON COLUMN predictions.metrica_sku.") == 8`;
  - `"NOT NULL" not in sql` y `"DEFAULT" not in sql.upper()`;
  - `"VIEW" not in sql.upper()` y `"CREATE TABLE" not in sql.upper()`;
  - el texto del comentario de `ratio_variabilidad` contiene `"coeficientes de variacion"` y `"NO es Var(stock)/Var(demanda)"`;
  - `sql.strip().upper().rstrip(";").endswith("COMMIT")`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_variabilidad.py -v`
Expected: FAIL con `FileNotFoundError` sobre `009_metricas_variabilidad.sql`.

- [ ] **Step 3: Write `sql/009_metricas_variabilidad.sql`**

Cabecera al estilo de 008 (motor, fecha 02-11-2026, para qué sirve, nullable a propósito, orden de despliegue ANTES del job nuevo, "Correr después de 008. Idempotente. No toca ninguna vista."). Un solo `ALTER TABLE predictions.metrica_sku` con las ocho `ADD COLUMN IF NOT EXISTS` del spec, y ocho `COMMENT ON COLUMN`, todo entre `BEGIN;` y `COMMIT;`. Cada comentario dice qué mide y cuándo es NULL (ventana de 52 semanas cerradas; menos de 40 semanas con dato → NULL; media ≤ 0 → ese CV NULL; `cv_demanda` = 0 → ratio NULL). Específicos:
  - `cv_demanda`: desviación estándar / media de la demanda **cruda** (antes de winsorizar).
  - `ratio_variabilidad`: `cv_stock / cv_demanda`, cociente de coeficientes de variacion; NO es Var(stock)/Var(demanda).
  - `media_demanda`, `media_stock`: sirven para reconstruir el cociente de varianzas crudas.
  - `tasa_quiebre`: proporción de semanas con `stock_fin <= 0` (medición directa, no la marca `censurada`).
  - `semanas_variabilidad`: semanas efectivamente usadas.
  - `prop_stock_reconstruido`: proporción de la ventana con `origen = 'reconstruido'`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_variabilidad.py -v`
Expected: PASS.

- [ ] **Step 5: Verificar idempotencia contra Postgres (si hay uno disponible)**

Run (dos veces seguidas, sobre una base con las migraciones 001–008 aplicadas): `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/009_metricas_variabilidad.sql`
Expected: ambas corridas terminan sin error; `\d predictions.metrica_sku` muestra las ocho columnas. Si no hay base disponible, se informa que este paso no se ejecutó.

- [ ] **Step 6: Commit**

```bash
git add sql/009_metricas_variabilidad.sql tests/test_variabilidad.py docs/superpowers/plans/2026-11-02-metricas-variabilidad-migracion.md
git commit -m "feat: migración 009 con métricas de variabilidad en metrica_sku"
```

---

## Self-review

- Cobertura del spec (sección 2): ocho columnas con tipos, `COMMENT` en cada una con la aclaración de `ratio_variabilidad`, nullable, sin vistas, idempotente → Task 1. Sección 3 y criterios de aceptación sobre datos reales quedan fuera de este plan, explícitamente.
- Consistencia de nombres: las ocho columnas se escriben igual en Global Constraints, Interfaces, test y migración.
- Proporción: plan corto, una tarea; acorde a una migración de ~40 líneas.
