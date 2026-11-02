-- ============================================================================
-- Métricas de variabilidad del inventario por SKU
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Fecha    : 02-11-2026
--
-- En cada corrida el job calcula, por SKU y sobre las últimas 52 semanas
-- cerradas, la variabilidad del stock y de la demanda cruda, y las guarda en
-- metrica_sku para observar en el tiempo si el uso del modelo las mejora. Se
-- guardan las piezas (CV de cada lado, medias, tasa de quiebre) además del
-- cociente, porque el cociente solo no dice cuál de los dos lados se movió.
--
-- Nullable a propósito: con menos de 40 semanas con dato, o sin serie de stock
-- en la corrida, las ocho columnas van en NULL. Quien lea la tabla debe tratar
-- NULL como "no calculable", no como cero.
--
-- Orden de despliegue: aplicar ANTES de desplegar el job nuevo; si no, el
-- INSERT en metrica_sku falla y la corrida queda en error.
--
-- Correr después de 008. Idempotente. No toca ninguna vista.
-- ============================================================================

BEGIN;

ALTER TABLE predictions.metrica_sku
  ADD COLUMN IF NOT EXISTS cv_demanda              NUMERIC(8,4),
  ADD COLUMN IF NOT EXISTS cv_stock                NUMERIC(8,4),
  ADD COLUMN IF NOT EXISTS ratio_variabilidad      NUMERIC(8,4),
  ADD COLUMN IF NOT EXISTS media_demanda           NUMERIC(14,2),
  ADD COLUMN IF NOT EXISTS media_stock             NUMERIC(14,2),
  ADD COLUMN IF NOT EXISTS tasa_quiebre            NUMERIC(6,4),
  ADD COLUMN IF NOT EXISTS semanas_variabilidad    SMALLINT,
  ADD COLUMN IF NOT EXISTS prop_stock_reconstruido NUMERIC(6,4);

COMMENT ON COLUMN predictions.metrica_sku.cv_demanda IS
  'Coeficiente de variacion (desviacion estandar / media) de la demanda cruda, antes de winsorizar, en las ultimas 52 semanas cerradas. NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero). Tambien NULL si la media es <= 0.';
COMMENT ON COLUMN predictions.metrica_sku.cv_stock IS
  'Coeficiente de variacion (desviacion estandar / media) del stock al cierre de semana en las ultimas 52 semanas cerradas. Incluye las semanas en quiebre. NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero). Tambien NULL si la media es <= 0.';
COMMENT ON COLUMN predictions.metrica_sku.ratio_variabilidad IS
  'Cociente de coeficientes de variacion: cv_stock / cv_demanda. NO es Var(stock)/Var(demanda): ese cociente de varianzas crudas crece con el cuadrado del nivel y no es interpretable. NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero). Tambien NULL si falta algun CV o cv_demanda es 0.';
COMMENT ON COLUMN predictions.metrica_sku.media_demanda IS
  'Media de la demanda cruda en la ventana de variabilidad. Con media_stock permite reconstruir el cociente de varianzas crudas. NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero).';
COMMENT ON COLUMN predictions.metrica_sku.media_stock IS
  'Media del stock al cierre de semana en la ventana de variabilidad. NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero).';
COMMENT ON COLUMN predictions.metrica_sku.tasa_quiebre IS
  'Proporcion de semanas de la ventana con stock_fin <= 0 (medicion directa, no la marca censurada de la demanda). Entre 0 y 1. NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero).';
COMMENT ON COLUMN predictions.metrica_sku.semanas_variabilidad IS
  'Semanas efectivamente usadas en el calculo de variabilidad (52 con historia completa). NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero).';
COMMENT ON COLUMN predictions.metrica_sku.prop_stock_reconstruido IS
  'Proporcion de la ventana con origen = reconstruido en stock_historico. Sirve para saber sobre que datos se calculo cada corrida al compararlas. NULL con menos de 40 semanas o sin serie de stock (NULL = no calculable, no cero).';

COMMIT;
