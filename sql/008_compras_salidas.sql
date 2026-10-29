-- ============================================================================
-- Compras y salidas semanales por SKU
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Fecha    : 29-10-2026
--
-- El panel de compras muestra cuándo y cuánto llegó de cada producto frente a
-- lo que sale de bodega. El job separa dos medidas más del mismo cálculo que ya
-- alimenta flujo_neto y las guarda en la fila SKU-semana de stock_historico:
-- no hay tabla nueva.
--
-- Nullable a propósito: las filas de corridas anteriores no tienen el dato y no
-- se inventa. Quien lea la tabla debe tratar NULL como ausencia, no como cero.
--
-- Orden de despliegue: aplicar ANTES de desplegar el job nuevo; si no, el
-- bloque del stock histórico falla, se registra y la corrida sigue sin serie.
--
-- Correr después de 007. Idempotente. No toca ninguna vista.
-- ============================================================================

BEGIN;

ALTER TABLE predictions.stock_historico
  ADD COLUMN IF NOT EXISTS compras NUMERIC(14,2),
  ADD COLUMN IF NOT EXISTS salidas NUMERIC(14,2);

COMMENT ON COLUMN predictions.stock_historico.compras IS
  'Recepciones netas del proveedor en la semana (FCC - NCC). Excluye GRI y GAR.';
COMMENT ON COLUMN predictions.stock_historico.salidas IS
  'Salida de bodega en la semana: venta facturada mas bonificado (FCV + FDV - NCV + GDI).';

COMMIT;
