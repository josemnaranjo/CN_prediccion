-- ============================================================================
-- Stock histórico semanal por SKU
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Fecha    : 22-10-2026
--
-- stock_snapshot solo guarda la foto de cada corrida, y Softland no conserva
-- el inventario pasado. El job reconstruye hacia atrás el stock semanal desde
-- una captura ancla (restando los movimientos de bodega) y lo completa hacia
-- adelante con los cierres reales posteriores. Esta tabla guarda esa serie:
-- una fila por corrida, SKU, bodega y lunes de semana ISO.
--
-- stock_fin es el stock al cierre de la semana. No se recorta a cero: una
-- semana negativa delata movimientos que Softland no registra, y el job la
-- cuenta en su control de calidad. flujo_neto es entradas menos salidas de
-- la semana.
--
-- origen distingue cómo se obtuvo cada punto; ver el COMMENT de la columna.
--
-- Orden de despliegue: aplicar ANTES de desplegar el job nuevo; si no, el
-- bloque falla, se registra y la corrida sigue sin serie.
--
-- Correr después de 006. Idempotente.
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS predictions.stock_historico (
    batch_run_id  BIGINT        NOT NULL REFERENCES predictions.batch_run(id) ON DELETE CASCADE,
    kopr          VARCHAR(20)   NOT NULL,
    bodega        VARCHAR(10)   NOT NULL,
    semana        DATE          NOT NULL,
    stock_fin     NUMERIC(14,2) NOT NULL,
    flujo_neto    NUMERIC(14,2) NOT NULL,
    origen        VARCHAR(12)   NOT NULL,
    PRIMARY KEY (batch_run_id, kopr, bodega, semana),
    CONSTRAINT stock_historico_semana_lunes CHECK (EXTRACT(ISODOW FROM semana) = 1),
    CONSTRAINT stock_historico_origen  CHECK (origen IN ('reconstruido', 'capturado'))
);

COMMENT ON COLUMN predictions.stock_historico.origen IS
    'reconstruido: stock calculado hacia atras desde la captura ancla restando movimientos de bodega. capturado: cierre real tomado de una captura de stock_snapshot (la ancla y las semanas posteriores).';

-- Para leer la serie de un SKU a traves de las corridas
CREATE INDEX IF NOT EXISTS stock_historico_kopr_semana_idx
    ON predictions.stock_historico (kopr, semana);

COMMIT;
