-- ============================================================================
-- Foto del stock por corrida
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Fecha    : 01-10-2026
--
-- El gráfico necesita responder "¿para cuántas semanas me alcanza el stock?",
-- y eso requiere el inventario actual, que hasta ahora el batch no leía.
--
-- Se guarda una fila por SKU y por corrida en vez de un único valor vigente.
-- Cuesta lo mismo y tiene un efecto lateral valioso: al cabo de unos meses
-- esta tabla ES la serie histórica de stock que hoy no existe en Softland,
-- y que descartamos reconstruir por lo costoso. Llega sola.
--
-- Se almacena el hecho crudo (las unidades en bodega). La cobertura es un
-- derivado del stock y de la proyección, así que vive en una vista y no en
-- una columna: si cambia la forma de calcularla, no hay que rehacer datos.
--
-- Nota sobre el campo: se usa STFI1 (stock físico). STDV1 (stock disponible)
-- no se mantiene en esta instalación — devuelve cero incluso para productos
-- con catorce mil unidades en bodega.
--
-- Correr después de 003. Idempotente.
-- ============================================================================

CREATE TABLE IF NOT EXISTS predictions.stock_snapshot (
    id            BIGSERIAL PRIMARY KEY,
    batch_run_id  BIGINT        NOT NULL REFERENCES predictions.batch_run(id) ON DELETE CASCADE,
    kopr          VARCHAR(20)   NOT NULL,
    bodega        VARCHAR(10)   NOT NULL,
    stock_fisico  NUMERIC(14,2) NOT NULL,
    capturado_en  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT stock_snapshot_unico UNIQUE (batch_run_id, kopr, bodega)
);

-- Para leer la foto de una corrida
CREATE INDEX IF NOT EXISTS stock_snapshot_run_idx
    ON predictions.stock_snapshot (batch_run_id, kopr);
-- Para leer la evolución de un producto en el tiempo
CREATE INDEX IF NOT EXISTS stock_snapshot_historia_idx
    ON predictions.stock_snapshot (kopr, capturado_en);

COMMENT ON TABLE predictions.stock_snapshot IS
    'Unidades en bodega al momento de cada corrida. Acumula la serie histórica de stock que Softland no guarda.';
COMMENT ON COLUMN predictions.stock_snapshot.stock_fisico IS
    'MAEST.STFI1. No usar STDV1: no se mantiene en esta instalación.';


-- ----------------------------------------------------------------------------
-- Cobertura: cuántas semanas alcanza el stock al ritmo de la demanda proyectada
-- ----------------------------------------------------------------------------
-- La demanda de referencia es el promedio de las semanas proyectadas, no la
-- primera sola, porque es menos sensible al ruido de un punto.
--
-- Las tres cifras responden la misma pregunta con distinta demanda:
--   cobertura_semanas → con la demanda central
--   cobertura_min     → si la demanda resulta alta (se agota antes)
--   cobertura_max     → si resulta baja (alcanza más)
--
-- El supuesto, que el dashboard debe rotular: no ingresa mercadería nueva.
-- Las órdenes de compra en tránsito no se consideran.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW predictions.v_cobertura_actual AS
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
         THEN ROUND(st.stock_fisico / p.demanda_semanal, 1) END AS cobertura_semanas,
    CASE WHEN p.demanda_sup > 0
         THEN ROUND(st.stock_fisico / p.demanda_sup, 1) END     AS cobertura_min,
    CASE WHEN p.demanda_inf > 0
         THEN ROUND(st.stock_fisico / p.demanda_inf, 1) END     AS cobertura_max,
    COALESCE(m.en_quiebre, FALSE)                      AS en_quiebre,
    m.mape_semanal,
    m.mape_4sem,
    c.id AS batch_run_id
FROM corrida AS c
JOIN predictions.stock_snapshot AS st ON st.batch_run_id = c.id
JOIN predictions.sku            AS k  ON k.kopr = st.kopr
LEFT JOIN proyectado            AS p  ON p.kopr = st.kopr
LEFT JOIN predictions.metrica_sku AS m ON m.batch_run_id = c.id AND m.kopr = st.kopr
WHERE k.activo;

COMMENT ON VIEW predictions.v_cobertura_actual IS
    'Una fila por SKU con el stock y para cuántas semanas alcanza. Ordenar por cobertura_semanas deja arriba los productos en riesgo: es el ORDER BY del selector de producto.';
