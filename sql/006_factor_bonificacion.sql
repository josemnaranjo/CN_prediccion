-- ============================================================================
-- Factor de bonificación y cobertura con salida amplificada
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Fecha    : 21-10-2026
--
-- La bodega no se vacía solo con la venta facturada: también sale mercadería
-- bonificada (GDI). La demanda proyectada modela únicamente la venta facturada,
-- así que la cobertura "stock / demanda" sobreestimaba cuánto dura el stock.
--
-- El job calcula un factor por SKU (bonificado sobre venta neta, agregado en
-- las últimas 13 semanas) y lo guarda en metrica_sku.factor_bonificacion.
-- NULL significa que no se pudo calcular o que el SKU está excluido.
--
-- La vista v_cobertura_actual divide el stock por la salida de bodega:
--   salida_semanal = demanda_semanal * (1 + factor_bonificacion)
-- Con factor NULL se toma 0, y la cobertura queda idéntica a la de la
-- migración 004. Las columnas existentes conservan nombre, orden y
-- significado; las dos nuevas (factor_bonificacion, salida_semanal) van al
-- final.
--
-- La vista se recrea con DROP + CREATE porque CREATE OR REPLACE solo admite
-- agregar columnas al final sin cambiar las existentes, y aquí cambia la
-- expresión de las columnas de cobertura dentro de una vista que se reescribe
-- completa; así el resultado no depende de la definición previa.
--
-- Orden de despliegue: aplicar esta migración ANTES de desplegar el job
-- nuevo; si no, el INSERT en metrica_sku falla y la corrida queda en error.
--
-- Correr después de 005. Idempotente.
-- ============================================================================

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
