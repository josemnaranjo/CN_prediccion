-- ============================================================================
-- Ventana de evaluación en la vista de la serie
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Fecha    : 19-10-2026
--
-- El backtest ahora dibuja más semanas de las que evalúa: la línea se extiende
-- hacia atrás, pero las métricas siguen calculándose solo sobre las últimas
-- `semanas_evaluacion`. La vista expone ese número para que el gráfico pueda
-- distinguir la zona evaluada del resto. `metrica_sku.semanas_backtest` ya lo
-- guarda; no hay columnas nuevas en tablas.
--
-- Solo se agrega una columna al final de la vista, por eso basta
-- CREATE OR REPLACE.
--
-- Correr después de 004. Idempotente.
-- ============================================================================

CREATE OR REPLACE VIEW predictions.v_serie_actual AS
SELECT
    s.kopr,
    p.nombre,
    p.cluster_abcxyz,
    s.semana,
    (s.semana + 6) AS semana_fin,
    EXTRACT(ISOYEAR FROM s.semana)::INT AS anio_iso,
    EXTRACT(WEEK    FROM s.semana)::INT AS semana_iso,
    s.tipo,
    s.unidades,
    s.unidades_inf,
    s.unidades_sup,
    s.unidades_crudas,
    s.fue_tratada,
    s.censurada,
    COALESCE(m.en_quiebre, FALSE) AS sku_en_quiebre,
    m.mape_semanal,
    m.mape_4sem,
    r.semana_corte,
    r.id AS batch_run_id,
    m.semanas_backtest AS semanas_evaluacion
FROM predictions.serie_semanal AS s
JOIN predictions.batch_run     AS r ON r.id = s.batch_run_id
LEFT JOIN predictions.sku      AS p ON p.kopr = s.kopr
LEFT JOIN predictions.metrica_sku AS m
       ON m.batch_run_id = s.batch_run_id AND m.kopr = s.kopr
WHERE r.id = (
    SELECT id FROM predictions.batch_run
    WHERE status = 'ok'
    ORDER BY finished_at DESC
    LIMIT 1
);

COMMENT ON VIEW predictions.v_serie_actual IS
    'Serie vigente para el gráfico, con las métricas del SKU. Filtrar por kopr y ordenar por semana; `semana_corte` marca dónde termina lo real. `semanas_evaluacion` indica cuántas de las últimas filas de backtest son la ventana de evaluación. Si `sku_en_quiebre` es verdadero no habrá filas de tipo proyeccion.';
