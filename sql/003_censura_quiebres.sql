-- ============================================================================
-- Marca de demanda censurada por quiebre de stock
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Fecha    : 30-09-2026
--
-- En una semana de quiebre lo observado no es la demanda: es lo que alcanzó a
-- venderse antes de que se acabara el producto. Medir el error del modelo
-- contra esas semanas lo castiga por un acierto.
--
-- El dataset de entrenamiento traía la columna `quiebre_stock`, construida con
-- detección automática de rachas en cero más confirmación manual del Jefe de
-- Ventas. Eso no es reproducible en un batch, así que el job detecta el caso
-- por su forma: demanda bajo el 10 % de la mediana de las 13 semanas previas.
-- Ese criterio recupera las 16 semanas confirmadas a mano en el estudio y
-- además captura las semanas con 1 o 2 unidades, que la detección original
-- —limitada a ceros exactos en rachas de 3 o más— nunca marcó.
--
-- Correr después de 001. Idempotente.
-- ============================================================================

ALTER TABLE predictions.serie_semanal
    ADD COLUMN IF NOT EXISTS censurada BOOLEAN;

COMMENT ON COLUMN predictions.serie_semanal.censurada IS
    'Semana con demanda probablemente censurada por falta de stock. Solo para tipo=''real''. Excluida del MAPE; el valor se conserva en la serie.';

ALTER TABLE predictions.metrica_sku
    ADD COLUMN IF NOT EXISTS semanas_censuradas INTEGER,
    ADD COLUMN IF NOT EXISTS en_quiebre         BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN predictions.metrica_sku.en_quiebre IS
    'El SKU estaba sin stock en la última semana cerrada. El batch no genera proyección para estos productos y el dashboard debe indicarlo en vez de mostrar una línea.';


-- ----------------------------------------------------------------------------
-- La vista expone la marca y el estado de quiebre
-- ----------------------------------------------------------------------------
-- Se recrea en vez de reemplazarse: CREATE OR REPLACE VIEW solo admite agregar
-- columnas al final, y acá se intercalan en medio.
DROP VIEW IF EXISTS predictions.v_serie_actual;

CREATE VIEW predictions.v_serie_actual AS
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
    r.id AS batch_run_id
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
    'Serie vigente para el gráfico, con las métricas del SKU. Filtrar por kopr y ordenar por semana; `semana_corte` marca dónde termina lo real. Si `sku_en_quiebre` es verdadero no habrá filas de tipo proyeccion.';
