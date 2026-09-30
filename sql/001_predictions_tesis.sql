-- ============================================================================
-- Esquema de predicciones — modelo de la tesis (ARIMA por SKU, semanal)
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Proyecto : Módulo de pronóstico de demanda — Distribuidora de Confites JNC
-- Fecha    : 29-09-2026
--
-- Reemplaza el esquema de la prueba de concepto. Cambios respecto de aquel:
--   * la semana se identifica por la FECHA DEL LUNES ISO, no por (año, semana);
--     el par (año, número de semana) es ambiguo en los cambios de año y ya
--     produjo un error de etiquetado en el dataset de entrenamiento
--   * una sola tabla de serie con una columna discriminadora `tipo`, de modo
--     que el gráfico de dos líneas se arme con una única consulta ordenada
--   * se persiste la serie real y el backtest, no solo la proyección
--   * las métricas guardan el MAPE semanal y el acumulado a cuatro semanas
--
-- Idempotente: se puede correr varias veces.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS predictions;


-- ----------------------------------------------------------------------------
-- Ejecuciones del batch
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions.batch_run (
    id                BIGSERIAL PRIMARY KEY,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at       TIMESTAMPTZ,
    status            TEXT        NOT NULL DEFAULT 'running',
    modelo            TEXT,                  -- 'ARIMA(1,1,1)'
    skus_procesados   INTEGER,
    skus_fallidos     INTEGER,
    -- Trazabilidad de los datos de entrada: permite reproducir una corrida
    semana_inicio     DATE,                  -- lunes de la primera semana leída
    semana_corte      DATE,                  -- lunes de la última semana con dato real
    semanas_historia  INTEGER,
    duracion_segundos NUMERIC(10,2),
    error_mensaje     TEXT,
    CONSTRAINT batch_run_status_chk
        CHECK (status IN ('running','ok','error'))
);

COMMENT ON COLUMN predictions.batch_run.semana_corte IS
    'Lunes de la última semana cerrada con venta real. La proyección arranca la semana siguiente.';


-- ----------------------------------------------------------------------------
-- Catálogo de productos incluidos en el módulo
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions.sku (
    kopr            VARCHAR(20) PRIMARY KEY,   -- MAEDDO.KOPRCT sin espacios
    nombre          TEXT,
    superfamilia    TEXT,
    cluster_abcxyz  VARCHAR(4),                -- AX / AY
    activo          BOOLEAN     NOT NULL DEFAULT TRUE,
    actualizado_en  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE predictions.sku IS
    'Universo del módulo: los 22 SKUs AX/AY del estudio. `activo` permite sacar un producto sin borrar su historia.';


-- ----------------------------------------------------------------------------
-- Serie semanal unificada: real, backtest y proyección
-- ----------------------------------------------------------------------------
-- El gráfico de dos líneas se arma con un solo SELECT sobre esta tabla:
--   línea "venta real"  -> tipo = 'real'
--   línea "modelo"      -> tipo IN ('backtest','proyeccion'), que se dibuja
--                          continua hasta la semana de corte y punteada después
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions.serie_semanal (
    id              BIGSERIAL PRIMARY KEY,
    batch_run_id    BIGINT      NOT NULL REFERENCES predictions.batch_run(id) ON DELETE CASCADE,
    kopr            VARCHAR(20) NOT NULL,
    semana          DATE        NOT NULL,   -- LUNES de la semana ISO
    tipo            TEXT        NOT NULL,
    unidades        NUMERIC(14,2) NOT NULL,
    unidades_inf    NUMERIC(14,2),          -- extremo inferior del intervalo
    unidades_sup    NUMERIC(14,2),          -- extremo superior
    -- Solo para tipo='real': deja ver el efecto de la winsorización sin
    -- necesidad de una segunda tabla
    unidades_crudas NUMERIC(14,2),
    fue_tratada     BOOLEAN,
    CONSTRAINT serie_tipo_chk
        CHECK (tipo IN ('real','backtest','proyeccion')),
    CONSTRAINT serie_lunes_chk
        CHECK (EXTRACT(ISODOW FROM semana) = 1),
    CONSTRAINT serie_unica
        UNIQUE (batch_run_id, kopr, semana, tipo)
);

CREATE INDEX IF NOT EXISTS serie_semanal_lectura_idx
    ON predictions.serie_semanal (batch_run_id, kopr, semana);
CREATE INDEX IF NOT EXISTS serie_semanal_tipo_idx
    ON predictions.serie_semanal (batch_run_id, tipo);

COMMENT ON CONSTRAINT serie_lunes_chk ON predictions.serie_semanal IS
    'Garantiza en la base que la fecha guardada es un lunes: si el batch calcula mal la semana, la inserción falla en vez de producir un gráfico desalineado.';
COMMENT ON COLUMN predictions.serie_semanal.unidades_crudas IS
    'Demanda antes de winsorizar. Igual a `unidades` cuando la semana no fue tratada.';


-- ----------------------------------------------------------------------------
-- Métricas de calidad por SKU
-- ----------------------------------------------------------------------------
-- Se calculan sobre el backtest de las últimas semanas cerradas. El MAPE
-- excluye las semanas con demanda cero, donde el error relativo no está
-- definido. `mape_4sem` es el que corresponde mostrar en la vista de cuatro
-- semanas; `mape_semanal` en la vista semanal.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions.metrica_sku (
    id                BIGSERIAL PRIMARY KEY,
    batch_run_id      BIGINT      NOT NULL REFERENCES predictions.batch_run(id) ON DELETE CASCADE,
    kopr              VARCHAR(20) NOT NULL,
    semanas_backtest  INTEGER     NOT NULL,
    semanas_para_mape INTEGER     NOT NULL,   -- las que tienen demanda > 0
    mape_semanal      NUMERIC(8,2),
    mae_semanal       NUMERIC(14,2),
    mape_4sem         NUMERIC(8,2),           -- sobre demanda acumulada a 4 semanas
    bloques_4sem      INTEGER,
    CONSTRAINT metrica_unica UNIQUE (batch_run_id, kopr)
);


-- ----------------------------------------------------------------------------
-- Vista de lectura para el dashboard
-- ----------------------------------------------------------------------------
-- Expone siempre la última corrida exitosa, para que el backend no tenga que
-- resolver cuál es. Añade el año y el número de semana ISO ya calculados,
-- que sirven para la etiqueta del eje pero NO como clave.
-- ----------------------------------------------------------------------------
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
    r.semana_corte,
    r.id AS batch_run_id
FROM predictions.serie_semanal AS s
JOIN predictions.batch_run     AS r ON r.id = s.batch_run_id
LEFT JOIN predictions.sku      AS p ON p.kopr = s.kopr
WHERE r.id = (
    SELECT id FROM predictions.batch_run
    WHERE status = 'ok'
    ORDER BY finished_at DESC
    LIMIT 1
);

COMMENT ON VIEW predictions.v_serie_actual IS
    'Serie vigente para el gráfico. Filtrar por kopr y ordenar por semana; `semana_corte` marca dónde termina lo real y empieza la proyección.';
