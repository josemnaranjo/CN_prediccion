-- ============================================================================
-- Carga inicial del catálogo de productos del módulo
-- ----------------------------------------------------------------------------
-- Motor    : PostgreSQL
-- Proyecto : Módulo de pronóstico de demanda — Distribuidora de Confites JNC
-- Fecha    : 30-09-2026
--
-- Los 22 SKUs de clusters AX/AY que constituyen el universo del estudio.
--
-- Por qué hace falta este script: el batch refresca el nombre y la
-- superfamilia de cada producto desde Softland en cada corrida, pero el
-- cluster ABC-XYZ no existe en el ERP. Es el resultado de la clasificación
-- por volumen y variabilidad hecha en la tesis, así que se carga una vez
-- por ambiente y de ahí en adelante el batch lo respeta.
--
-- Correr DESPUÉS de 001_predictions_tesis.sql, sobre la misma base.
-- Es idempotente: se puede repetir sin duplicar nada.
--
-- Los nombres van como referencia para poder leer la tabla antes de la
-- primera corrida. Si el batch ya escribió el nombre oficial de Softland,
-- este script lo conserva y solo actualiza el cluster.
-- ============================================================================

INSERT INTO predictions.sku (kopr, nombre, cluster_abcxyz) VALUES
  -- Cluster AX: alta rotación, baja variabilidad. Son los únicos que cumplen
  -- el objetivo de MAPE <= 20 % a nivel semanal.
  ('5120996', 'KEGOL FRUTILLA 12X48X432GR LEY',  'AX'),
  ('5121030', 'CAR.MENTA CHOCOL.12X1X530grLEY',  'AX'),
  ('5131018', 'ALKA MENTOL 20x1x400gr.LEY',      'AX'),
  ('7821025', 'TURRON MANI DEU 1x28x45g LEY',    'AX'),

  -- Cluster AY: alta rotación, variabilidad media.
  ('5101032', 'TURRON GALLETA 4X50X25GRS LEY',   'AY'),
  ('5107116', 'GOLPE 4X30X27G',                  'AY'),
  ('5107119', 'OBLEA BOB 6X20X27G',              'AY'),
  ('5128163', 'MORF TRADICIONAL 10X180G',        'AY'),
  ('5131012', 'ALKA ICE CEREZA 1X20X390G',       'AY'),
  ('5140960', 'HOLANDA BOCADITOS 1X21X300gLEY',  'AY'),
  ('5140989', 'ALKA CRISTAL 12X1X450GRS.LEY',    'AY'),
  ('5141300', 'BUTTER AND CREAM 6X1X810GRS',     'AY'),
  ('5141403', 'BIGTIME ULTRA REDBERRY 12X288G',  'AY'),
  ('5141790', 'ALKA 2 CEREZA 12X12X28G',         'AY'),
  ('5141863', 'ALKA 2 MENTOL 12X12X28G',         'AY'),
  ('5159024', 'TABLETA BOB LECHE 6X24X30G',      'AY'),
  ('5168471', 'SELZ MINI QUESO BS 1X40X35 LEY',  'AY'),
  ('5168472', 'SELZ MINI JAMON BS 1X40X35 LEY',  'AY'),
  ('7813164', 'GROSSO SPEARMINT 12X100X700 LEY', 'AY'),
  ('7813166', 'GROSSO TUTTIFRUT 12X100X700LEY',  'AY'),
  ('7833667', 'CHUP.DEU FRUT/SURT.10X24X432G.',  'AY'),
  ('7840940', 'HOLANDA SURTIDO 1x21x400gr.LEY',  'AY')
ON CONFLICT (kopr) DO UPDATE
   SET cluster_abcxyz = EXCLUDED.cluster_abcxyz,
       nombre         = COALESCE(predictions.sku.nombre, EXCLUDED.nombre),
       actualizado_en = NOW();


-- ----------------------------------------------------------------------------
-- Verificación
-- ----------------------------------------------------------------------------
SELECT cluster_abcxyz, COUNT(*) AS skus
FROM predictions.sku
WHERE activo
GROUP BY cluster_abcxyz
ORDER BY cluster_abcxyz;
-- ESPERADO: AX = 4, AY = 18


-- ============================================================================
-- Nota sobre el SKU 5131012 (ALKA ICE CEREZA)
-- ----------------------------------------------------------------------------
-- Estuvo ocho semanas consecutivas en cero entre agosto y septiembre de 2026.
-- NO está descontinuado: se verificó en Softland que vendió 999 unidades en
-- los 90 días previos al 30-09-2026 y que tiene stock. Fue un quiebre largo,
-- ya repuesto. Se mantiene en el universo.
--
-- El batch detecta esa situación por su cuenta y no genera proyección
-- mientras el producto esté sin stock; ver 003_censura_quiebres.sql.
-- ============================================================================
