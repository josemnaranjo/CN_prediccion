# CN_prediccion — job de pronóstico de demanda JNC

Job batch que estima la demanda semanal de los 22 SKUs del módulo de
prevención de quiebres de Distribuidora de Confites JNC. Lee las ventas de
Softland (SQL Server), aplica el modelo validado en la tesis y persiste los
resultados en PostgreSQL para que el dashboard los consuma.

## Qué modelo corre

**ARIMA(1,1,1) por SKU sobre demanda semanal ISO winsorizada**, con horizonte
de cuatro semanas. Es el modelo que ganó la comparación del estudio del
21-09-2026 frente a Random Forest, LightGBM y XGBoost por SKU, un Random
Forest global y tres líneas base.

| Métrica | Valor |
|---|---|
| MAPE semanal (agrupado) | 28,16 % |
| MAPE semanal (mediana por SKU) | 25,2 % |
| MAPE sobre demanda acumulada a 4 semanas (mediana) | 17,6 % |
| SKUs con MAPE a 4 semanas ≤ 20 % | 15 de 22 |

El objetivo académico de MAPE ≤ 20 % se evalúa sobre la demanda acumulada en
el horizonte de reposición, no semana a semana. La justificación está en el
documento de ayuda memoria del seminario.

## Archivos

| Archivo | Rol |
|---|---|
| `batch_prediccion_jnc.py` | El job. Es lo único que se ejecuta. |
| `sql/001_predictions_tesis.sql` | Esquema `predictions` en PostgreSQL. Correr una vez por ambiente. |
| `batch_runner.py` | Job de la prueba de concepto (Prophet, SKUs de Perfil A, escala diaria). **Superado.** Se conserva como referencia; no se copia a la imagen. |

## Variables de entorno

| Variable | Contenido |
|---|---|
| `M32QUILLOTA_DATABASE_URL` | `sqlserver://host:puerto;database=...;user=...;password=...` |
| `POSTGRES_DATABASE_URL` | URL de conexión de PostgreSQL |

En Railway las define el servicio. En local, el job busca automáticamente un
`.env` propio o el de `dashboard_nestjs`, así que basta con tener ese archivo.

## Uso

```bash
python batch_prediccion_jnc.py                 # corrida normal
python batch_prediccion_jnc.py --dry-run       # calcula y reporta, no escribe
python batch_prediccion_jnc.py --sku 5131018   # un solo producto
python batch_prediccion_jnc.py --verbose       # métrica de cada SKU

# Verificación contra el dataset de la tesis, sin tocar ninguna base:
python batch_prediccion_jnc.py --desde-csv dataset_cluster_AXAY_24-08-2026.csv --dry-run
```

Ese último comando debe reproducir exactamente las cifras del estudio: 111
semanas capadas, 38.657 unidades retiradas, MAPE semanal agrupado 28,16 % y
MAPE a 4 semanas mediano 17,58 %. Si alguna se mueve, algo cambió en la
lógica y hay que revisarlo antes de desplegar.

## Qué escribe

Todo en el esquema `predictions`:

- **`batch_run`** — una fila por corrida, con el rango de historia usado y el
  estado final.
- **`sku`** — catálogo de los 22 productos. El nombre y la superfamilia se
  refrescan desde Softland en cada corrida; el cluster ABC-XYZ se carga una
  vez a mano, porque no vive en el ERP.
- **`serie_semanal`** — las tres series, distinguidas por la columna `tipo`:
  `real` (demanda observada, con la cruda y la bandera de winsorización),
  `backtest` (predicción a un paso de las últimas semanas cerradas) y
  `proyeccion` (cuatro semanas hacia adelante, con intervalo al 80 %).
- **`metrica_sku`** — MAPE semanal y a cuatro semanas por producto.

La vista `predictions.v_serie_actual` expone siempre la última corrida
exitosa; es la que debe leer el backend.

## Dos decisiones que no se deben cambiar sin rehacer el estudio

**La semana se identifica por la fecha de su lunes.** El par (año, número de
semana) es ambiguo en los cambios de año: la semana 1 del año ISO puede
empezar en diciembre. Ese error apareció en el dataset de entrenamiento y
desplazó tres semanas en un año completo.

**La regla de agregación de ventas está reconciliada contra el dataset de
entrenamiento:** `FCV` y `FDV` en positivo, `NCV` en negativo, por
`MAEDDO.KOPRCT` y `MAEDDO.CAPRCO1`, sin filtros de empresa, sucursal ni
estado. Coincide exactamente en 153 de 157 semanas para los 22 SKUs.
Modificar la lista de tipos de documento invalida esa equivalencia: `FCC` son
compras y `GDI`/`GRI` son guías internas, y sumarlas duplicaría la demanda.

## Frecuencia

Semanal, los lunes temprano, cuando la semana anterior ya cerró. El job
descarta la semana en curso porque está incompleta.
