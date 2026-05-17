"""
batch_runner.py — Job batch de predicción de quiebres de stock JNC
Lee ventas + stock de MSSQL, ejecuta Prophet/ARIMA por SKU,
escribe predicciones y alertas en PostgreSQL (schema predictions).
"""

import os
import sys
import time
import itertools
import warnings
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pymssql
import psycopg2
from psycopg2.extras import execute_values
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

SKUS_PERFIL_A = [
    '5131018', '5121030', '7821025', '5141403', '7833667', '5141300',
    '5158412', '5101032', '5120996', '5120995', '5120994', '5151288',
    '5151290', '9990019', '9990047', '9990011', '9990020', '9990010',
    '9990005', '9990049', '9990009', '9990017', '9990035', '9990038'
]

# ---------------------------------------------------------------------------
# Conexiones
# ---------------------------------------------------------------------------

def parse_mssql_url(url: str) -> dict:
    """Parsea M32QUILLOTA_DATABASE_URL al formato que necesita pymssql.
    Formato: sqlserver://host:port;database=X;user=Y;password=Z;...
    """
    url = url.removeprefix('sqlserver://')
    parts = url.split(';')
    host_port = parts[0]
    host, port = host_port.split(':') if ':' in host_port else (host_port, '1433')

    params = {}
    for part in parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            params[k.strip().lower()] = v.strip()

    return {
        'server': host,
        'port': int(port),
        'database': params.get('database', ''),
        'user': params.get('user', ''),
        'password': params.get('password', ''),
    }


def connect_mssql():
    url = os.environ['M32QUILLOTA_DATABASE_URL']
    cfg = parse_mssql_url(url)
    return pymssql.connect(
        server=cfg['server'],
        port=cfg['port'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        charset='UTF-8',
    )


def connect_postgres():
    return psycopg2.connect(os.environ['POSTGRES_DATABASE_URL'])


# ---------------------------------------------------------------------------
# Lectura MSSQL
# ---------------------------------------------------------------------------

def fetch_ventas(conn) -> pd.DataFrame:
    skus_str = ','.join(f"'{s}'" for s in SKUS_PERFIL_A)
    query = f"""
    SELECT
      RTRIM(DDO.KOPRCT) AS kopr,
      CONVERT(DATE, EDO.FEEMDO) AS fecha,
      SUM(DDO.CAPRCO1 * CASE WHEN EDO.TIDO = 'NCV' THEN -1 ELSE 1 END) AS unidades
    FROM dbo.MAEEDO AS EDO WITH (NOLOCK)
    INNER JOIN dbo.MAEDDO AS DDO WITH (NOLOCK) ON EDO.IDMAEEDO = DDO.IDMAEEDO
    WHERE EDO.TIDO IN ('BLV','BSV','FCV','FDV','FEV','NCV')
      AND EDO.FEEMDO >= DATEADD(MONTH, -24, GETDATE())
      AND EDO.FEEMDO < DATEADD(DAY, 1, GETDATE())
      AND RTRIM(DDO.KOPRCT) IN ({skus_str})
    GROUP BY RTRIM(DDO.KOPRCT), CONVERT(DATE, EDO.FEEMDO)
    ORDER BY RTRIM(DDO.KOPRCT), fecha
    """
    df = pd.read_sql(query, conn)
    df['kopr'] = df['kopr'].astype(str).str.strip()
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


def fetch_stock(conn) -> pd.DataFrame:
    skus_str = ','.join(f"'{s}'" for s in SKUS_PERFIL_A)
    query = f"""
    SELECT RTRIM(KOPR) AS kopr, STFI1, STDV1
    FROM dbo.MAEST WITH (NOLOCK)
    WHERE KOBO = 'B01' AND RTRIM(KOPR) IN ({skus_str})
    """
    df = pd.read_sql(query, conn)
    df['kopr'] = df['kopr'].astype(str).str.strip()
    return df


def fetch_productos(conn) -> pd.DataFrame:
    skus_str = ','.join(f"'{s}'" for s in SKUS_PERFIL_A)
    query = f"""
    SELECT RTRIM(PR.KOPR) AS kopr, RTRIM(PR.NOKOPR) AS nombre, RTRIM(FM.NOKOFM) AS superfamilia
    FROM dbo.MAEPR AS PR WITH (NOLOCK)
    INNER JOIN dbo.TABFM AS FM WITH (NOLOCK) ON PR.FMPR = FM.KOFM
    WHERE RTRIM(PR.KOPR) IN ({skus_str})
    """
    df = pd.read_sql(query, conn)
    df['kopr'] = df['kopr'].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Modelos — Multi-config Prophet + ARIMA (basado en prototipos prediccion_jnc)
# ---------------------------------------------------------------------------

MIN_SEMANAS = 20
PROPHET_MIN_SEMANAS = 50
VALIDATION_WEEKS = 26

PROPHET_CONFIGS = [
    {'name': 'base_mult', 'mode': 'multiplicative', 'cps': 0.05, 'monthly': True},
    {'name': 'base_add', 'mode': 'additive', 'cps': 0.05, 'monthly': True},
    {'name': 'flex_mult', 'mode': 'multiplicative', 'cps': 0.1, 'monthly': True},
    {'name': 'flex_add', 'mode': 'additive', 'cps': 0.1, 'monthly': True},
    {'name': 'rigid_mult', 'mode': 'multiplicative', 'cps': 0.01, 'monthly': False},
    {'name': 'rigid_add', 'mode': 'additive', 'cps': 0.01, 'monthly': False},
]

PROPHET_CONFIGS_SHORT = [
    {'name': 'short_add', 'mode': 'additive', 'cps': 0.01, 'monthly': False},
    {'name': 'short_mult', 'mode': 'multiplicative', 'cps': 0.01, 'monthly': False},
    {'name': 'short_flex', 'mode': 'additive', 'cps': 0.05, 'monthly': False},
]

HOLIDAYS = pd.DataFrame({
    'holiday': ['fiestas_patrias'] * 3 + ['navidad'] * 3,
    'ds': pd.to_datetime([
        '2024-09-18', '2025-09-18', '2026-09-18',
        '2024-12-25', '2025-12-25', '2026-12-25',
    ]),
    'lower_window': [-3, -3, -3, -7, -7, -7],
    'upper_window': [3, 3, 3, 3, 3, 3],
})


def to_weekly(df_sku: pd.DataFrame) -> pd.DataFrame:
    data = df_sku.set_index('fecha')
    weekly = data['unidades'].resample('W-SUN', label='left', closed='left').sum()
    return weekly.reset_index().rename(columns={'fecha': 'ds', 'unidades': 'y'})


def _fit_prophet(train: pd.DataFrame, cfg: dict, use_holidays: bool) -> Prophet:
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=cfg['mode'],
        changepoint_prior_scale=cfg['cps'],
    )
    if cfg.get('monthly') and len(train) >= 20:
        m.add_seasonality(name='monthly', period=4.35, fourier_order=3)
    if use_holidays:
        m.holidays = HOLIDAYS
    m.fit(train)
    return m


def _prophet_mape(train: pd.DataFrame, test: pd.DataFrame, cfg: dict, use_holidays: bool) -> float | None:
    try:
        m = _fit_prophet(train, cfg, use_holidays)
        future = m.make_future_dataframe(periods=len(test), freq='W-SUN')
        fc = m.predict(future)
        merged = test.merge(fc[['ds', 'yhat']], on='ds', how='inner')
        merged['yhat'] = merged['yhat'].clip(lower=0)
        if len(merged) < 3:
            return None
        mask = merged['y'] > 0
        if mask.sum() < 3:
            return None
        return float((np.abs((merged.loc[mask, 'y'] - merged.loc[mask, 'yhat']) / merged.loc[mask, 'y'])).mean() * 100)
    except:
        return None


def run_prophet_multiconfig(weekly: pd.DataFrame) -> dict | None:
    n = len(weekly)
    val_size = min(VALIDATION_WEEKS, n // 3)
    if val_size < 4:
        return None

    train = weekly.iloc[:-val_size]
    test = weekly.iloc[-val_size:]

    if len(train) < 15:
        return None

    configs = PROPHET_CONFIGS if len(train) >= 40 else PROPHET_CONFIGS_SHORT
    use_holidays = len(train) >= 40

    best_mape = float('inf')
    best_cfg = configs[0]

    for cfg in configs:
        mape = _prophet_mape(train, test, cfg, use_holidays)
        if mape is not None and mape < best_mape:
            best_mape = mape
            best_cfg = cfg

    if best_mape == float('inf'):
        return None

    try:
        m = _fit_prophet(weekly, best_cfg, use_holidays)
        future = m.make_future_dataframe(periods=4, freq='W-SUN')
        fc = m.predict(future)
        forecast_rows = fc.tail(4)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_rows['yhat'] = forecast_rows['yhat'].clip(lower=0)
        forecast_rows['yhat_lower'] = forecast_rows['yhat_lower'].clip(lower=0)

        return {
            'modelo': 'prophet',
            'config': best_cfg['name'],
            'forecast': forecast_rows,
            'mape': round(best_mape, 1),
        }
    except Exception as e:
        print(f"  [Prophet final error] {e}")
        return None


def run_arima_validated(weekly: pd.DataFrame) -> dict | None:
    n = len(weekly)
    val_size = min(VALIDATION_WEEKS, n // 3)
    if val_size < 4:
        return None

    train_df = weekly.iloc[:-val_size]
    test_df = weekly.iloc[-val_size:]
    train_y = train_df['y'].values
    test_y = test_df['y'].values

    if len(train_y) < 10:
        return None

    best_aic = float('inf')
    best_order = (1, 1, 1)
    for p, d, q in itertools.product(range(4), range(3), range(4)):
        if p == 0 and q == 0:
            continue
        try:
            fit = ARIMA(train_y, order=(p, d, q)).fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
        except:
            continue

    try:
        fit = ARIMA(train_y, order=best_order).fit()
        pred = np.maximum(fit.forecast(steps=len(test_y)), 0)
        n_match = min(len(pred), len(test_y))
        mask = test_y[:n_match] > 0
        if mask.sum() < 3:
            mape = None
        else:
            mape = float(np.mean(np.abs((test_y[:n_match][mask] - pred[:n_match][mask]) / test_y[:n_match][mask])) * 100)
    except:
        mape = None

    try:
        all_y = weekly['y'].values
        model = ARIMA(all_y, order=best_order).fit()
        fc = model.get_forecast(steps=4)
        pred_final = np.maximum(fc.predicted_mean, 0)
        ci = fc.conf_int(alpha=0.2)

        last_date = weekly['ds'].max()
        dates = [last_date + timedelta(weeks=i+1) for i in range(4)]

        forecast_rows = pd.DataFrame({
            'ds': dates,
            'yhat': pred_final,
            'yhat_lower': np.maximum(ci[:, 0], 0),
            'yhat_upper': ci[:, 1],
        })

        return {
            'modelo': 'arima',
            'config': str(best_order),
            'forecast': forecast_rows,
            'mape': round(mape, 1) if mape is not None else None,
        }
    except Exception as e:
        print(f"  [ARIMA error] {e}")
        return None


def select_best_model(weekly: pd.DataFrame) -> dict | None:
    result_prophet = None
    result_arima = None

    if len(weekly) >= PROPHET_MIN_SEMANAS:
        result_prophet = run_prophet_multiconfig(weekly)

    result_arima = run_arima_validated(weekly)

    if result_prophet and result_arima:
        mape_p = result_prophet['mape'] if result_prophet['mape'] is not None else float('inf')
        mape_a = result_arima['mape'] if result_arima['mape'] is not None else float('inf')
        if abs(mape_p - mape_a) < 1:
            return result_prophet
        return result_prophet if mape_p <= mape_a else result_arima

    return result_prophet or result_arima


# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------

def calc_alert(kopr: str, nombre: str, superfamilia: str,
               weekly: pd.DataFrame, forecast_row: pd.Series,
               stock_row: pd.Series | None) -> dict:
    stock_actual = float(stock_row['STFI1']) if stock_row is not None else 0.0
    stock_disponible = float(stock_row['STDV1']) if stock_row is not None else 0.0

    last_4 = weekly.tail(4)['y']
    demanda_semanal_prom = float(last_4.mean()) if len(last_4) > 0 else 0.0
    demanda_proxima = float(forecast_row['yhat'])
    demanda_diaria = demanda_semanal_prom / 7.0 if demanda_semanal_prom > 0 else 0.0
    coverage_days = stock_actual / demanda_diaria if demanda_diaria > 0 else 999.0

    if coverage_days <= 3:
        risk_level = 'critico'
    elif coverage_days <= 7:
        risk_level = 'alto'
    elif coverage_days <= 14:
        risk_level = 'medio'
    else:
        risk_level = 'bajo'

    quiebre_estimado = None
    if coverage_days < 14:
        quiebre_estimado = (datetime.now() + timedelta(days=coverage_days)).date()

    return {
        'kopr': kopr,
        'nombre': nombre,
        'superfamilia': superfamilia,
        'stock_actual': stock_actual,
        'stock_disponible': stock_disponible,
        'demanda_semanal_prom': round(demanda_semanal_prom, 2),
        'demanda_proxima_semana': round(demanda_proxima, 2),
        'coverage_days': round(coverage_days, 2),
        'risk_level': risk_level,
        'quiebre_estimado': quiebre_estimado,
    }


# ---------------------------------------------------------------------------
# Escritura PostgreSQL
# ---------------------------------------------------------------------------

def create_batch_run(pg_conn) -> int:
    with pg_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO predictions.batch_run (started_at, status)
            VALUES (NOW(), 'running') RETURNING id
        """)
        batch_id = cur.fetchone()[0]
    pg_conn.commit()
    return batch_id


def save_forecasts(pg_conn, batch_id: int, rows: list[dict]):
    if not rows:
        return
    values = [
        (batch_id, r['kopr'], r['nombre'], r['superfamilia'],
         r['semana'], r['unidades_predichas'], r['unidades_lower'],
         r['unidades_upper'], r['modelo'], r['mape'])
        for r in rows
    ]
    with pg_conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.forecast
            (batch_run_id, kopr, nombre, superfamilia, semana,
             unidades_predichas, unidades_lower, unidades_upper, modelo, mape_historico)
            VALUES %s
        """, values)
    pg_conn.commit()


def save_alerts(pg_conn, batch_id: int, alerts: list[dict]):
    if not alerts:
        return
    values = [
        (batch_id, a['kopr'], a['nombre'], a['superfamilia'],
         a['stock_actual'], a['stock_disponible'], a['demanda_semanal_prom'],
         a['demanda_proxima_semana'], a['coverage_days'], a['risk_level'],
         a['quiebre_estimado'])
        for a in alerts
    ]
    with pg_conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.alert
            (batch_run_id, kopr, nombre, superfamilia,
             stock_actual, stock_disponible, demanda_semanal_prom,
             demanda_proxima_semana, coverage_days, risk_level, quiebre_estimado)
            VALUES %s
        """, values)
    pg_conn.commit()


def finish_batch_run(pg_conn, batch_id: int, status: str,
                     skus_ok: int, skus_fail: int, duration: float,
                     error_msg: str | None = None):
    with pg_conn.cursor() as cur:
        cur.execute("""
            UPDATE predictions.batch_run
            SET finished_at = NOW(), status = %s,
                skus_processed = %s, skus_failed = %s,
                duration_seconds = %s, error_message = %s
            WHERE id = %s
        """, (status, skus_ok, skus_fail, round(duration, 2), error_msg, batch_id))
    pg_conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if 'POSTGRES_DATABASE_URL' not in os.environ:
        print("[ERROR] POSTGRES_DATABASE_URL no definida"); sys.exit(1)
    if 'M32QUILLOTA_DATABASE_URL' not in os.environ:
        print("[ERROR] M32QUILLOTA_DATABASE_URL no definida"); sys.exit(1)

    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"BATCH RUNNER — Predicción de quiebres de stock JNC")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    print("\n[1/5] Conectando a bases de datos...")
    mssql_conn = connect_mssql()
    pg_conn = connect_postgres()
    print("  MSSQL: OK")
    print("  PostgreSQL: OK")

    batch_id = create_batch_run(pg_conn)
    print(f"  batch_run id: {batch_id}")

    try:
        print("\n[2/5] Leyendo datos de MSSQL...")
        df_ventas = fetch_ventas(mssql_conn)
        df_stock = fetch_stock(mssql_conn)
        df_productos = fetch_productos(mssql_conn)
        mssql_conn.close()

        print(f"  Ventas: {len(df_ventas):,} registros diarios")
        print(f"  Stock: {len(df_stock)} SKUs")
        print(f"  Productos: {len(df_productos)} SKUs")

        stock_map = {row['kopr']: row for _, row in df_stock.iterrows()}
        prod_map = {row['kopr']: row for _, row in df_productos.iterrows()}

        print(f"\n[3/5] Procesando {len(SKUS_PERFIL_A)} SKUs...")
        print(f"{'SKU':<12} {'Nombre':<24} {'Sem':>4} {'Modelo':<8} {'Config':<12} {'MAPE':>7} {'Cobert':>8} {'Riesgo':<8}")
        print("-" * 90)

        all_forecasts = []
        all_alerts = []
        skus_ok = 0
        skus_fail = 0

        for kopr in SKUS_PERFIL_A:
            prod = prod_map.get(kopr)
            nombre = str(prod['nombre']).strip() if prod is not None else kopr
            superfamilia = str(prod['superfamilia']).strip() if prod is not None else ''

            sku_ventas = df_ventas[df_ventas['kopr'] == kopr].copy()
            if sku_ventas.empty:
                print(f"{kopr:<12} {'SIN VENTAS':<24}")
                skus_fail += 1
                continue

            weekly = to_weekly(sku_ventas)
            weekly = weekly[weekly['y'] > 0]

            if len(weekly) < MIN_SEMANAS:
                print(f"{kopr:<12} {nombre:<24} {len(weekly):>4} {'EXCL(<20)':<8}")
                skus_fail += 1
                continue

            result = select_best_model(weekly)

            if result is None:
                print(f"{kopr:<12} {nombre:<24} {len(weekly):>4} {'ERROR':<8}")
                skus_fail += 1
                continue

            for _, row in result['forecast'].iterrows():
                all_forecasts.append({
                    'kopr': kopr,
                    'nombre': nombre,
                    'superfamilia': superfamilia,
                    'semana': row['ds'].date(),
                    'unidades_predichas': round(float(row['yhat']), 2),
                    'unidades_lower': round(float(row['yhat_lower']), 2),
                    'unidades_upper': round(float(row['yhat_upper']), 2),
                    'modelo': result['modelo'],
                    'mape': round(result['mape'], 2) if result['mape'] is not None else None,
                })

            stock_row = stock_map.get(kopr)
            first_forecast = result['forecast'].iloc[0]
            alert = calc_alert(kopr, nombre, superfamilia, weekly, first_forecast, stock_row)
            all_alerts.append(alert)
            skus_ok += 1

            mape_str = f"{result['mape']:.1f}%" if result['mape'] else 'N/A'
            cfg_str = result.get('config', '')
            cov_str = f"{alert['coverage_days']:.0f}d" if alert['coverage_days'] < 999 else '>999d'
            print(f"{kopr:<12} {nombre:<24} {len(weekly):>4} {result['modelo']:<8} {cfg_str:<12} {mape_str:>7} {cov_str:>8} {alert['risk_level']:<8}")

        print(f"\n[4/5] Escribiendo resultados en PostgreSQL...")
        save_forecasts(pg_conn, batch_id, all_forecasts)
        save_alerts(pg_conn, batch_id, all_alerts)
        print(f"  Forecasts: {len(all_forecasts)} filas")
        print(f"  Alertas: {len(all_alerts)} filas")

        duration = time.time() - start_time
        finish_batch_run(pg_conn, batch_id, 'success', skus_ok, skus_fail, duration)

        print(f"\n[5/5] Resumen")
        print(f"{'='*70}")
        print(f"  SKUs procesados: {skus_ok}")
        print(f"  SKUs fallidos:   {skus_fail}")
        print(f"  Duración:        {duration:.1f}s")

        risk_counts = {}
        for a in all_alerts:
            risk_counts[a['risk_level']] = risk_counts.get(a['risk_level'], 0) + 1
        for level in ['critico', 'alto', 'medio', 'bajo']:
            print(f"  {level:<10}: {risk_counts.get(level, 0)}")

        print(f"{'='*70}")
        print(f"Batch {batch_id} completado exitosamente.\n")

    except Exception as e:
        duration = time.time() - start_time
        finish_batch_run(pg_conn, batch_id, 'failed', 0, 0, duration, str(e))
        print(f"\n[ERROR] Batch {batch_id} falló: {e}")
        raise
    finally:
        pg_conn.close()


if __name__ == '__main__':
    main()
