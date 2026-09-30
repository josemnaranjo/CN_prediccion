"""
batch_prediccion_jnc.py — Job batch de pronóstico de demanda JNC

Reemplaza a batch_runner.py, que corría la configuración de la prueba de
concepto (Prophet sobre los SKUs de Perfil A, en escala diaria). Este batch
implementa el modelo validado en la tesis:

    ARIMA(1,1,1) por SKU, sobre demanda semanal ISO winsorizada,
    22 SKUs de los clusters AX/AY, horizonte de 4 semanas.

Lee las ventas de Softland (MSSQL), aplica el mismo tratamiento de atípicos
que el dataset de entrenamiento, hace un backtest de las últimas semanas
cerradas y proyecta cuatro semanas hacia adelante. Persiste las tres series
—real, backtest y proyección— en PostgreSQL, esquema `predictions`.

Decisiones que vienen del estudio y no deben cambiarse sin rehacerlo:

  * La semana se identifica por la FECHA DE SU LUNES. El par (año, número de
    semana) es ambiguo en los cambios de año: la semana 1 del año ISO puede
    empezar en diciembre. Ese error ya apareció en el dataset de entrenamiento
    y aquí se evita de raíz.
  * La regla de agregación de ventas está reconciliada contra el dataset de
    entrenamiento: coincide exactamente en 153 de 157 semanas para los 22 SKUs.
    Cambiar la lista de tipos de documento invalida esa reconciliación.
  * Los atípicos se winsorizan por SKU con el consenso de tres criterios. Los
    atípicos bajos NO se tratan: son quiebres de stock, es decir demanda
    censurada, no errores de registro.

Uso:
    python batch_prediccion_jnc.py                    # corrida normal
    python batch_prediccion_jnc.py --dry-run          # no escribe en PostgreSQL
    python batch_prediccion_jnc.py --sku 5131018      # un solo producto
    python batch_prediccion_jnc.py --desde-csv ruta.csv --dry-run
        # usa un CSV con columnas sku, anio, semana_iso, unidades_semana en
        # lugar de MSSQL. Sirve para verificar contra el dataset de la tesis.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ============================================================================
# Configuración
# ============================================================================

# Los 22 SKUs del estudio (clusters AX y AY). Códigos de MAEDDO.KOPRCT.
SKUS_TESIS: tuple[str, ...] = (
    "5101032", "5107116", "5107119", "5120996", "5121030", "5128163",
    "5131012", "5131018", "5140960", "5140989", "5141300", "5141403",
    "5141790", "5141863", "5159024", "5168471", "5168472", "7813164",
    "7813166", "7821025", "7833667", "7840940",
)

# Tipos de documento de venta. Validado contra el dataset de entrenamiento:
# FCV y FDV suman, NCV resta. Los demás tipos que tocan estos productos son
# compras (FCC) y guías internas (GDI, GRI), que no son demanda de cliente.
TIPOS_VENTA: tuple[str, ...] = ("FCV", "FDV", "NCV")
TIPOS_NEGATIVOS: frozenset[str] = frozenset({"NCV"})


@dataclass
class Config:
    # --- Ventana de datos ---------------------------------------------------
    meses_historia: int = 42        # se leen 42 meses y se usan los que hagan falta
    semanas_entrenamiento: int = 78  # 18 meses, igual que en el estudio
    semanas_backtest: int = 12       # tramo de evaluación, igual que el set de prueba
    min_semanas_sku: int = 30        # mínimo para intentar modelar un SKU

    # --- Modelo -------------------------------------------------------------
    orden_arima: tuple[int, int, int] = (1, 1, 1)
    horizonte: int = 4               # semanas a proyectar
    nivel_intervalo: float = 0.80    # intervalo de confianza de la proyección

    # --- Tratamiento de atípicos (idéntico al notebook del 21-09-2026) ------
    k_iqr: float = 1.5
    z_mad: float = 3.5
    ventana_mediana_movil: int = 13
    k_mediana_movil: float = 4.0
    min_criterios: int = 2
    tratar_inferiores: bool = False  # los atípicos bajos son quiebres de stock

    # --- Operación ----------------------------------------------------------
    skus: tuple[str, ...] = SKUS_TESIS
    bloque_acumulado: int = 4        # tamaño del bloque para el MAPE acumulado

    nombre_modelo: str = field(init=False)

    def __post_init__(self) -> None:
        p, d, q = self.orden_arima
        self.nombre_modelo = f"ARIMA({p},{d},{q})"


CONFIG = Config()

log = logging.getLogger("batch_jnc")


# ============================================================================
# Semanas ISO
# ============================================================================

def lunes_de(fecha) -> date:
    """Lunes de la semana ISO que contiene la fecha.

    Es la única forma en que este batch identifica una semana. No se usa el
    par (año, número de semana) porque es ambiguo en los cambios de año.
    """
    f = pd.Timestamp(fecha).date()
    return f - timedelta(days=f.weekday())


def lunes_desde_anio_semana(anio: int, semana: int) -> date:
    """Lunes a partir de (año ISO, semana ISO). Solo para leer datos antiguos."""
    return date.fromisocalendar(int(anio), int(semana), 1)


def rango_semanas(inicio: date, fin: date) -> list[date]:
    """Todos los lunes entre dos lunes, ambos incluidos."""
    n = (fin - inicio).days // 7 + 1
    return [inicio + timedelta(weeks=i) for i in range(n)]


# ============================================================================
# Conexiones
# ============================================================================

def parse_mssql_url(url: str) -> dict:
    """Convierte M32QUILLOTA_DATABASE_URL al formato de pymssql.

    Formato de entrada: sqlserver://host:port;database=X;user=Y;password=Z;...
    """
    url = url.removeprefix("sqlserver://")
    partes = url.split(";")
    host_port = partes[0]
    host, puerto = host_port.split(":") if ":" in host_port else (host_port, "1433")

    params: dict[str, str] = {}
    for parte in partes[1:]:
        if "=" in parte:
            k, v = parte.split("=", 1)
            params[k.strip().lower()] = v.strip()

    return {
        "server": host,
        "port": int(puerto),
        "database": params.get("database", ""),
        "user": params.get("user", ""),
        "password": params.get("password", ""),
    }


def cargar_env_local() -> None:
    """Carga un .env cuando se corre desde el computador, si existe.

    En Railway las variables vienen del servicio y esto no hace nada. En local
    evita tener que exportarlas a mano: busca el .env del backend, que es donde
    ya están definidas. Si python-dotenv no está instalado, sigue sin ruido.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    aqui = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(aqui, ".env"),
        os.path.join(aqui, "..", "dashboard_nestjs", ".env"),
        os.path.join(aqui, "..", "..", "dashboard_nestjs", ".env"),
    ]
    for ruta in candidatos:
        if os.path.exists(ruta):
            load_dotenv(ruta)
            log.info("Variables de entorno cargadas de %s", os.path.normpath(ruta))
            return


def conectar_mssql():
    import pymssql  # import diferido: el modo --desde-csv no lo necesita

    cfg = parse_mssql_url(os.environ["M32QUILLOTA_DATABASE_URL"])
    return pymssql.connect(charset="UTF-8", **cfg)


def conectar_postgres():
    import psycopg2

    return psycopg2.connect(os.environ["POSTGRES_DATABASE_URL"])


# ============================================================================
# Lectura de ventas
# ============================================================================

def _lista_sql(valores) -> str:
    return ",".join(f"'{v}'" for v in valores)


def leer_ventas_mssql(conn, cfg: Config = CONFIG) -> pd.DataFrame:
    """Ventas diarias por SKU desde Softland.

    Esta consulta es la que quedó validada en la reconciliación del 29-09-2026
    contra el dataset de entrenamiento. Cualquier cambio en los tipos de
    documento, en la columna de cantidad o en la clave de producto rompe la
    equivalencia con los datos sobre los que se midió el error del modelo.
    """
    query = f"""
    SELECT
      RTRIM(DDO.KOPRCT) AS kopr,
      CONVERT(DATE, EDO.FEEMDO) AS fecha,
      SUM(DDO.CAPRCO1 * CASE WHEN EDO.TIDO IN ({_lista_sql(TIPOS_NEGATIVOS)})
                             THEN -1 ELSE 1 END) AS unidades
    FROM dbo.MAEEDO AS EDO WITH (NOLOCK)
    INNER JOIN dbo.MAEDDO AS DDO WITH (NOLOCK) ON EDO.IDMAEEDO = DDO.IDMAEEDO
    WHERE EDO.TIDO IN ({_lista_sql(TIPOS_VENTA)})
      AND EDO.FEEMDO >= DATEADD(MONTH, -{cfg.meses_historia}, GETDATE())
      AND EDO.FEEMDO <  DATEADD(DAY, 1, GETDATE())
      AND RTRIM(DDO.KOPRCT) IN ({_lista_sql(cfg.skus)})
    GROUP BY RTRIM(DDO.KOPRCT), CONVERT(DATE, EDO.FEEMDO)
    """
    df = pd.read_sql(query, conn)
    df["kopr"] = df["kopr"].astype(str).str.strip()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["unidades"] = df["unidades"].astype(float)
    return df


def leer_productos_mssql(conn, cfg: Config = CONFIG) -> pd.DataFrame:
    query = f"""
    SELECT RTRIM(PR.KOPR) AS kopr, RTRIM(PR.NOKOPR) AS nombre,
           RTRIM(FM.NOKOFM) AS superfamilia
    FROM dbo.MAEPR AS PR WITH (NOLOCK)
    INNER JOIN dbo.TABFM AS FM WITH (NOLOCK) ON PR.FMPR = FM.KOFM
    WHERE RTRIM(PR.KOPR) IN ({_lista_sql(cfg.skus)})
    """
    df = pd.read_sql(query, conn)
    df["kopr"] = df["kopr"].astype(str).str.strip()
    return df


def semanalizar(ventas_diarias: pd.DataFrame) -> pd.DataFrame:
    """Agrega las ventas diarias a semanas ISO, identificadas por su lunes."""
    d = ventas_diarias.copy()
    d["semana"] = d["fecha"].map(lunes_de)
    return (d.groupby(["kopr", "semana"], as_index=False)["unidades"]
              .sum()
              .rename(columns={"unidades": "demanda"}))


def leer_ventas_csv(ruta: str) -> pd.DataFrame:
    """Lee un CSV ya semanalizado. Solo para pruebas y verificación.

    Acepta el formato del dataset de la tesis (sku, anio, semana_iso,
    unidades_semana). Ojo: en ese archivo `anio` es el año calendario y no el
    año ISO, así que las semanas de cambio de año quedan mal ubicadas. Se lee
    tal cual para poder reproducir exactamente los números del estudio.
    """
    d = pd.read_csv(ruta, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["kopr"] = d["sku"].astype(str).str.strip()
    d["semana"] = [lunes_desde_anio_semana(a, s)
                   for a, s in zip(d["anio"], d["semana_iso"])]
    d["demanda"] = d["unidades_semana"].astype(float)
    return d[["kopr", "semana", "demanda"]]


def completar_grilla(semanal: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Rellena con cero las semanas sin movimiento y recorta la última semana.

    La última semana del calendario casi siempre está en curso, así que su
    demanda está incompleta y no puede entrar ni al modelo ni al gráfico como
    dato real: haría creer que la demanda se desplomó.
    """
    if semanal.empty:
        return semanal

    ultima_cerrada = lunes_de(date.today()) - timedelta(weeks=1)
    semanal = semanal[semanal["semana"] <= ultima_cerrada]

    inicio, fin = semanal["semana"].min(), semanal["semana"].max()
    grilla = pd.MultiIndex.from_product(
        [sorted(semanal["kopr"].unique()), rango_semanas(inicio, fin)],
        names=["kopr", "semana"])

    d = (semanal.set_index(["kopr", "semana"])
                .reindex(grilla, fill_value=0.0)
                .reset_index())
    # Las semanas con neto negativo son notas de crédito que superan la venta
    # del período. No son demanda negativa: se llevan a cero, igual que en el
    # dataset de entrenamiento.
    d["demanda"] = d["demanda"].clip(lower=0.0)
    return d.sort_values(["kopr", "semana"]).reset_index(drop=True)


# ============================================================================
# Tratamiento de atípicos
# ============================================================================

def detectar_atipicos(serie: pd.Series, cfg: Config = CONFIG) -> pd.DataFrame:
    """Tres criterios independientes sobre la serie de un SKU.

    Réplica exacta de tratamiento_outliers_JNC_completo_21-09-2026.ipynb.
    Una semana se marca cuando al menos `min_criterios` coinciden, lo que evita
    que la peculiaridad de un método contamine la limpieza.
    """
    from scipy import stats

    x = serie.astype(float)
    out = pd.DataFrame(index=serie.index)

    # 1) Rango intercuartílico
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_sup = q3 + cfg.k_iqr * iqr
    out["c_iqr_sup"] = x > limite_sup
    out["c_iqr_inf"] = x < q1 - cfg.k_iqr * iqr

    # 2) z robusto basado en la MAD
    mad = stats.median_abs_deviation(x.dropna())
    if mad > 0:
        z = 0.6745 * (x - x.median()) / mad
        out["c_mad_sup"] = z > cfg.z_mad
        out["c_mad_inf"] = z < -cfg.z_mad
    else:
        out["c_mad_sup"] = out["c_mad_inf"] = False

    # 3) Desvío respecto de la mediana móvil local
    med_movil = x.rolling(cfg.ventana_mediana_movil, center=True, min_periods=3).median()
    residuo = x - med_movil
    escala = residuo.abs().median()
    if escala and escala > 0:
        out["c_local_sup"] = residuo > cfg.k_mediana_movil * escala
        out["c_local_inf"] = residuo < -cfg.k_mediana_movil * escala
    else:
        out["c_local_sup"] = out["c_local_inf"] = False

    out["n_sup"] = out[["c_iqr_sup", "c_mad_sup", "c_local_sup"]].sum(axis=1)
    out["n_inf"] = out[["c_iqr_inf", "c_mad_inf", "c_local_inf"]].sum(axis=1)
    out["limite_sup_iqr"] = limite_sup
    out["mediana_movil"] = med_movil
    return out


def winsorizar(datos: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Capa los atípicos altos al umbral del IQR, dentro de cada SKU.

    Se conserva la fila —no se elimina— para no romper la continuidad de la
    grilla semanal. Devuelve las columnas `demanda` (tratada), `demanda_cruda`
    y `fue_tratada`.

    Advertencia operativa: la mediana móvil está centrada, así que el
    tratamiento de las últimas semanas usa una ventana incompleta y puede
    ajustarse levemente cuando lleguen semanas nuevas. Es aceptable porque esto
    es limpieza del histórico, no una variable predictora, pero implica que la
    serie real mostrada puede revisarse en las últimas semanas entre corridas.
    """
    partes = []
    for kopr, g in datos.groupby("kopr", sort=False):
        g = g.sort_values("semana").copy()
        det = detectar_atipicos(g["demanda"], cfg)

        es_atipico = det["n_sup"] >= cfg.min_criterios
        if cfg.tratar_inferiores:
            es_atipico |= det["n_inf"] >= cfg.min_criterios

        g["demanda_cruda"] = g["demanda"].astype(float)
        nuevo = np.minimum(g["demanda"], det["limite_sup_iqr"])
        g.loc[es_atipico, "demanda"] = nuevo[es_atipico]
        g["demanda"] = g["demanda"].astype(float).clip(lower=0.0)
        g["fue_tratada"] = es_atipico & (g["demanda"] != g["demanda_cruda"])
        partes.append(g)

    return pd.concat(partes, ignore_index=True)


# ============================================================================
# Modelo
# ============================================================================

def _ajustar(serie: np.ndarray, cfg: Config):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(serie, order=cfg.orden_arima,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)


def backtest_arima(serie: pd.Series,
                   cfg: Config = CONFIG) -> tuple[pd.Series, list[tuple[float, float]]]:
    """Backtest sobre las últimas `semanas_backtest` semanas, en dos horizontes.

    Se ajusta una vez sobre la ventana de entrenamiento y después se avanza
    semana a semana incorporando el valor real observado, sin reestimar los
    parámetros (`append(..., refit=False)`). Cada predicción usa solo
    información anterior a la semana que estima.

    Devuelve dos cosas:

    - la predicción **a un paso** de cada semana, que es la que se dibuja en
      el gráfico como tramo de validación del modelo;
    - los bloques de `bloque_acumulado` semanas, cada uno pronosticado de una
      sola vez desde el inicio del bloque, sin conocer ninguna de sus semanas.
      Este segundo horizonte es el que corresponde a lo que hace el producto
      —proyectar cuatro semanas hacia adelante— y por lo tanto el que debe
      sustentar la métrica que se muestra en la vista de cuatro semanas.
      Sumar predicciones a un paso daría un error mucho menor y engañoso,
      porque cada una conoce el resultado de la semana anterior.
    """
    n_test = cfg.semanas_backtest
    entrenamiento = serie.iloc[:-n_test]
    prueba = serie.iloc[-n_test:]

    if len(entrenamiento) < cfg.min_semanas_sku:
        return pd.Series(np.nan, index=prueba.index), []

    ventana = entrenamiento.iloc[-cfg.semanas_entrenamiento:]
    res = _ajustar(ventana.values.astype(float), cfg)

    B = cfg.bloque_acumulado
    valores = prueba.values.astype(float)
    pred_1paso: list[float] = []
    bloques: list[tuple[float, float]] = []

    for i in range(0, n_test, B):
        tramo = valores[i:i + B]
        if len(tramo) == B:
            # Pronóstico de todo el bloque desde su inicio, a ciegas.
            f = np.clip(np.asarray(res.forecast(B), dtype=float), 0, None)
            bloques.append((float(tramo.sum()), float(f.sum())))
        for real in tramo:
            pred_1paso.append(float(np.clip(res.forecast(1)[0], 0, None)))
            res = res.append([real], refit=False)

    return pd.Series(pred_1paso, index=prueba.index), bloques


def proyectar_arima(serie: pd.Series, cfg: Config = CONFIG) -> pd.DataFrame:
    """Reajusta con toda la ventana disponible y proyecta `horizonte` semanas."""
    ventana = serie.iloc[-cfg.semanas_entrenamiento:]
    res = _ajustar(ventana.values.astype(float), cfg)

    pron = res.get_forecast(cfg.horizonte)
    media = np.clip(np.asarray(pron.predicted_mean, dtype=float), 0, None)
    intervalo = np.asarray(pron.conf_int(alpha=1 - cfg.nivel_intervalo), dtype=float)

    ultima = serie.index[-1]
    semanas = [ultima + timedelta(weeks=i) for i in range(1, cfg.horizonte + 1)]
    return pd.DataFrame({
        "semana": semanas,
        "unidades": media,
        "unidades_inf": np.clip(intervalo[:, 0], 0, None),
        "unidades_sup": np.clip(intervalo[:, 1], 0, None),
    })


# ============================================================================
# Métricas
# ============================================================================

def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """MAPE excluyendo las semanas con demanda cero, donde no está definido."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat) & (y != 0)
    if not ok.any():
        return float("nan")
    return float(np.abs((y[ok] - yhat[ok]) / y[ok]).mean() * 100)


def mape_bloques(bloques: list[tuple[float, float]]) -> tuple[float, int]:
    """MAPE sobre la demanda acumulada de cada bloque.

    Es la métrica del horizonte de la decisión: la reposición se decide sobre
    lo que se venderá en las próximas semanas, no sobre una semana aislada.
    Cada bloque viene de un pronóstico hecho de una sola vez desde su inicio.
    """
    validos = [(y, p) for y, p in bloques if y != 0]
    if not validos:
        return float("nan"), 0
    err = [abs(y - p) / y for y, p in validos]
    return float(np.mean(err) * 100), len(validos)


# ============================================================================
# Proceso por SKU
# ============================================================================

def procesar_sku(kopr: str, g: pd.DataFrame, cfg: Config = CONFIG) -> dict | None:
    """Backtest, métricas y proyección de un SKU. None si no hay historia."""
    g = g.sort_values("semana")
    serie = pd.Series(g["demanda"].values, index=list(g["semana"]), dtype=float)

    minimo = cfg.min_semanas_sku + cfg.semanas_backtest
    if len(serie) < minimo:
        log.warning("%s: %d semanas, se requieren %d. Se omite.",
                    kopr, len(serie), minimo)
        return None

    pred_backtest, bloques_pron = backtest_arima(serie, cfg)
    reales = serie.loc[pred_backtest.index]

    m_sem = mape(reales.values, pred_backtest.values)
    m_acum, bloques = mape_bloques(bloques_pron)
    mae = float(np.abs(reales.values - pred_backtest.values).mean())

    return {
        "kopr": kopr,
        "backtest": pred_backtest,
        "proyeccion": proyectar_arima(serie, cfg),
        "metricas": {
            "semanas_backtest": int(len(pred_backtest)),
            "semanas_para_mape": int((reales.values != 0).sum()),
            "mape_semanal": m_sem,
            "mae_semanal": mae,
            "mape_4sem": m_acum,
            "bloques_4sem": bloques,
        },
    }


# ============================================================================
# Persistencia
# ============================================================================

def _nulo(v):
    """NaN -> None, para que psycopg2 escriba NULL y no el texto 'nan'."""
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def abrir_corrida(conn, cfg: Config, inicio: date, corte: date, n_semanas: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions.batch_run
                (status, modelo, semana_inicio, semana_corte, semanas_historia)
            VALUES ('running', %s, %s, %s, %s) RETURNING id
            """,
            (cfg.nombre_modelo, inicio, corte, n_semanas))
        batch_id = cur.fetchone()[0]
    conn.commit()
    return batch_id


def cerrar_corrida(conn, batch_id: int, estado: str, ok: int, fallidos: int,
                   segundos: float, error: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE predictions.batch_run
               SET finished_at = NOW(), status = %s, skus_procesados = %s,
                   skus_fallidos = %s, duracion_segundos = %s, error_mensaje = %s
             WHERE id = %s
            """,
            (estado, ok, fallidos, round(segundos, 2), error, batch_id))
    conn.commit()


def guardar_productos(conn, productos: pd.DataFrame, clusters: dict[str, str]) -> None:
    from psycopg2.extras import execute_values

    filas = [(r.kopr, r.nombre, r.superfamilia, clusters.get(r.kopr))
             for r in productos.itertuples()]
    if not filas:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.sku (kopr, nombre, superfamilia, cluster_abcxyz)
            VALUES %s
            ON CONFLICT (kopr) DO UPDATE
               SET nombre = EXCLUDED.nombre,
                   superfamilia = EXCLUDED.superfamilia,
                   cluster_abcxyz = COALESCE(EXCLUDED.cluster_abcxyz,
                                             predictions.sku.cluster_abcxyz),
                   actualizado_en = NOW()
        """, filas)
    conn.commit()


def guardar_serie(conn, batch_id: int, filas: list[tuple]) -> None:
    from psycopg2.extras import execute_values

    if not filas:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.serie_semanal
                (batch_run_id, kopr, semana, tipo, unidades,
                 unidades_inf, unidades_sup, unidades_crudas, fue_tratada)
            VALUES %s
        """, [(batch_id, *f) for f in filas], page_size=1000)
    conn.commit()


def guardar_metricas(conn, batch_id: int, metricas: list[dict]) -> None:
    from psycopg2.extras import execute_values

    if not metricas:
        return
    filas = [(batch_id, m["kopr"], m["semanas_backtest"], m["semanas_para_mape"],
              _nulo(m["mape_semanal"]), _nulo(m["mae_semanal"]),
              _nulo(m["mape_4sem"]), m["bloques_4sem"]) for m in metricas]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.metrica_sku
                (batch_run_id, kopr, semanas_backtest, semanas_para_mape,
                 mape_semanal, mae_semanal, mape_4sem, bloques_4sem)
            VALUES %s
        """, filas)
    conn.commit()


def armar_filas_serie(tratado: pd.DataFrame, resultados: list[dict]) -> list[tuple]:
    """Convierte las tres series al formato de predictions.serie_semanal."""
    filas: list[tuple] = []

    for r in tratado.itertuples():
        filas.append((r.kopr, r.semana, "real", round(float(r.demanda), 2),
                      None, None, round(float(r.demanda_cruda), 2),
                      bool(r.fue_tratada)))

    for res in resultados:
        for semana, valor in res["backtest"].items():
            if np.isfinite(valor):
                filas.append((res["kopr"], semana, "backtest",
                              round(float(valor), 2), None, None, None, None))
        for p in res["proyeccion"].itertuples():
            filas.append((res["kopr"], p.semana, "proyeccion",
                          round(float(p.unidades), 2),
                          round(float(p.unidades_inf), 2),
                          round(float(p.unidades_sup), 2), None, None))
    return filas


# ============================================================================
# Main
# ============================================================================

def resumen(resultados: list[dict], cfg: Config) -> None:
    if not resultados:
        log.warning("No hubo SKUs procesados.")
        return
    m = pd.DataFrame([r["metricas"] | {"kopr": r["kopr"]} for r in resultados])
    log.info("Resumen del backtest (%d semanas por SKU):", cfg.semanas_backtest)
    log.info("  MAPE semanal  — mediana %.2f%% | media %.2f%%",
             m["mape_semanal"].median(), m["mape_semanal"].mean())
    log.info("  MAPE %d semanas — mediana %.2f%% | SKUs <= 20%%: %d de %d",
             cfg.bloque_acumulado, m["mape_4sem"].median(),
             int((m["mape_4sem"] <= 20).sum()), len(m))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Batch de pronóstico de demanda JNC")
    ap.add_argument("--dry-run", action="store_true",
                    help="calcula todo pero no escribe en PostgreSQL")
    ap.add_argument("--sku", action="append", dest="skus",
                    help="procesar solo este SKU (repetible)")
    ap.add_argument("--desde-csv", dest="csv",
                    help="leer las ventas semanales de un CSV en vez de MSSQL")
    ap.add_argument("--sin-tratamiento", action="store_true",
                    help="omitir la winsorización (para comparar)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")

    cargar_env_local()

    cfg = CONFIG
    if args.skus:
        cfg.skus = tuple(args.skus)

    t0 = time.time()
    productos = pd.DataFrame(columns=["kopr", "nombre", "superfamilia"])

    # --- 1. Ventas ---------------------------------------------------------
    if args.csv:
        log.info("Leyendo ventas semanales de %s", args.csv)
        semanal = leer_ventas_csv(args.csv)
        semanal = semanal[semanal["kopr"].isin(cfg.skus)]
    else:
        log.info("Leyendo ventas de Softland (%d meses, %d SKUs)",
                 cfg.meses_historia, len(cfg.skus))
        with conectar_mssql() as cx:
            diarias = leer_ventas_mssql(cx, cfg)
            productos = leer_productos_mssql(cx, cfg)
        log.info("  %d filas diarias, %d productos en el maestro",
                 len(diarias), len(productos))
        semanal = semanalizar(diarias)

    datos = completar_grilla(semanal, cfg)
    if datos.empty:
        log.error("No hay datos de ventas. Se aborta.")
        return 1

    inicio, corte = datos["semana"].min(), datos["semana"].max()
    n_semanas = len(rango_semanas(inicio, corte))
    log.info("Historia: %s a %s (%d semanas, %d SKUs)",
             inicio, corte, n_semanas, datos["kopr"].nunique())

    faltantes = sorted(set(cfg.skus) - set(datos["kopr"]))
    if faltantes:
        log.warning("Sin ventas en el período: %s", ", ".join(faltantes))

    # --- 2. Tratamiento ----------------------------------------------------
    if args.sin_tratamiento:
        datos["demanda_cruda"] = datos["demanda"]
        datos["fue_tratada"] = False
        log.info("Winsorización omitida por --sin-tratamiento")
    else:
        datos = winsorizar(datos, cfg)
        n = int(datos["fue_tratada"].sum())
        retiradas = datos["demanda_cruda"].sum() - datos["demanda"].sum()
        log.info("Winsorización: %d semanas capadas (%.2f%%), %.0f unidades retiradas (%.2f%%)",
                 n, 100 * n / len(datos), retiradas,
                 100 * retiradas / max(datos["demanda_cruda"].sum(), 1))

    # --- 3. Modelo ---------------------------------------------------------
    resultados, fallidos = [], []
    for kopr, g in datos.groupby("kopr", sort=True):
        try:
            r = procesar_sku(kopr, g, cfg)
            if r is None:
                fallidos.append(kopr)
                continue
            resultados.append(r)
            log.debug("  %s  MAPE semanal %.1f%%  |  MAPE %d sem %.1f%%", kopr,
                      r["metricas"]["mape_semanal"], cfg.bloque_acumulado,
                      r["metricas"]["mape_4sem"])
        except Exception as e:  # noqa: BLE001 — un SKU no debe tumbar la corrida
            log.error("  %s falló: %s: %s", kopr, type(e).__name__, e)
            fallidos.append(kopr)

    log.info("Modelados %d SKUs, fallidos %d (%.1fs)",
             len(resultados), len(fallidos), time.time() - t0)
    resumen(resultados, cfg)

    # --- 4. Persistencia ---------------------------------------------------
    if args.dry_run:
        log.info("--dry-run: no se escribe en PostgreSQL")
        return 0

    conn = conectar_postgres()
    batch_id = abrir_corrida(conn, cfg, inicio, corte, n_semanas)
    try:
        clusters = {}  # el cluster ABC-XYZ se administra fuera del batch
        if len(productos):
            guardar_productos(conn, productos, clusters)
        guardar_serie(conn, batch_id, armar_filas_serie(datos, resultados))
        guardar_metricas(conn, batch_id, [r["metricas"] | {"kopr": r["kopr"]}
                                          for r in resultados])
        cerrar_corrida(conn, batch_id, "ok", len(resultados), len(fallidos),
                       time.time() - t0)
        log.info("Corrida %d guardada.", batch_id)
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        cerrar_corrida(conn, batch_id, "error", len(resultados), len(fallidos),
                       time.time() - t0, f"{type(e).__name__}: {e}")
        log.exception("La corrida %d falló al persistir", batch_id)
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
