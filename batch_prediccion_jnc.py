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
from dataclasses import dataclass, field, replace as dc_replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

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
    semanas_entrenamiento: int = 78   # 18 meses, igual que en el estudio
    semanas_evaluacion: int = 12      # ventana de las métricas. NO cambiar: define
                                      # las cifras reportadas en la tesis
    semanas_backtest: int | None = None  # ventana dibujada; None = todo lo posible
    semanas_refit: int = 12           # cada cuántas semanas se reestima el modelo;
                                      # debe ser igual a semanas_evaluacion para que
                                      # las métricas sigan comparables con el estudio
    min_semanas_sku: int = 30         # mínimo para intentar modelar un SKU

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

    # --- Detección de quiebres de stock -------------------------------------
    # Una semana se marca como censurada cuando su demanda cae por debajo de
    # `umbral_censura` veces la mediana de las semanas previas. Reemplaza la
    # detección del estudio (rachas de 3+ semanas en cero exacto confirmadas a
    # mano por el Jefe de Ventas), que no capturaba las semanas con 1 o 2
    # unidades ni los quiebres cortos.
    umbral_censura: float = 0.10
    ventana_censura: int = 13        # semanas previas para la mediana de referencia
    min_ventas_censura: float = 10.0  # si la mediana previa es menor, no se evalúa

    # --- Stock ---------------------------------------------------------------
    bodega: str = "B01"              # bodega de despacho de JNC
    cobertura_aviso: float = 4.0     # semanas; bajo esto se avisa en el log

    # --- Stock histórico ----------------------------------------------------
    # Reconstrucción hacia atrás del stock semanal desde una captura ancla de
    # stock_snapshot. Es un dato paralelo: no entra al modelo ni a las métricas.
    # Con reconstruir_stock = False el job se comporta como antes.
    reconstruir_stock: bool = True
    dias_minimos_ancla: int = 10     # antigüedad mínima de la captura ancla
    tido_entrada: tuple[str, ...] = ("FCC", "GRI", "GAR")      # suman stock
    tido_salida: tuple[str, ...] = ("FCV", "FDV", "GDI")       # restan stock
    tido_entrada_neg: tuple[str, ...] = ("NCC",)  # devolución de compra: resta
    tido_salida_neg: tuple[str, ...] = ("NCV",)   # devolución de venta: suma
    # Recepciones del proveedor para el panel de compras: subconjunto de las
    # entradas (GRI y GAR no son compras).
    tido_compra: tuple[str, ...] = ("FCC",)
    tido_compra_neg: tuple[str, ...] = ("NCC",)
    zona_horaria: str = "America/Santiago"  # para ubicar "lunes de madrugada"
    hora_madrugada_max: int = 6      # capturas locales antes de esta hora cuentan como cierre

    # --- Operación ----------------------------------------------------------
    skus: tuple[str, ...] = SKUS_TESIS
    bloque_acumulado: int = 4        # tamaño del bloque para el MAPE acumulado

    # --- Bonificación (GDI) -------------------------------------------------
    # Mercadería bonificada ("compras 4, te regalo 1"): sale de la bodega pero no
    # aparece en ninguna factura. El factor es un dato paralelo a la demanda: no
    # entra al entrenamiento ni a las métricas.
    semanas_factor_bonif: int = 13   # ventana del factor, en semanas cerradas
    semanas_factor_largo: int = 52   # solo para advertir deriva de la mecánica comercial
    tido_bonificacion: tuple[str, ...] = ("GDI",)
    skus_sin_factor: tuple[str, ...] = ("5131012",)  # unidad de bonificado ≠ unidad de venta
    factor_bonif_maximo: float = 0.60    # sobre esto, solo advierte
    umbral_deriva_factor: float = 0.10   # |factor13 - factor52| sobre esto, advierte

    nombre_modelo: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.semanas_backtest is not None
                and self.semanas_backtest < self.semanas_evaluacion):
            raise ValueError(
                f"semanas_backtest ({self.semanas_backtest}) debe ser >= "
                f"semanas_evaluacion ({self.semanas_evaluacion})")
        if self.semanas_refit <= 0:
            raise ValueError(f"semanas_refit ({self.semanas_refit}) debe ser > 0")
        if set(self.tido_bonificacion) & set(TIPOS_VENTA):
            raise ValueError(
                f"tido_bonificacion {self.tido_bonificacion} no puede incluir "
                f"tipos de venta {TIPOS_VENTA}: se contaría doble")
        grupos = {"tido_entrada": self.tido_entrada,
                  "tido_salida": self.tido_salida,
                  "tido_entrada_neg": self.tido_entrada_neg,
                  "tido_salida_neg": self.tido_salida_neg}
        nombres = list(grupos)
        for i, a in enumerate(nombres):
            for b_ in nombres[i + 1:]:
                comunes = set(grupos[a]) & set(grupos[b_])
                if comunes:
                    raise ValueError(
                        f"{a} y {b_} no pueden compartir tipos de documento "
                        f"{sorted(comunes)}: el stock se contaría doble")
        if not set(self.tido_compra) <= set(self.tido_entrada):
            raise ValueError(f"tido_compra {self.tido_compra} debe ser subconjunto "
                             f"de tido_entrada {self.tido_entrada}")
        if not set(self.tido_compra_neg) <= set(self.tido_entrada_neg):
            raise ValueError(f"tido_compra_neg {self.tido_compra_neg} debe ser "
                             f"subconjunto de tido_entrada_neg {self.tido_entrada_neg}")
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


def _query_ventas(cfg: Config) -> str:
    venta = _lista_sql(TIPOS_VENTA)
    negativos = _lista_sql(TIPOS_NEGATIVOS)
    bonif = _lista_sql(cfg.tido_bonificacion)
    return f"""
    SELECT
      RTRIM(DDO.KOPRCT) AS kopr,
      CONVERT(DATE, EDO.FEEMDO) AS fecha,
      SUM(CASE WHEN EDO.TIDO IN ({venta})
               THEN DDO.CAPRCO1 * CASE WHEN EDO.TIDO IN ({negativos}) THEN -1 ELSE 1 END
               ELSE 0 END) AS unidades,
      SUM(CASE WHEN EDO.TIDO IN ({bonif}) THEN DDO.CAPRCO1 ELSE 0 END) AS bonificado,
      SUM(CASE WHEN EDO.TIDO IN ({venta}) THEN 1 ELSE 0 END) AS lineas_venta
    FROM dbo.MAEEDO AS EDO WITH (NOLOCK)
    INNER JOIN dbo.MAEDDO AS DDO WITH (NOLOCK) ON EDO.IDMAEEDO = DDO.IDMAEEDO
    WHERE EDO.TIDO IN ({_lista_sql(TIPOS_VENTA + tuple(cfg.tido_bonificacion))})
      AND EDO.FEEMDO >= DATEADD(MONTH, -{cfg.meses_historia}, GETDATE())
      AND EDO.FEEMDO <  DATEADD(DAY, 1, GETDATE())
      AND RTRIM(DDO.KOPRCT) IN ({_lista_sql(cfg.skus)})
    GROUP BY RTRIM(DDO.KOPRCT), CONVERT(DATE, EDO.FEEMDO)
    """


def separar_ventas_y_bonificado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte el resultado de `_query_ventas` en la venta y el bonificado.

    La venta conserva solo los días con al menos una línea de venta, que son
    exactamente las filas que devolvía la consulta antes de agregar GDI. Un
    día con solo guías de bonificado no debe aparecer ahí: ampliaría la grilla
    de semanas y movería la historia sobre la que se midió el modelo.
    """
    ventas = (df.loc[df["lineas_venta"] > 0, ["kopr", "fecha", "unidades"]]
                .reset_index(drop=True))
    bonificado = (df.loc[df["bonificado"] != 0, ["kopr", "fecha", "bonificado"]]
                    .reset_index(drop=True))
    return ventas, bonificado


def leer_ventas_mssql(conn, cfg: Config = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ventas diarias por SKU y bonificado diario, desde Softland, en una sola pasada.

    La parte de venta es la consulta validada en la reconciliación del
    29-09-2026 contra el dataset de entrenamiento. Cualquier cambio en los tipos
    de documento, en la columna de cantidad o en la clave de producto rompe la
    equivalencia con los datos sobre los que se midió el error del modelo; por
    eso el bonificado solo se *agrega* como segundo agregado y la venta que se
    devuelve es la misma de antes.
    """
    df = pd.read_sql(_query_ventas(cfg), conn)
    df["kopr"] = df["kopr"].astype(str).str.strip()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["unidades"] = df["unidades"].astype(float)
    df["bonificado"] = df["bonificado"].astype(float)
    return separar_ventas_y_bonificado(df)


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


def leer_stock_mssql(conn, cfg: Config = CONFIG) -> pd.DataFrame:
    """Unidades en bodega al momento de la corrida.

    Se usa STFI1, el stock físico. STDV1 —el "disponible"— no se mantiene en
    esta instalación: devuelve cero incluso para productos con más de catorce
    mil unidades en bodega, así que cualquier cálculo de cobertura hecho sobre
    ese campo daría quiebre inminente en todos los productos, siempre.
    """
    query = f"""
    SELECT RTRIM(KOPR) AS kopr, RTRIM(KOBO) AS bodega, STFI1 AS stock_fisico
    FROM dbo.MAEST WITH (NOLOCK)
    WHERE RTRIM(KOBO) = '{cfg.bodega}'
      AND RTRIM(KOPR) IN ({_lista_sql(cfg.skus)})
    """
    df = pd.read_sql(query, conn)
    df["kopr"] = df["kopr"].astype(str).str.strip()
    df["bodega"] = df["bodega"].astype(str).str.strip()
    df["stock_fisico"] = df["stock_fisico"].astype(float).clip(lower=0)
    return df


def signos_tido(cfg: Config = CONFIG) -> dict[str, int]:
    """Signo con que cada tipo de documento mueve el stock de la bodega.

    Entradas y notas de venta (devuelven mercadería) suman; salidas y notas de
    compra (devuelven al proveedor) restan.
    """
    signos = {t: 1 for t in cfg.tido_entrada + cfg.tido_salida_neg}
    signos.update({t: -1 for t in cfg.tido_salida + cfg.tido_entrada_neg})
    return signos


def _query_movimientos(cfg: Config, desde: date, hasta_excl: date) -> str:
    """Flujo neto semanal por SKU en la bodega, con el signo ya aplicado.

    No se agrega a `_query_ventas`: esa consulta es la validada contra el
    dataset de entrenamiento y la reconciliación del stock es unidad por
    unidad, así que cada una conserva su propio filtro y su propio conjunto de
    documentos. Aquí la bodega es DDO.BOSULIDO y la semana es el lunes de
    FEEMDO (1900-01-01 fue lunes).
    """
    signos = signos_tido(cfg)
    mas = _lista_sql([t for t, s in signos.items() if s > 0])
    menos = _lista_sql([t for t, s in signos.items() if s < 0])
    lunes = "DATEADD(DAY, -(DATEDIFF(DAY, '19000101', EDO.FEEMDO) % 7), CONVERT(DATE, EDO.FEEMDO))"
    return f"""
    SELECT
      RTRIM(DDO.KOPRCT) AS kopr,
      {lunes} AS semana,
      SUM(CASE WHEN EDO.TIDO IN ({mas}) THEN DDO.CAPRCO1
               WHEN EDO.TIDO IN ({menos}) THEN -DDO.CAPRCO1
          END) AS flujo_neto,
      SUM(CASE WHEN EDO.TIDO IN ({_lista_sql(cfg.tido_compra)}) THEN DDO.CAPRCO1
               WHEN EDO.TIDO IN ({_lista_sql(cfg.tido_compra_neg)}) THEN -DDO.CAPRCO1
          END) AS compras,
      SUM(CASE WHEN EDO.TIDO IN ({_lista_sql(cfg.tido_salida)}) THEN DDO.CAPRCO1
               WHEN EDO.TIDO IN ({_lista_sql(cfg.tido_salida_neg)}) THEN -DDO.CAPRCO1
          END) AS salidas
    FROM dbo.MAEEDO AS EDO WITH (NOLOCK)
    INNER JOIN dbo.MAEDDO AS DDO WITH (NOLOCK) ON EDO.IDMAEEDO = DDO.IDMAEEDO
    WHERE EDO.TIDO IN ({_lista_sql(list(signos))})
      AND DDO.BOSULIDO = '{cfg.bodega}'
      AND RTRIM(DDO.KOPRCT) IN ({_lista_sql(cfg.skus)})
      AND EDO.FEEMDO >= '{desde.strftime('%Y%m%d')}'
      AND EDO.FEEMDO < '{hasta_excl.strftime('%Y%m%d')}'
    GROUP BY RTRIM(DDO.KOPRCT), {lunes}
    """


def leer_movimientos_stock(conn, cfg: Config, desde: date,
                           semana_ancla: date) -> pd.DataFrame:
    """Flujo neto semanal por SKU desde `desde` hasta la semana ancla inclusive."""
    hasta_excl = semana_ancla + timedelta(days=7)
    df = pd.read_sql(_query_movimientos(cfg, desde, hasta_excl), conn)
    df["kopr"] = df["kopr"].astype(str).str.strip()
    df["semana"] = pd.to_datetime(df["semana"]).dt.date
    df["flujo_neto"] = df["flujo_neto"].astype(float)
    for col in ("compras", "salidas"):
        # SUM da NULL si ningún documento del grupo cae en el CASE (p. ej. solo GRI)
        df[col] = (df[col] if col in df else 0.0)
        df[col] = pd.to_numeric(df[col]).fillna(0.0).astype(float)
    return df


def semanalizar_bonificado(bonificado: pd.DataFrame) -> pd.DataFrame:
    """Agrega el bonificado diario a semanas ISO (lunes)."""
    if bonificado.empty:
        return pd.DataFrame(columns=["kopr", "semana", "bonificado"])
    d = bonificado.copy()
    d["semana"] = d["fecha"].map(lunes_de)
    return d.groupby(["kopr", "semana"], as_index=False)["bonificado"].sum()


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


def ultima_semana_cerrada(hoy: date | None = None) -> date:
    """Lunes de la última semana ISO completa (la semana en curso no cuenta)."""
    return lunes_de(hoy or date.today()) - timedelta(weeks=1)


def completar_grilla(semanal: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Rellena con cero las semanas sin movimiento y recorta la última semana.

    La última semana del calendario casi siempre está en curso, así que su
    demanda está incompleta y no puede entrar ni al modelo ni al gráfico como
    dato real: haría creer que la demanda se desplomó.
    """
    if semanal.empty:
        return semanal

    ultima_cerrada = ultima_semana_cerrada()
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
# Factor de bonificación
# ============================================================================

FACTOR_MAX_ALMACENABLE = 99.9999  # límite de NUMERIC(6,4)


def _suma_ventana(df: pd.DataFrame | None, col: str, hasta: date, semanas: int) -> dict:
    if df is None or df.empty:
        return {}
    desde = hasta - timedelta(weeks=semanas - 1)
    d = df[(df["semana"] >= desde) & (df["semana"] <= hasta)]
    return d.groupby("kopr")[col].sum().to_dict()


def _razon_bonificacion(kopr: str, bonif: float, venta: float, semanas: int) -> float | None:
    if venta <= 0:
        return None
    f = bonif / venta
    if not 0 <= f < FACTOR_MAX_ALMACENABLE:
        log.warning("  %s: factor de bonificación %.4f fuera de rango en %d semanas "
                    "(bonificado %.0f, venta neta %.0f). Se deja sin factor.",
                    kopr, f, semanas, bonif, venta)
        return None
    return float(f)


def calcular_factores_bonificacion(ventas: pd.DataFrame, bonificado: pd.DataFrame | None,
                                   ultima_cerrada: date,
                                   cfg: Config = CONFIG) -> dict[str, dict[str, float | None]]:
    """Bonificado sobre venta neta, agregado por SKU en la ventana y luego dividido.

    No es el promedio de los factores semanales: semana a semana el factor es
    errático porque la entrega del bonificado no cae en la semana de la factura.
    Al agregar ese ruido de calendario desaparece, y como la tarjeta proyecta
    cuatro semanas, el agregado es la escala que corresponde.

    `ventas` es la venta neta semanal cruda (sin winsorizar). Con
    `bonificado=None` (no hay fuente, p. ej. --desde-csv) devuelve {}.
    """
    if bonificado is None:
        return {}

    resultado: dict[str, dict[str, float | None]] = {}
    ventanas = {"factor": cfg.semanas_factor_bonif, "factor_largo": cfg.semanas_factor_largo}
    sumas = {clave: (_suma_ventana(ventas, "demanda", ultima_cerrada, n),
                     _suma_ventana(bonificado, "bonificado", ultima_cerrada, n))
             for clave, n in ventanas.items()}

    for kopr in cfg.skus:
        if kopr in cfg.skus_sin_factor:
            resultado[kopr] = {"factor": None, "factor_largo": None}
            continue
        resultado[kopr] = {
            clave: _razon_bonificacion(kopr, bon.get(kopr, 0.0), ven.get(kopr, 0.0), n)
            for clave, n in ventanas.items()
            for ven, bon in [sumas[clave]]
        }
    return resultado


def advertir_factores(factores: dict, cfg: Config = CONFIG) -> None:
    """Avisa de factores atípicos: reflejan decisiones comerciales que cambian sin aviso."""
    if not factores:
        return
    con_factor = sum(1 for v in factores.values() if v["factor"] is not None)
    log.info("Factor de bonificación (%d semanas): %d de %d SKUs con factor",
             cfg.semanas_factor_bonif, con_factor, len(factores))
    for kopr, v in sorted(factores.items()):
        f, largo = v["factor"], v["factor_largo"]
        if f is None:
            continue
        if f > cfg.factor_bonif_maximo:
            log.warning("  %s: factor de bonificación %.1f%% sobre el máximo de %.0f%%. "
                        "Se usa igual; verificar que no sea un error de datos.",
                        kopr, 100 * f, 100 * cfg.factor_bonif_maximo)
        if largo is not None and abs(f - largo) > cfg.umbral_deriva_factor:
            log.warning("  %s: el factor de %d semanas (%.1f%%) difiere del de %d semanas "
                        "(%.1f%%) en más de %.0f puntos: la mecánica comercial cambió.",
                        kopr, cfg.semanas_factor_bonif, 100 * f,
                        cfg.semanas_factor_largo, 100 * largo,
                        100 * cfg.umbral_deriva_factor)


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


def marcar_censuradas(datos: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Marca las semanas cuya demanda parece censurada por falta de stock.

    En esas semanas lo observado no es la demanda: es lo que alcanzó a
    venderse antes de que se acabara el producto. Medir el error contra ellas
    castiga al modelo por un acierto — predijo demanda y no hubo qué vender.

    El criterio es relativo al propio SKU: la semana se marca si su demanda es
    menor que `umbral_censura` veces la mediana de las `ventana_censura`
    semanas anteriores. Solo se mira hacia atrás, así que la marca de una
    semana no cambia cuando llegan semanas nuevas.

    La marca NO altera la serie que alimenta al modelo. El estudio dejó los
    ceros de los quiebres dentro de la serie de entrenamiento y solo los
    excluyó del cálculo de variabilidad; mantener ese criterio evita cambiar
    un método ya validado. Lo que sí hace la marca es sacar esas semanas del
    MAPE y avisarle al dashboard que no muestre proyección de un producto que
    hoy está sin stock.
    """
    partes = []
    for _, g in datos.groupby("kopr", sort=False):
        g = g.sort_values("semana").copy()
        referencia = (g["demanda"].shift(1)
                      .rolling(cfg.ventana_censura, min_periods=4).median())
        g["mediana_previa"] = referencia
        g["censurada"] = (
            referencia.notna()
            & (referencia >= cfg.min_ventas_censura)
            & (g["demanda"] < cfg.umbral_censura * referencia)
        )
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


def _n_dibujo(n_serie: int, cfg: Config) -> int:
    """Semanas de backtest que se dibujan para una serie de largo `n_serie`."""
    disponible = n_serie - cfg.semanas_entrenamiento
    n = disponible if cfg.semanas_backtest is None else min(cfg.semanas_backtest, disponible)
    n = max(n, cfg.semanas_evaluacion)   # nunca menos que la ventana de evaluación
    return min(n, n_serie - 1)           # nunca toda la serie


def _puntos_refit(n_serie: int, inicio: int, cfg: Config) -> list[int]:
    """Posiciones donde se reestima el modelo, contadas hacia atrás desde el final.

    Contar desde el final (y no desde el inicio) garantiza un reajuste justo al
    comienzo de la ventana de evaluación y ninguno dentro de ella. Siempre
    incluye `inicio`, que necesita un primer ajuste.
    """
    pts = {inicio}
    p = n_serie - cfg.semanas_refit
    while p > inicio:
        pts.add(p)
        p -= cfg.semanas_refit
    return sorted(pts)


def backtest_arima(serie: pd.Series,
                   cfg: Config = CONFIG) -> tuple[pd.Series, list[tuple[float, float]]]:
    """Backtest con reajuste periódico sobre ventana móvil, en dos horizontes.

    La predicción a un paso se genera para toda la ventana de dibujo (ver
    `_n_dibujo`). El modelo se reestima cada `semanas_refit` semanas sobre las
    `semanas_entrenamiento` previas (solo datos anteriores); entre reajustes se
    avanza con `append(..., refit=False)`. Cada predicción usa solo información
    anterior a la semana que estima.

    Los bloques acumulados de `bloque_acumulado` semanas se calculan solo dentro
    de la ventana de evaluación (las últimas `semanas_evaluacion`), alineados a
    su inicio, cada uno pronosticado de una sola vez desde su comienzo, a ciegas.
    Sumar predicciones a un paso daría un error mucho menor y engañoso.
    """
    N = len(serie)
    n = _n_dibujo(N, cfg)
    inicio = N - n
    if inicio < cfg.min_semanas_sku:
        return pd.Series(np.nan, index=serie.index[inicio:]), []

    eval_ini = N - cfg.semanas_evaluacion
    refits = set(_puntos_refit(N, inicio, cfg))
    B = cfg.bloque_acumulado
    valores = serie.values.astype(float)

    pred_1paso: list[float] = []
    bloques: list[tuple[float, float]] = []
    res = None
    for i in range(inicio, N):
        if i in refits:
            ventana = valores[max(0, i - cfg.semanas_entrenamiento):i]
            if i < eval_ini:
                # Solo alimenta la línea dibujada: un fallo no debe perder el SKU.
                try:
                    res = _ajustar(ventana, cfg)
                except Exception as e:
                    log.warning("backtest: falló el reajuste en i=%d (%s); "
                                "predicciones NaN hasta el próximo reajuste", i, e)
                    res = None
            else:
                res = _ajustar(ventana, cfg)   # ventana de evaluación: sin protección
        if res is None:
            pred_1paso.append(float("nan"))
            continue
        if i >= eval_ini and (i - eval_ini) % B == 0 and i + B <= N:
            # Pronóstico de todo el bloque desde su inicio, a ciegas.
            f = np.clip(np.asarray(res.forecast(B), dtype=float), 0, None)
            bloques.append((float(valores[i:i + B].sum()), float(f.sum())))
        pred_1paso.append(float(np.clip(res.forecast(1)[0], 0, None)))
        res = res.append([valores[i]], refit=False)

    return pd.Series(pred_1paso, index=serie.index[inicio:]), bloques


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

def mape(y: np.ndarray, yhat: np.ndarray, excluir: np.ndarray | None = None) -> float:
    """MAPE excluyendo las semanas con demanda cero y las marcadas en `excluir`.

    En cero el error relativo no está definido. Las excluidas son las semanas
    censuradas por quiebre de stock, donde lo observado no es la demanda.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat) & (y != 0)
    if excluir is not None:
        ok &= ~np.asarray(excluir, bool)
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

    minimo = cfg.min_semanas_sku + cfg.semanas_evaluacion
    if len(serie) < minimo:
        log.warning("%s: %d semanas, se requieren %d. Se omite.",
                    kopr, len(serie), minimo)
        return None

    censurada = (pd.Series(g["censurada"].values, index=list(g["semana"]))
                 if "censurada" in g else pd.Series(False, index=serie.index))

    pred_backtest, bloques_pron = backtest_arima(serie, cfg)

    # Las métricas salen solo de la ventana de evaluación; el resto del
    # backtest es únicamente para dibujar la línea del modelo.
    eval_idx = pred_backtest.index[-cfg.semanas_evaluacion:]
    reales_eval = serie.loc[eval_idx]
    pred_eval = pred_backtest.loc[eval_idx]
    cens_eval = censurada.loc[eval_idx].to_numpy(bool)

    # Un bloque acumulado se descarta completo si alguna de sus semanas está
    # censurada: su suma tampoco representa la demanda del período.
    B = cfg.bloque_acumulado
    bloques_limpios = [b for i, b in enumerate(bloques_pron)
                       if not cens_eval[i * B:(i + 1) * B].any()]

    m_sem = mape(reales_eval.values, pred_eval.values, excluir=cens_eval)
    m_acum, bloques = mape_bloques(bloques_limpios)
    mae = float(np.abs(reales_eval.values - pred_eval.values).mean())

    # Sin stock en la última semana cerrada. La proyección se genera igual
    # —estima demanda, que existe aunque no haya producto— pero el dashboard
    # debe advertirlo: esa demanda no se va a concretar si no se repone.
    en_quiebre = bool(cens_eval[-1]) if len(cens_eval) else False

    return {
        "kopr": kopr,
        "backtest": pred_backtest,
        "proyeccion": proyectar_arima(serie, cfg),
        "metricas": {
            "semanas_backtest": int(len(pred_eval)),   # semanas de las métricas
            "semanas_dibujo": int(len(pred_backtest)), # solo log; no se persiste
            "semanas_para_mape": int(((reales_eval.values != 0) & ~cens_eval).sum()),
            "semanas_censuradas": int(cens_eval.sum()),
            "mape_semanal": m_sem,
            "mae_semanal": mae,
            "mape_4sem": m_acum,
            "bloques_4sem": bloques,
            "en_quiebre": en_quiebre,
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


def guardar_stock(conn, batch_id: int, stock: pd.DataFrame) -> None:
    from psycopg2.extras import execute_values

    if stock is None or stock.empty:
        return
    filas = [(batch_id, r.kopr, r.bodega, round(float(r.stock_fisico), 2))
             for r in stock.itertuples()]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.stock_snapshot
                (batch_run_id, kopr, bodega, stock_fisico)
            VALUES %s
            ON CONFLICT (batch_run_id, kopr, bodega) DO UPDATE
               SET stock_fisico = EXCLUDED.stock_fisico
        """, filas)
    conn.commit()


def guardar_stock_historico(conn, batch_id: int, bodega: str,
                            serie: pd.DataFrame) -> None:
    from psycopg2.extras import execute_values

    if serie is None or serie.empty:
        return

    def medida(r, col):
        v = getattr(r, col, None)
        return None if _nulo(v) is None else round(float(v), 2)

    filas = [(batch_id, r.kopr, bodega, r.semana, round(float(r.stock_fin), 2),
              round(float(r.flujo_neto), 2), r.origen,
              medida(r, "compras"), medida(r, "salidas")) for r in serie.itertuples()]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.stock_historico
                (batch_run_id, kopr, bodega, semana, stock_fin, flujo_neto, origen,
                 compras, salidas)
            VALUES %s
        """, filas, page_size=1000)
    conn.commit()


def guardar_serie(conn, batch_id: int, filas: list[tuple]) -> None:
    from psycopg2.extras import execute_values

    if not filas:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.serie_semanal
                (batch_run_id, kopr, semana, tipo, unidades,
                 unidades_inf, unidades_sup, unidades_crudas, fue_tratada,
                 censurada)
            VALUES %s
        """, [(batch_id, *f) for f in filas], page_size=1000)
    conn.commit()


def guardar_metricas(conn, batch_id: int, metricas: list[dict]) -> None:
    from psycopg2.extras import execute_values

    if not metricas:
        return

    def _factor(m):
        f = _nulo(m.get("factor_bonificacion"))
        return None if f is None else round(float(f), 4)

    filas = [(batch_id, m["kopr"], m["semanas_backtest"], m["semanas_para_mape"],
              m["semanas_censuradas"], _nulo(m["mape_semanal"]),
              _nulo(m["mae_semanal"]), _nulo(m["mape_4sem"]),
              m["bloques_4sem"], m["en_quiebre"], _factor(m)) for m in metricas]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO predictions.metrica_sku
                (batch_run_id, kopr, semanas_backtest, semanas_para_mape,
                 semanas_censuradas, mape_semanal, mae_semanal, mape_4sem,
                 bloques_4sem, en_quiebre, factor_bonificacion)
            VALUES %s
        """, filas)
    conn.commit()


# ============================================================================
# Inventario histórico: cierres semanales y ancla
# ============================================================================
# Una captura tomada a las 00:0x del día D refleja el cierre del día D-1. Por
# eso la captura que cierra una semana ISO es la del lunes de madrugada. El
# cierre se calcula a partir de la hora local, nunca se asume.

@dataclass(frozen=True)
class Ancla:
    semana: date              # lunes de la semana que la captura cierra
    capturado_en: datetime
    antiguedad_dias: int
    stock: dict               # kopr -> stock al cierre


def cierres_semanales(capturas: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Una fila (kopr, semana, stock_fin, capturado_en) por cierre semanal real."""
    cols = ["kopr", "semana", "stock_fin", "capturado_en"]
    if capturas is None or capturas.empty:
        return pd.DataFrame(columns=cols)
    df = capturas.copy()
    local = pd.to_datetime(df["capturado_en"], utc=True).dt.tz_convert(cfg.zona_horaria)
    df["capturado_en"] = local
    df["_fecha"] = local.dt.date
    df = df[(local.dt.weekday == 0) & (local.dt.hour < cfg.hora_madrugada_max)]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.sort_values("capturado_en").drop_duplicates(["kopr", "_fecha"], keep="first")
    df["semana"] = df["_fecha"].map(lambda f: f - timedelta(days=7))
    df = df.rename(columns={"stock_fisico": "stock_fin"})
    return df[cols].reset_index(drop=True)


def elegir_ancla_de(cierres: pd.DataFrame, cfg: Config,
                    ahora: datetime) -> Ancla | None:
    """Cierre más reciente con al menos `dias_minimos_ancla` de antigüedad."""
    if cierres is None or cierres.empty:
        log.info("Sin cierres semanales en stock_snapshot: no se reconstruye "
                 "stock histórico")
        return None
    limite = ahora - timedelta(days=cfg.dias_minimos_ancla)
    elegibles = cierres[cierres["capturado_en"] <= limite]
    if elegibles.empty:
        log.info("Ningún cierre semanal tiene %d días de antigüedad: no se "
                 "reconstruye stock histórico", cfg.dias_minimos_ancla)
        return None
    semana = max(elegibles["semana"])
    filas = elegibles[elegibles["semana"] == semana]
    capturado = filas["capturado_en"].max().to_pydatetime()
    antiguedad = (ahora - capturado).days
    log.info("Ancla de stock: semana %s, capturada %s, antigüedad %d días",
             semana, capturado.isoformat(), antiguedad)
    return Ancla(semana=semana, capturado_en=capturado, antiguedad_dias=antiguedad,
                 stock={str(k): float(v) for k, v in zip(filas["kopr"], filas["stock_fin"])})


def leer_capturas(conn_pg, cfg: Config = CONFIG) -> pd.DataFrame:
    with conn_pg.cursor() as cur:
        cur.execute("""
            SELECT kopr, stock_fisico, capturado_en
            FROM predictions.stock_snapshot
            WHERE bodega = %s
        """, (cfg.bodega,))
        filas = cur.fetchall()
    df = pd.DataFrame(filas, columns=["kopr", "stock_fisico", "capturado_en"])
    df["stock_fisico"] = df["stock_fisico"].astype(float)
    return df


def elegir_ancla(conn_pg, cfg: Config = CONFIG,
                 ahora: datetime | None = None) -> Ancla | None:
    ahora = ahora or datetime.now(ZoneInfo(cfg.zona_horaria))
    return elegir_ancla_de(cierres_semanales(leer_capturas(conn_pg, cfg), cfg), cfg, ahora)


def reconstruir_stock_historico(ancla: Ancla, movimientos: pd.DataFrame, cfg: Config,
                                inicio: date) -> pd.DataFrame:
    """Stock al cierre de cada semana, caminando hacia atrás desde la ancla.

    stock(t) = stock(t+1) - flujo_neto(t+1). Pura: sin conexiones. Los negativos
    se conservan (son la señal de calidad); los movimientos posteriores a la
    ancla se descartan.
    """
    cols = ["kopr", "semana", "stock_fin", "flujo_neto"]
    semanas = rango_semanas(inicio, ancla.semana)
    skus = sorted(ancla.stock)
    if movimientos is not None and not movimientos.empty:
        sin_captura = sorted(set(movimientos["kopr"].astype(str)) - set(skus))
        if sin_captura:
            log.warning("%d SKU con movimientos y sin captura en la ancla, no se "
                        "reconstruyen: %s", len(sin_captura), ", ".join(sin_captura[:20]))
    if not skus or not semanas:
        return pd.DataFrame(columns=cols)
    grilla = pd.DataFrame([(k, s) for k in skus for s in semanas], columns=["kopr", "semana"])
    if movimientos is not None and not movimientos.empty:
        mov = movimientos[["kopr", "semana", "flujo_neto"]].copy()
        mov["kopr"] = mov["kopr"].astype(str)
        mov["semana"] = pd.to_datetime(mov["semana"]).dt.date
        mov = mov.groupby(["kopr", "semana"], as_index=False)["flujo_neto"].sum()
        grilla = grilla.merge(mov, on=["kopr", "semana"], how="left")
    else:
        grilla["flujo_neto"] = 0.0
    grilla["flujo_neto"] = grilla["flujo_neto"].fillna(0.0).astype(float)
    grilla = grilla.sort_values(["kopr", "semana"]).reset_index(drop=True)
    # flujo posterior estricto = suma inversa acumulada - el propio flujo
    inv = grilla.groupby("kopr")["flujo_neto"].transform(lambda s: s[::-1].cumsum()[::-1])
    posterior = inv - grilla["flujo_neto"]
    grilla["stock_fin"] = grilla["kopr"].map(ancla.stock).astype(float) - posterior
    return grilla[cols]


def armar_serie_historica(reconstruido: pd.DataFrame, cierres: pd.DataFrame,
                          ancla: Ancla) -> pd.DataFrame:
    """Serie final (kopr, semana, stock_fin, flujo_neto, origen) ordenada.

    La ancla y las semanas posteriores con cierre real son 'capturado'; las
    anteriores a la ancla, 'reconstruido' aun si existe una captura real. El
    flujo de una semana posterior es la diferencia contra el stock_fin de la
    semana previa de la serie (0 si esa semana no está).
    """
    cols = ["kopr", "semana", "stock_fin", "flujo_neto", "origen"]
    partes = []
    if reconstruido is not None and not reconstruido.empty:
        rec = reconstruido[["kopr", "semana", "stock_fin", "flujo_neto"]].copy()
        rec["kopr"] = rec["kopr"].astype(str)
        es_ancla = rec["semana"] == ancla.semana
        rec.loc[es_ancla, "stock_fin"] = rec.loc[es_ancla, "kopr"].map(ancla.stock)
        rec["origen"] = ["capturado" if a else "reconstruido" for a in es_ancla]
        partes.append(rec[rec["semana"] <= ancla.semana])
    base = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=cols)
    previo = {(k, s): v for k, s, v in zip(base["kopr"], base["semana"], base["stock_fin"])}
    nuevas = []
    if cierres is not None and not cierres.empty:
        post = cierres[cierres["semana"] > ancla.semana].copy()
        post["kopr"] = post["kopr"].astype(str)
        post = post.sort_values(["kopr", "semana"]).drop_duplicates(["kopr", "semana"],
                                                                    keep="last")
        for r in post.itertuples():
            stock = float(r.stock_fin)
            ant = previo.get((r.kopr, r.semana - timedelta(days=7)))
            flujo = 0.0 if ant is None else stock - float(ant)
            previo[(r.kopr, r.semana)] = stock
            nuevas.append((r.kopr, r.semana, stock, flujo, "capturado"))
    if nuevas:
        base = pd.concat([base, pd.DataFrame(nuevas, columns=cols)], ignore_index=True)
    return base[cols].sort_values(["kopr", "semana"]).reset_index(drop=True)


def agregar_compras_salidas(serie: pd.DataFrame, movimientos: pd.DataFrame,
                            hasta: date) -> pd.DataFrame:
    """Agrega `compras` y `salidas` semanales a la serie histórica.

    Dentro de la ventana leída (`semana <= hasta`) una semana sin movimiento vale
    0; después queda NaN (ausencia, no cero). `compras` no baja de cero: una
    semana con más NCC que FCC no es una compra negativa.
    """
    out = serie.copy()
    cols = ["kopr", "semana", "compras", "salidas"]
    if movimientos is not None and len(movimientos):
        mov = movimientos[cols].copy()
        mov["kopr"] = mov["kopr"].astype(str)
        mov = mov.groupby(["kopr", "semana"], as_index=False)[["compras", "salidas"]].sum()
        out = out.merge(mov, on=["kopr", "semana"], how="left")
    else:
        out["compras"] = np.nan
        out["salidas"] = np.nan
    en_ventana = out["semana"] <= hasta
    for col in ("compras", "salidas"):
        out[col] = out[col].where(~en_ventana, out[col].fillna(0.0)).astype(float)
    out["compras"] = out["compras"].clip(lower=0.0)
    return out


def calidad_stock_historico(df: pd.DataFrame) -> dict:
    """Registra el control de calidad: las semanas con stock negativo."""
    total = len(df)
    neg = df[df["stock_fin"] < 0] if total else df
    for kopr, g in neg.groupby("kopr"):
        log.info("Stock negativo %s: %d semanas, peor %.2f, última %s",
                 kopr, len(g), g["stock_fin"].min(), g["semana"].max())
    pct = round(100.0 * len(neg) / total, 2) if total else 0.0
    log.info("Stock histórico: %d de %d semanas-SKU negativas (%.2f%%; referencia 3.7%%)",
             len(neg), total, pct)
    if not 3.0 <= pct <= 5.0:
        log.warning("Porcentaje de stock negativo %.2f%% fuera del rango esperado "
                    "3%%-5%% (referencia 3.7%%)", pct)
    return {"semanas": total, "negativas": len(neg), "pct": pct}


def armar_filas_serie(tratado: pd.DataFrame, resultados: list[dict]) -> list[tuple]:
    """Convierte las tres series al formato de predictions.serie_semanal."""
    filas: list[tuple] = []

    for r in tratado.itertuples():
        filas.append((r.kopr, r.semana, "real", round(float(r.demanda), 2),
                      None, None, round(float(r.demanda_cruda), 2),
                      bool(r.fue_tratada), bool(getattr(r, "censurada", False))))

    for res in resultados:
        for semana, valor in res["backtest"].items():
            if np.isfinite(valor):
                filas.append((res["kopr"], semana, "backtest",
                              round(float(valor), 2), None, None, None, None, None))
        if res["proyeccion"] is None:   # SKU sin stock: no se proyecta
            continue
        for p in res["proyeccion"].itertuples():
            filas.append((res["kopr"], p.semana, "proyeccion",
                          round(float(p.unidades), 2),
                          round(float(p.unidades_inf), 2),
                          round(float(p.unidades_sup), 2), None, None, None))
    return filas


# ============================================================================
# Main
# ============================================================================

def resumen(resultados: list[dict], cfg: Config) -> None:
    if not resultados:
        log.warning("No hubo SKUs procesados.")
        return
    m = pd.DataFrame([r["metricas"] | {"kopr": r["kopr"]} for r in resultados])
    # Se reporta la mediana y no la media: con MAPE, un solo SKU de demanda
    # baja distorsiona el promedio en un orden de magnitud.
    log.info("Resumen del backtest (%d semanas evaluadas por SKU; línea dibujada de %d a %d semanas):",
             cfg.semanas_evaluacion, int(m["semanas_dibujo"].min()), int(m["semanas_dibujo"].max()))
    log.info("  MAPE semanal    — mediana %.2f%% | SKUs sobre 100%%: %d de %d",
             m["mape_semanal"].median(), int((m["mape_semanal"] > 100).sum()), len(m))
    log.info("  MAPE %d semanas  — mediana %.2f%% | SKUs <= 20%%: %d de %d",
             cfg.bloque_acumulado, m["mape_4sem"].median(),
             int((m["mape_4sem"] <= 20).sum()), len(m))

    cens = int(m["semanas_censuradas"].sum())
    quiebre = m.loc[m["en_quiebre"], "kopr"].tolist()
    log.info("  Semanas censuradas por quiebre en el backtest: %d", cens)
    if quiebre:
        log.warning("  SKUs sin stock al cierre (el dashboard debe advertirlo): %s",
                    ", ".join(quiebre))


def resumen_cobertura(resultados: list[dict], stock: pd.DataFrame,
                      cfg: Config = CONFIG, factores: dict | None = None) -> None:
    """Para cuántas semanas alcanza el stock de cada SKU, incluyendo la salida
    por bonificado cuando hay factor.

    Es el mismo cálculo que hace la vista v_cobertura_actual; acá solo se
    reporta, para que la corrida avise de los productos en riesgo sin que
    nadie tenga que consultar la base.
    """
    if stock is None or stock.empty or not resultados:
        return

    por_sku = {r["kopr"]: float(r["proyeccion"]["unidades"].mean())
               for r in resultados if r["proyeccion"] is not None}
    filas = []
    for r in stock.itertuples():
        d = por_sku.get(r.kopr)
        if not d or d <= 0:
            continue
        f = ((factores or {}).get(r.kopr) or {}).get("factor") or 0.0
        filas.append((r.kopr, float(r.stock_fisico) / (d * (1 + f))))
    if not filas:
        return

    filas.sort(key=lambda f: f[1])
    mediana = float(np.median([f[1] for f in filas]))
    log.info("Cobertura de stock: mediana %.1f semanas en %d SKUs", mediana, len(filas))
    criticos = [f for f in filas if f[1] < cfg.cobertura_aviso]
    if criticos:
        log.warning("  Bajo %.0f semanas de cobertura: %s", cfg.cobertura_aviso,
                    ", ".join(f"{k} ({c:.1f})" for k, c in criticos))


def generar_stock_historico(conn_pg, batch_id: int, inicio: date, cfg: Config) -> int:
    """Reconstruye y guarda la serie histórica de stock. Devuelve las filas guardadas."""
    ancla = elegir_ancla(conn_pg, cfg)
    if ancla is None:
        log.warning("Stock histórico: no hay captura de cierre utilizable; se omite")
        return 0
    conn_pg.rollback()  # cierra la transacción de lectura antes de ir a Softland
    skus = set(cfg.skus)
    ancla = dc_replace(ancla, stock={k: v for k, v in ancla.stock.items() if k in skus})
    # Las semanas capturadas posteriores al ancla también llevan compras y
    # salidas; la reconstrucción descarta lo que pasa del ancla.
    hasta = max(ancla.semana, ultima_semana_cerrada())
    with conectar_mssql() as cx:
        movimientos = leer_movimientos_stock(cx, cfg, inicio, hasta)
    reconstruido = reconstruir_stock_historico(ancla, movimientos, cfg, inicio)
    calidad_stock_historico(reconstruido)
    cierres = cierres_semanales(leer_capturas(conn_pg, cfg), cfg)
    serie = armar_serie_historica(reconstruido, cierres, ancla)
    serie = agregar_compras_salidas(serie, movimientos, hasta)
    guardar_stock_historico(conn_pg, batch_id, cfg.bodega, serie)
    log.info("Stock histórico: %d filas guardadas (ancla %s)", len(serie), ancla.semana)
    return len(serie)


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
    stock = pd.DataFrame(columns=["kopr", "bodega", "stock_fisico"])

    # --- 1. Ventas y stock --------------------------------------------------
    if args.csv:
        log.info("Leyendo ventas semanales de %s (sin stock)", args.csv)
        semanal = leer_ventas_csv(args.csv)
        semanal = semanal[semanal["kopr"].isin(cfg.skus)]
        bonif_sem = None
        log.info("Sin datos de bonificado en el modo --desde-csv: el factor queda NULL")
    else:
        log.info("Leyendo ventas de Softland (%d meses, %d SKUs)",
                 cfg.meses_historia, len(cfg.skus))
        with conectar_mssql() as cx:
            diarias, bonificado = leer_ventas_mssql(cx, cfg)
            productos = leer_productos_mssql(cx, cfg)
            stock = leer_stock_mssql(cx, cfg)
        log.info("  %d filas diarias, %d productos en el maestro, "
                 "stock de %d SKUs en bodega %s, %d filas de bonificado",
                 len(diarias), len(productos), len(stock), cfg.bodega,
                 len(bonificado))
        semanal = semanalizar(diarias)
        bonif_sem = semanalizar_bonificado(bonificado)

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

    factores = calcular_factores_bonificacion(semanal, bonif_sem, ultima_semana_cerrada(), cfg)
    advertir_factores(factores, cfg)

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

    datos = marcar_censuradas(datos, cfg)
    n_cens = int(datos["censurada"].sum())
    log.info("Semanas marcadas como censuradas por quiebre: %d de %d (%.2f%%), "
             "umbral %.0f%% de la mediana previa",
             n_cens, len(datos), 100 * n_cens / len(datos), 100 * cfg.umbral_censura)

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
    for r in resultados:
        r["metricas"]["factor_bonificacion"] = (factores.get(r["kopr"]) or {}).get("factor")
    resumen(resultados, cfg)
    resumen_cobertura(resultados, stock, cfg, factores)

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
        guardar_stock(conn, batch_id, stock)
        guardar_serie(conn, batch_id, armar_filas_serie(datos, resultados))
        guardar_metricas(conn, batch_id, [r["metricas"] | {"kopr": r["kopr"]}
                                          for r in resultados])
        if cfg.reconstruir_stock and args.csv:
            log.info("Stock histórico omitido: --desde-csv no tiene conexión a Softland")
        elif cfg.reconstruir_stock:
            try:
                generar_stock_historico(conn, batch_id, inicio, cfg)
            except Exception:  # noqa: BLE001 — el histórico no debe tumbar la corrida
                log.exception("Stock histórico falló; la corrida %d sigue como ok", batch_id)
                try:  # una transacción abortada rompería cerrar_corrida
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    log.exception("rollback falló tras el error del stock histórico")
                if getattr(conn, "closed", 0):
                    log.warning("Conexión PostgreSQL cerrada; se reconecta para cerrar la corrida")
                    conn = conectar_postgres()
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