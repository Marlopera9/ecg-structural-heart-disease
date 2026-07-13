import pandas as pd
import ast


def convertir_si_es_texto(valor):
    """Convierte un string tipo diccionario en un diccionario real de Python.
    Si ya es un diccionario (por ejemplo, si la celda se ejecutó dos veces),
    lo devuelve tal cual, sin fallar."""
    if isinstance(valor, str):
        return ast.literal_eval(valor)
    return valor


def obtener_superclases(codigos_dict, agg_df):
    """Dado el diccionario de códigos diagnósticos de un ECG, devuelve
    la lista de superclases (NORM, MI, STTC, CD, HYP) a las que pertenece."""
    superclases = set()
    for codigo in codigos_dict.keys():
        if codigo in agg_df.index:
            superclases.add(agg_df.loc[codigo, 'diagnostic_class'])
    return list(superclases)


def asignar_split(fold):
    """Traduce el número de strat_fold oficial de PTB-XL a train/val/test,
    evitando así cualquier fuga de datos entre pacientes."""
    if fold <= 8:
        return 'train'
    elif fold == 9:
        return 'val'
    else:
        return 'test'


def cargar_metadatos_limpios(ruta_data='../data'):
    """
    Carga los metadatos de PTB-XL y aplica todas las decisiones de limpieza
    documentadas en el README (Fase 1):
      - Excluye registros sin ninguna superclase diagnóstica clara
      - Excluye registros de pacientes con marcapasos
      - Asigna la columna 'split' (train/val/test) según el strat_fold oficial
    """
    df = pd.read_csv(f'{ruta_data}/ptbxl_database.csv', index_col='ecg_id')
    agg_df = pd.read_csv(f'{ruta_data}/scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df['diagnostic'] == 1]

    df['scp_codes'] = df['scp_codes'].apply(convertir_si_es_texto)
    df['superclases_diagnosticas'] = df['scp_codes'].apply(
        lambda x: obtener_superclases(x, agg_df)
    )
    df['num_superclases'] = df['superclases_diagnosticas'].apply(len)

    df = df[df['num_superclases'] > 0].copy()
    df = df[df['pacemaker'].isna()].copy()

    df['split'] = df['strat_fold'].apply(asignar_split)

    return df