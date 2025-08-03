import pandas as pd

def carregar_dados(caminho):
    df = pd.read_csv(caminho, encoding='latin1', sep=None, engine='python')
    df.rename(columns=lambda x: x.strip(), inplace=True)

    return df
