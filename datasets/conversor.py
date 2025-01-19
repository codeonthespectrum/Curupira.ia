import re
import pandas as pd

banco_palavras = pd.read_csv('datasets/base_tep2.csv', names=['PALAVRAS'], encoding='utf-8')
#print(banco_palavras.head())
banco_palavras['PALAVRAS'] = banco_palavras['PALAVRAS'].astype(str)

padrao = r"(\w+)] \{([\w\s,]+)\}" 
dados = []

for index, row in banco_palavras.iterrows():
    texto = row['PALAVRAS']
   # print(f"Processando texto: {texto}")

    for match in re.finditer(padrao, texto):
        categoria = match.group(1)
        palavras = match.group(2).split(',')
        for palavra in palavras:
            polaridade = "Negativa" if palavra in [ ] else "Positiva"
            dados.append([palavra, polaridade, categoria])

df = pd.DataFrame(dados, columns=['PALAVRAS', 'POLARIDADE', 'CATEGORIA'])
print(df)

df.to_csv('datasets/base_tep3.csv', index=False, encoding='utf-8')