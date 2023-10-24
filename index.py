# Descricao: Esse programa detecta se um email é SPAM (1), ou não(0)

import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix

def processaTexto(texto):
    #remove pontuação caractere a caractere
    nopunc = [char for char in texto if char not in string.punctuation]
    #junta os caracteres em palavras novamente
    nopunc = ''.join(nopunc)
    cleaWords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return cleaWords

# nltk.download() 

messages = pd.read_csv("Projetos\Spam-Detector\spam.csv", encoding='latin-1') # Chamando a Função do Pandas para ler um arquivo CSV

messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True) # Chamando a Função drop para apagar as colunas descritas

messages = messages.rename(columns={'v1': 'tipo','v2': 'mensagem'})

# Imprime tudo, no caso todas as linhas do arquivo csv
#print(messages.to_string())

# Exibe count (total de mensagens), quantas são únicas (ou seja, tem valores repetidos)
# top (a mensagem que mais se repete) e freq (quantas vezes a mensagem top repete)
# print(messages.groupby('tipo').describe().iloc[0])
# print(messages.groupby('tipo').describe().iloc[1])

messages['length'] = messages['mensagem'].apply(len) # Calcula o número de caracteres de cada mensagem de texto e armazena o resultado na coluna 'leigth'.

# messages.hist(column='length', by = 'tipo', bins=70, figsize=(15,6)) # Desenha um gráfico do número de mensagens pelo tamanho

#messages['mensagem'].apply(processaTexto)

#print(messages.iloc[:,1].head()) # Exibe as cinco primeiras mensagens de texto da coluna 'mensagem' do DataFrame 'messages'

# Divide a base de dados em conjuntos de treinamento e teste.
# msg_train e msg_test são os conjuntos de treinamento e teste de mensagens.
# class_train e class_test são os rótulos correspondentes.
# O parâmetro 'test_size' controla o tamanho do conjunto de teste, no caso o 0.1 equivale a 10%.
msg_train, msg_test, class_train, class_test = train_test_split(messages['mensagem'], messages['tipo'], test_size=0.1)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer = processaTexto)), # 'bow' (Bag of Words): conta a ocorrência de cada palavra no vocabulário.
    ('tfidf', TfidfTransformer()), # 'tfidf' (Term Frequency-Inverse Document Frequency): Normaliza a contagem das palavras.
    ('classifier', MultinomialNB()) # 'classifier' (Classificador): Treina esses vetores no classificador naive bayes
])

# msg_test.iloc[0] = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"

pipeline.fit(msg_train, class_train)

preditions = pipeline.predict(msg_test)

# Exibe todas as mensagens marcadas como "spam"

print(classification_report(class_test, preditions))

