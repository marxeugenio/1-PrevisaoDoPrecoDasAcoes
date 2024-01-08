import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def prepare_data(df, forecast_col, forecast_out, test_size):
    df['label'] = df[forecast_col].shift(-forecast_out)
    df.dropna(inplace=True)
    
    # Convertendo os valores de string para float
    df[forecast_col] = df[forecast_col].apply(lambda x: float(x.replace(',', '')))
    
    x = np.array(df[[forecast_col]])
    x = preprocessing.scale(x)
    
    x_lately = x[-forecast_out:]
    x = x[:-forecast_out]
    
    label = np.array(df['label'])
    
    # Convertendo os valores de string para float
    label = np.array([float(val.replace(',', '')) for val in label])
    
    x_train, x_test, y_train, y_test = train_test_split(x, label[:-forecast_out], test_size=test_size, random_state=0)

    response = [x_train, x_test, y_train, y_test, x_lately]
    return response

# Carregando os dados (substitua pelo caminho correto para seu arquivo CSV)
df = pd.read_csv(r"C:\Users\marxe\OneDrive\Área de Trabalho\170 Projetos\1. Previsão do preço das ações\BitcoinData.csv")

forecast_col = "Price"
forecast_out = 5
test_size = 0.2 

# Chamando o método para preparar os dados
x_train, x_test, y_train, y_test, x_lately = prepare_data(df, forecast_col, forecast_out, test_size)

# Inicializando o modelo de regressão linear
learner = LinearRegression()

# Treinando o modelo
learner.fit(x_train, y_train)

# Avaliando o modelo
score = learner.score(x_test, y_test)

# Fazendo previsões
forecast = learner.predict(x_lately)

# Criando um objeto JSON com os resultados
response = {}
response["test_score"] = score
response["forecast_set"] = forecast

print(response)
