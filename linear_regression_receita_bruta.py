import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df = pd.read_excel("conjunto_dataset.xlsx")

# Correlação entre variáveis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlação entre Variáveis")
plt.show()

for col in df.columns:
    if col not in ["Ano", "Receita_Bruta_Realizada"]:
        plt.figure()
        sns.scatterplot(data=df, x=col, y="Receita_Bruta_Realizada")
        plt.title(f"Relação entre {col} e Receita_Bruta_Realizada")
        plt.show()


# Carregar os dados
df = pd.read_excel("conjunto_dataset.xlsx")

# Definir features e target
X = df[['Receita_Corrente', 'Receita_Impostos', 'Despesa_Empenhada_Corrente',
        'Despesa_Empenhada_Capital', 'Despesa_Paga_Corrente']]
y = df['Receita_Bruta_Realizada']
anos = df['Ano']



# Dividir treino e teste (por exemplo, até 2020 para treino)
X_train = X[df['Ano'] <= 2020]
X_test = X[df['Ano'] > 2020]
y_train = y[df['Ano'] <= 2020]
y_test = y[df['Ano'] > 2020]
anos_test = anos[df['Ano'] > 2020]

# Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Prever e avaliar
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Erro médio absoluto:", mae)

# Mostrar valores reais vs previstos
for ano, real, previsto in zip(anos_test, y_test, y_pred):
    error = abs(real - previsto)
    print(f"Ano: {ano} | Real: {real:,.0f} | Previsto: {previsto:,.0f} | Erro: {error:,.0f}")

# Plotar gráfico
plt.plot(anos_test, y_test, label='Real', marker='o')
plt.plot(anos_test, y_pred, label='Previsto', marker='x')
plt.title('Previsão de Receita Bruta Realizada')
plt.xlabel('Ano')
plt.ylabel('R$')
plt.legend()
plt.grid(True)
plt.xticks(anos_test)
plt.show()


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
print(f"MAPE: {mape:.2f}%")