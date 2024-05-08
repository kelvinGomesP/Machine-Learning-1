import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch import optim
import tqdm


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


def fill_in_age(x):
    if x["Pclass_1"] == 1:
        return 35
    elif x["Pclass_2"] == 1:
        return 28
    else:
        return 25


class configura(data.Dataset):
    def __init__(self, filename):
        # converter para tensor/ converte para numpy
        data_matrix = filename.values
        data_matrix = torch.from_numpy(data_matrix).to(DEVICE)
        # extrai para tensor
        self.data = torch.index_select(
            data_matrix, dim=1, index=torch.arange(1, 13).to(DEVICE)
        )
        self.data = self.data.float()
        self.target = data_matrix[:, 0]  # coluna 1 sai verdadeira
        self.n_samples = self.data.shape[0]  # numero de amostras

    def __len__(self):  # Comprimento do conjunto de dados
        return self.n_samples

    def __getitem__(self, index):  # Retorna uma amostra de dados e seu rótulo
        # Retorna os dados e os rótulos como tensores do PyTorch
        return self.data[index], self.target[index]


class modelo(nn.Module):
    def __init__(self, neuro_in=12, neuro_hid=6, neuro_out=2):  # Alteração aqui
        super(modelo, self).__init__()
        self.n_in = neuro_in
        self.n_hid = neuro_hid
        self.n_out = neuro_out
        self.linearlinear = nn.Sequential(
            nn.Linear(neuro_in, neuro_hid),
            # Definindo a primeira camada linear com 12 entradas e 4 saídas
            nn.ReLU(),  # Alteração aqui
            nn.Linear(neuro_hid, neuro_out),
            # Ajustando a segunda camada linear para receber 4 entradas
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linearlinear(x)
        return x


class test_data(data.Dataset):
    def __init__(self, filename):
        # Carregar os dados do arquivo
        data_matrix = filename.values

        # Converter para tensor do PyTorch
        data_matrix = torch.from_numpy(data_matrix).to(DEVICE)

        # Selecionar as colunas de 1 a 12 (excluindo a coluna 0, que geralmente é o índice)
        self.data = torch.index_select(
            data_matrix, dim=1, index=torch.arange(2, 14).to(DEVICE)
        )

        # Converter os dados para o tipo float
        self.data = self.data.float()

        self.target = data_matrix[:, 1] 

        # Calcular o número de amostras
        self.n_samples = self.data.shape[0]

    def __len__(self):
        # Retornar o número de amostras
        return self.n_samples

    def __getitem__(self, index):
        # Retorna uma amostra de dados e seu ID de passageiro correspondente para o índice especificado
        return self.data[index], self.target[index]


if __name__ == "__main__":
    df1 = pd.read_csv("test.csv")
    df2 = pd.read_csv("train.csv")
    df3 = pd.read_csv("gender_submission.csv")

    categorica1 = ["Pclass", "Sex", "Embarked"]

    # Primeiro, convertemos as colunas categóricas em índices inteiros
    for col in categorica1:
        df2[col] = pd.Categorical(df2[col])
        df2[col] = df2[col].cat.codes

    # Em seguida, aplicamos a codificação one-hot usando torch.nn.functional.one_hot
    # Vamos criar uma cópia do dataframe original para manter as colunas originais intactas
    df_cat1 = df2.copy()

    # Lista das colunas categóricas após a conversão em índices inteiros
    categorical_cols_indices = [df_cat1.columns.get_loc(col) for col in categorica1]

    # Aplicando a codificação one-hot
    for col_index in categorical_cols_indices:
        # Obtendo os valores únicos na coluna categórica
        unique_values = df_cat1.iloc[:, col_index].unique()

        # Mapeando cada valor único para um índice
        value_to_index = {val: idx for idx, val in enumerate(unique_values)}

        # Convertendo os valores para seus respectivos índices
        indexed_values = df_cat1.iloc[:, col_index].map(value_to_index)

        # Aplicando a codificação one-hot
        one_hot_encoded = F.one_hot(torch.tensor(indexed_values), num_classes=len(unique_values))
        
        # Adicionando as colunas one-hot ao dataframe
        for i, col_value in enumerate(unique_values):
            df_cat1[f'{df_cat1.columns[col_index]}_{col_value}'] = one_hot_encoded[:, i]

    # Agora podemos remover as colunas categóricas originais
    df_cat1.drop(categorica1, axis=1, inplace=True)

    print("Soma de nulos idade:", df_cat1["Age"].isnull().sum())

    columns_to_drop = [
        "PassengerId",
        "Cabin",
        "Pclass",
        "Name",
        "Sex",
        "Ticket",
        "Embarked",
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df_cat1.columns]

    df_treino = df_cat1.drop(columns_to_drop, axis=1)

    df_treino["Age"] = df_treino.apply(fill_in_age, axis=1)

    scaler = StandardScaler()
    df_treino[["Age", "Fare"]] = scaler.fit_transform(df_treino[["Age", "Fare"]])

    train = configura(df_treino)

    # Uma função de ativação ReLU (nn.ReLU), que introduz não linearidade após a primeira camada linear.
    # Uma segunda camada linear que recebe n_hid entradas e produz n_out saídas.
    # Uma função de ativação sigmóide (nn.Sigmoid), que é usada para garantir que a saída da rede esteja no intervalo [0, 1].

    # Otimizadores

    # Criando um DataLoader para carregar os dados em lotes
    train_loader = data.DataLoader(train, batch_size=200, shuffle=True)

    # Instanciando o modelo
    # Definindo o número correto de neurônios de entrada (12)
    model = modelo(neuro_in=12).to(DEVICE)

    # Definindo a função de perda
    criterion = nn.BCELoss()

    # Definindo o otimizador
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Treinamento do modelo
    epochs = 100
    for e in range(epochs):
        print(f"Época {e+1}")
        running_loss = 0  # Inicializa a perda acumulada para esta época

        for info, labels in tqdm.tqdm(train_loader, total=len(train_loader)):
            dummy_labels = labels.float()
            # Ajustando o redimensionamento dos rótulos para corresponder ao número de saídas do modelo
            dummy_labels = F.one_hot(dummy_labels.to(torch.int64), num_classes=2).float()

            optimizer.zero_grad()

            # Chamando o método forward do modelo para obter as previsões
            output = model(info)

            # Calculando a perda
            loss = criterion(output, dummy_labels)

            # Retropropagação e atualização dos pesos
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Ao final de cada época, imprime a perda média
        print(f"Loss média na época {e+1}: {running_loss / len(train_loader)}")

    categorica2 = ["Pclass", "Sex", "Embarked"]

    # Primeiro, convertemos as colunas categóricas em índices inteiros
    for col in categorica2:
        df2[col] = pd.Categorical(df2[col])
        df2[col] = df2[col].cat.codes

    # Em seguida, aplicamos a codificação one-hot usando torch.nn.functional.one_hot
    # Vamos criar uma cópia do dataframe original para manter as colunas originais intactas
    df_cat2 = df2.copy()

    # Lista das colunas categóricas após a conversão em índices inteiros
    categorical_cols_indices = [df_cat2.columns.get_loc(col) for col in categorica2]

    # Aplicando a codificação one-hot
    for col_index in categorical_cols_indices:
        # Obtendo os valores únicos na coluna categórica
        unique_values = df_cat2.iloc[:, col_index].unique()

        # Mapeando cada valor único para um índice
        value_to_index = {val: idx for idx, val in enumerate(unique_values)}

        # Convertendo os valores para seus respectivos índices
        indexed_values = df_cat2.iloc[:, col_index].map(value_to_index)

        # Aplicando a codificação one-hot
        one_hot_encoded = F.one_hot(torch.tensor(indexed_values), num_classes=len(unique_values))
        
        # Adicionando as colunas one-hot ao dataframe
        for i, col_value in enumerate(unique_values):
            df_cat2[f'{df_cat2.columns[col_index]}_{col_value}'] = one_hot_encoded[:, i]

    # Agora podemos remover as colunas categóricas originais
    df_cat2.drop(categorica2, axis=1, inplace=True)

    # Definir as colunas a serem excluídas
    colunas_para_deletar = ["Cabin", "Name", "Ticket"]
    df_teste = df_cat2.drop(columns=colunas_para_deletar)
    df_teste["Age"] = df_teste.apply(fill_in_age, axis=1)

    test_data = test_data(df_teste)
    test_loader = data.DataLoader(test_data, batch_size=200, num_workers=0)

    print("Teste do modelo")
    accuracy = []
    for info, labels in tqdm.tqdm(test_loader, total=len(test_loader)):
        output = model(info)
        output = torch.sigmoid(output)  # Aplicando a função sigmoid para obter probabilidades entre 0 e 1
        output = (output > 0.5).int()  # Convertendo as probabilidades para valores binários (0 ou 1)
        
        # Redimensionando os rótulos para valores entre 0 e 1
        labels = labels.float() / labels.max()
        
        accuracy += [torch.sum(output == labels.view(-1, 1)).item() / len(labels)]  # Calculando a acurácia
    print(f"Acuracia no teste: {np.mean(accuracy) * 100:.2f}%")


