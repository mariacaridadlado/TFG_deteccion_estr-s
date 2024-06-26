#lstm.py


#preprocesamiento_lstm.py
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pandas as pd
from scipy.signal import welch
import neurokit2 as nk
from pyts.approximation import SymbolicAggregateApproximation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):
    def __init__(self, hidden_size=200, num_layers=1):
        super(Net, self).__init__()
        self.fc1 = nn.Conv1d(in_channels=40, out_channels=6, kernel_size=10)
        self.fc2 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=3)
        self.fc3 = nn.Conv1d(in_channels=8, out_channels=10, kernel_size=1)
        self.fc4 = nn.Conv1d(in_channels=10, out_channels=40, kernel_size=1)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc5 = nn.Linear(hidden_size, 40)
        self.fc6 = nn.Linear(40, 20)
        self.fc7 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.fc1(x)), 5)
        x = F.dropout(x, 0.5)
        x = F.max_pool1d(F.relu(self.fc2(x)), 2)
        x = F.dropout(x, 0.5)
        x = F.max_pool1d(F.relu(self.fc3(x)), 2)
        x = F.dropout(x, 0.5)
        x = F.avg_pool1d(F.relu(self.fc4(x)), 2)
        x = F.dropout(x, 0.5)
        x, _ = self.lstm(x.unsqueeze(0))
        x = F.sigmoid(self.fc5(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5)
        x = self.fc7(x)
        return x


class MiDataset(Dataset):
  def __init__(self, data):

    self.train_data = torch.from_numpy(data.iloc[:,2:-1].values)
    self.train_labels = torch.from_numpy(data['Estrés'].values).float()
    self.n_samples= self.train_data.shape[0]

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx): #nos va a devolver el objeto en la posicion de idx
    return self.train_data[idx], self.train_labels[idx]
  
def cargar_datos(ruta, sujeto,columna_eda):
    archivo = ruta + str(sujeto) + ".csv"
    data = pd.read_csv(archivo)
    signal = data.iloc[:, columna_eda]

    return signal


def preprocesar_datos( sampling_rate,ruta, ruta_destino, columna_eda,sujetos=[2,3,4,5,6,7,8,9,10,11,13,14,15], window_length=30, overlap=0.1, subwindow_length=15):
    """
    Preprocesa los datos de una señal
    Parameters
    sampling_rate : int
        Frecuencia de muestreo de la señal EDA
    ruta : str
        Ruta donde se encuentran los archivos CSV
    ruta_destino : str
        Ruta donde se guardará el archivo CSV con los datos preprocesados
    columna_eda : int
        Número de columna que contiene la señal EDA en los archivos CSV
    sujetos : list
        Lista de sujetos a procesar
    window_length : int
        Longitud de la ventana en segundos
    overlap : float
        Segundo de superposición entre ventanas
    subwindow_length : int
        Longitud de la subventana en segundos
    """
    niveles = 2
    # Cargar los datos de la señal EDA
    for sujeto in sujetos:
        signal=cargar_datos(ruta,sujeto,columna_eda)


        fft_signal_original = np.fft.fft(signal)
        # Calcular la nueva longitud de la señal upsampled
        sampling_rate_upsampled = 64  # Hz
        upsampling_factor = sampling_rate_upsampled / sampling_rate
        new_length = int(len(signal) * upsampling_factor)
    
        # Interpolar los coeficientes de la DFT para aumentar la resolución en el dominio de la frecuencia
        fft_signal_upsampled = np.zeros(new_length, dtype=complex)
        fft_signal_upsampled[:len(fft_signal_original)//2] = fft_signal_original[:len(fft_signal_original)//2]
        fft_signal_upsampled[-len(fft_signal_original)//2:] = fft_signal_original[-len(fft_signal_original)//2:]

        # Aplicar la Transformada Inversa de Fourier Discreta (IDFT)
        signal_upsampled = np.fft.ifft(fft_signal_upsampled)

        # Tomar solo la parte real de la señal upsampled (los valores complejos pueden ser pequeñas aproximaciones numéricas)
        signal_upsampled = np.real(signal_upsampled)

    
        signals, info = nk.eda_process(signal_upsampled, sampling_rate=sampling_rate_upsampled)

    
        # Se aplica SAX directamente a la señal normalizada
        sax = SymbolicAggregateApproximation(n_bins=niveles, strategy='uniform')
        serie_sax = sax.fit_transform(signals["EDA_Clean"].values.reshape(1, -1))[0]

        categorias_estres = []
        # Asignar la categoría de estrés (0 o 1)
        categorias_estres = [0 if valor == 'a' else 1 for valor in serie_sax]

        # Crear un DataFrame con los datos EDA clean y la columna de estrés
        data = pd.DataFrame({'Señal': signals["EDA_Clean"], 'Estrés': categorias_estres})

        # Calcular si hay más 0 o 1 en cada fila
        data['Estrés'] = (data['Estrés'] == 0).astype(int)

        df=data
        # Calcular el número de muestras por ventana y el desplazamiento
        window_samples = window_length * sampling_rate_upsampled
        subwindow_samples = subwindow_length * sampling_rate_upsampled
        overlap_samples = int(overlap * sampling_rate_upsampled)
     #  Lista para almacenar los valores espectrales promedio de cada ventana
        average_spectral_values = []
        stress_values = []
        window_numbers = []

   # Iterar sobre las ventanas
        for i in range(0, len(df), overlap_samples):
            # Obtener los datos de la ventana actual
            window_data = df.iloc[i:i+window_samples]

        # Lista para almacenar los espectros de las subventanas
            subwindow_spectra = []

        # Iterar sobre las subventanas dentro de la ventana actual
            for j in range(0, window_samples - subwindow_samples + 1, overlap_samples):
                if (len(window_data.iloc[j:j+subwindow_samples])<960):
                    continue
            # Obtener los datos de la subventana actual
                subwindow_data = window_data.iloc[j:j+subwindow_samples]

         # Calcular el espectro de frecuencias utilizando Welch para la subventana
                freqs, psd = welch(subwindow_data['Señal'], fs=sampling_rate)

        # Agregar el espectro de la subventana a la lista
                subwindow_spectra.append(psd)

    # Calcular el espectro promedio de todas las subventanas
            average_spectrum = np.mean(subwindow_spectra, axis=0)

            if np.isnan(average_spectrum).any():
                continue

    # Agregar el espectro promedio a la lista de valores espectrales promedio
            average_spectral_values.append(average_spectrum)


    # Obtener el valor de estrés de la ventana actual
            stress_value = window_data['Estrés'].iloc[0]  # Suponemos que el valor de estrés es el mismo para toda la ventana
            stress_values.append(stress_value)

    # Obtener el número de ventana
            window_number = int(i / overlap_samples) + 1
            window_numbers.append(window_number)
    
    # Convertir las listas a Series para poder agregarlas al DataFrame final
        stress_series = pd.Series(stress_values, name='Estrés')
        window_number_series = pd.Series(window_numbers, name='Número de ventana')

# Convertir la lista de valores espectrales promedio a un DataFrame
        spectral_df = pd.DataFrame(average_spectral_values)

# Concatenar el DataFrame de valores espectrales con las Series de estrés y número de ventana
        final_df = pd.concat([stress_series, window_number_series, spectral_df.astype(float)], axis=1)
        final_df['id']=sujeto
        print(final_df.head())

        if sujeto == sujetos[0]:
            final_df.to_csv(ruta_destino , index=False)
        else:
            final_df.to_csv(ruta_destino , mode='a', header=False, index=False)

def crear_modelo_lstm(hidden_size=200, num_layers=1):
    """
    Crea un modelo LSTM
    Returns
    -------
    modelo : nn.Module
        Modelo LSTM

    """
    modelo = Net(hidden_size=hidden_size, num_layers=num_layers)
    return modelo

def entrenar_modelo(data,modelo, sujetos, n_epochs=10, batch_size=40, learning_rate=0.001):
    """
    Entrena un modelo LSTM para cada sujeto y calcula la exactitud
    Parameters
    ----------
    data : DataFrame
        Datos preprocesados
    modelo : nn.Module
        Modelo LSTM 
    sujetos : list
        Lista de sujetos a procesar
    n_epochs : int
        Número de épocas
    batch_size : int
        Tamaño del lote
    learning_rate : float
        Tasa de aprendizaje
    Returns
    -------
    modelo : nn.Module
        Modelo LSTM entrenado
    media_resultados : float
        Exactitud media
    """
    resultados = []
    #comprobar si el ordenador sopoorta cuda
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    modelo.to(device)

    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for i in sujetos:
        train_data = data[data['id'] != i]
        test_data = data[data['id'] == i]

        train_dataset = MiDataset(train_data)
        test_dataset = MiDataset(test_data)

        trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        for epoch in range(n_epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            modelo.train()
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = modelo(inputs.float())
                outputs = torch.sigmoid(outputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calcular el accuracy
                predicted = torch.round(outputs)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = train_correct / train_total

            if epoch == n_epochs - 1:
                resultados.append(train_accuracy)
                print(f"Sujeto: {i} Train Acc: {train_accuracy:.4f}")

    media_resultados = sum(resultados) / len(resultados)
    return modelo, media_resultados

#añadir uno de carga de modelo ya entrenado, con nuevos datos 
def cargar_modelo(ruta_modelo):
    """
    Carga un modelo LSTM previamente entrenado
    Parameters
    ----------
    ruta_modelo : str
        Ruta del modelo entrenado
    Returns
    -------
    modelo : nn.Module
        Modelo LSTM
    """
    modelo = Net()
    modelo.load_state_dict(torch.load(ruta_modelo))
    return modelo

#funcion que recibe el modelo y los datos de un sujeto y devuelve la predicción y el accuracy
def evaluar_modelo(modelo, data, sujeto, batch_size=40, criterion=nn.BCELoss()):
    """
    Evalúa un modelo LSTM en un sujeto y calcula la exactitud
    Parameters
    ----------
    modelo : nn.Module
        Modelo 
    data : DataFrame
        Datos preprocesados
    sujeto : int
        Sujeto a evaluar
    batch_size : int
        Tamaño del lote
    criterion : nn.Module
        Función de pérdida
    Returns
    -------
    predicted : Tensor
        Predicciones
    test_accuracy : float
        Exactitud

    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #comprobar si el ordenador sopoorta el device que le pasamos


    test_data = data[data['id'] == sujeto]
    test_dataset = MiDataset(test_data)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    modelo.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = modelo(inputs.float())
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Calcular el accuracy
            predicted = torch.round(outputs)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_total
    return predicted, test_accuracy






