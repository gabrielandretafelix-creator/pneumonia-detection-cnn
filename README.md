# 🫁 Detecção Experimental de Pneumonia com Redes Neurais Convolucionais
# Experimental Pneumonia Detection Using Convolutional Neural Networks

**Projeto acadêmico de Inteligência Artificial e Visão Computacional**  
**Academic project in Artificial Intelligence and Computer Vision**

[🇧🇷 Português](#português) | [🇺🇸 English](#english)

---

# Português

## 📌 Sobre o projeto

Este projeto acadêmico tem como objetivo explorar a classificação de imagens de raio-X de tórax por meio de técnicas de Inteligência Artificial e Deep Learning.

O sistema utiliza uma rede neural baseada na arquitetura **EfficientNetB0**, desenvolvida com TensorFlow e Keras, para classificar imagens nas categorias Normal e Pneumonia.

O projeto também possui uma interface gráfica desenvolvida com Streamlit, permitindo carregar imagens, visualizar as previsões do modelo e explorar recursos de interpretabilidade.

## ✨ Funcionalidades

- Classificação experimental de imagens de raio-X de tórax.
- Utilização da arquitetura EfficientNetB0 com TensorFlow/Keras.
- Interface interativa desenvolvida com Streamlit.
- Pré-processamento e redimensionamento de imagens.
- Test-Time Augmentation (TTA) para combinar previsões de diferentes versões da imagem.
- Visualização de mapas de calor utilizando Grad-CAM.
- Avaliação do modelo com métricas de classificação, matriz de confusão e curvas ROC e Precision-Recall.
- Registro local das análises e geração de relatórios em PDF.

## 🛠️ Tecnologias utilizadas

- **Python** — linguagem de programação.
- **TensorFlow e Keras** — desenvolvimento e execução da rede neural.
- **EfficientNetB0** — arquitetura utilizada no modelo.
- **Streamlit** — interface gráfica interativa.
- **MedMNIST/PneumoniaMNIST** — conjunto de dados utilizado nos experimentos.
- **NumPy e Pandas** — manipulação e organização de dados.
- **Scikit-learn** — métricas e avaliação do modelo.
- **Matplotlib e Seaborn** — gráficos e visualização de resultados.
- **Pillow** — processamento de imagens.

## 🗂️ Conjunto de dados

O projeto utiliza o **PneumoniaMNIST**, integrante do conjunto MedMNIST, para treinamento e avaliação experimental.

Mais informações:
https://medmnist.com/

## 🚀 Como executar

### 1. Pré-requisitos

- Python instalado.
- Acesso à internet para instalar as dependências e baixar os dados necessários.

### 2. Instalar as dependências

Abra o terminal na pasta do projeto e execute:

```powershell
py -m pip install -r requirements.txt
```

### 3. Treinar ou avaliar o modelo

Execute o script de treinamento:

```powershell
py pneumonia_model_v2.py
```

O script pode treinar o modelo quando necessário ou carregar um modelo previamente salvo para realizar a avaliação. O treinamento pode exigir tempo de processamento e recursos computacionais.

### 4. Executar a interface gráfica

Disponibilize o arquivo `Pneumonia-Model-melhor.keras` na mesma pasta do aplicativo e execute:

```powershell
py -m streamlit run app.py
```

O Streamlit abrirá a aplicação no navegador. A execução depende da presença do modelo treinado e da instalação das bibliotecas necessárias.

## 🎯 Objetivo acadêmico

O projeto permite estudar redes neurais convolucionais, transferência de aprendizado, classificação de imagens, avaliação de modelos e interpretabilidade em Inteligência Artificial.

Também representa uma aplicação prática dos conhecimentos de programação em Python e desenvolvimento de aplicações interativas.

## ⚠️ Limitações e aviso importante

Este sistema possui finalidade acadêmica e experimental. As previsões podem conter erros e não devem ser utilizadas isoladamente para diagnosticar doenças, orientar tratamentos ou tomar decisões clínicas.

O projeto não substitui a avaliação de profissionais de saúde e não foi validado para uso clínico ou hospitalar. O desempenho em um conjunto de dados experimental não garante resultados equivalentes em ambientes reais.

Não utilize dados pessoais ou informações reais de pacientes em demonstrações públicas.

## 👨‍💻 Autor

**Gabriel Felix Andreta**

Estudante de Ciência da Computação, com interesse em programação, Inteligência Artificial, Deep Learning e visão computacional.

---

# English

## 📌 About the project

This academic project explores chest X-ray image classification using Artificial Intelligence and Deep Learning techniques.

The system uses a neural network based on the **EfficientNetB0** architecture, developed with TensorFlow and Keras, to classify images into two categories: Normal and Pneumonia.

The project also includes an interactive graphical interface developed with Streamlit, allowing users to upload images, view model predictions, and explore interpretability features.

## ✨ Features

- Experimental classification of chest X-ray images.
- EfficientNetB0 architecture implemented with TensorFlow/Keras.
- Interactive graphical interface built with Streamlit.
- Image preprocessing and resizing.
- Test-Time Augmentation (TTA) to combine predictions from different image variations.
- Heatmap visualization using Grad-CAM.
- Model evaluation with classification metrics, a confusion matrix, ROC curves, and Precision-Recall curves.
- Local analysis history and PDF report generation.

## 🛠️ Technologies

- **Python** — programming language.
- **TensorFlow and Keras** — neural network development and execution.
- **EfficientNetB0** — model architecture.
- **Streamlit** — interactive graphical interface.
- **MedMNIST/PneumoniaMNIST** — dataset used in the experiments.
- **NumPy and Pandas** — data manipulation and organization.
- **Scikit-learn** — model evaluation and metrics.
- **Matplotlib and Seaborn** — charts and result visualization.
- **Pillow** — image processing.

## 🗂️ Dataset

The project uses **PneumoniaMNIST**, part of the MedMNIST collection, for experimental training and evaluation.

Learn more:
https://medmnist.com/

## 🚀 How to run the project

### 1. Prerequisites

- Python installed.
- Internet access to install dependencies and download the required data.

### 2. Install dependencies

Open a terminal in the project directory and run:

```powershell
py -m pip install -r requirements.txt
```

### 3. Train or evaluate the model

Run the training script:

```powershell
py pneumonia_model_v2.py
```

The script can train the model when necessary or load a previously saved model for evaluation. Training may require additional processing time and computational resources.

### 4. Run the graphical interface

Place the `Pneumonia-Model-melhor.keras` file in the same directory as the application, then run:

```powershell
py -m streamlit run app.py
```

Streamlit will open the application in a web browser. Execution requires the trained model file and the necessary Python dependencies.

## 🎯 Academic purpose

This project provides an opportunity to explore convolutional neural networks, transfer learning, image classification, model evaluation, and interpretability in Artificial Intelligence.

It also demonstrates the practical application of Python programming and interactive application development.

## ⚠️ Limitations and disclaimer

This system is intended for academic and experimental purposes only. Its predictions may be incorrect and must not be used on their own to diagnose diseases, recommend treatments, or make clinical decisions.

The project does not replace professional medical evaluation and has not been validated for clinical or hospital use. Performance on an experimental dataset does not guarantee equivalent results in real-world environments.

Do not use personal information or real patient data in public demonstrations.

## 👨‍💻 Author

**Gabriel Felix Andreta**

Computer Science student interested in programming, Artificial Intelligence, Deep Learning, and computer vision.

---

*Academic project — Artificial Intelligence and Computer Vision.*  
*Projeto acadêmico — Inteligência Artificial e Visão Computacional.*
