<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo_sctecB.png">
    <source media="(prefers-color-scheme: light)" srcset="img/logo_sctecW.png">
    <img alt="Logo SCTEC" src="img/logo_sctecW.png" width="200">
  </picture>
</p>

# 🧠 MNIST - Pipeline de Treinamento e Uso com V1, V2 e V3

Este projeto realiza o treinamento e a comparação de modelos para classificação de dígitos manuscritos do dataset MNIST usando diferentes pipelines de processamento e três abordagens de modelos:

- Random Forest
- SVM (RBF)
- CNN

Além disso, o projeto inclui:

- geração de gráficos de análise do dataset
- comparação de métricas por modelo
- visualização de matrizes de confusão
- teste de out-of-distribution (OOD)
- aplicação desktop para inferência em imagens digitais

---

## 📁 Estrutura do projeto

```text
.
├── app.py
├── main.py
├── README.md
├── images/
│   ├── V1/
│   ├── V2/
│   └── V3/
├── models/
│   ├── V1/
│   ├── V2/
│   └── V3/
│       ├── cnn_mnist.keras
│       ├── metadata.json
│       ├── random_forest_mnist.joblib
│       └── svm_mnist.joblib
└── src/
    ├── install.py
    ├── pipeline.py
    └── testOOD.py
```

---

## ⚙️ Como funciona o projeto

### 1. Treinamento principal
O arquivo principal do treinamento é o [main.py](main.py).

Ele executa, em sequência:

1. carregamento do dataset MNIST
2. visualização do balanceamento das classes
3. separação treino/teste
4. processamento de imagens conforme a versão escolhida
5. treinamento dos modelos:
   - Random Forest
   - SVM
   - CNN
6. avaliação comparativa com métricas e matrizes de confusão
7. teste OOD para verificar robustez em situações fora do domínio

### 2. App para uso em imagens
O arquivo [app.py](app.py) é uma interface gráfica em PySide6 para usar um modelo previamente treinado em uma imagem digitada ou em um recorte de dígito.

Ele permite escolher o tipo de modelo e a versão:

- CNN
- SVM
- Random Forest

com versões V1, V2 e V3.

---

## 🏋️ Diferentes formas de treino

### 🟢 V1 - Treino base
A versão V1 usa o processamento padrão do MNIST sem alterações adicionais na imagem.

- imagens em formato 28x28
- normalização por 255
- treinamento padrão dos modelos

Comando:

```bash
python main.py V1
```

---

### 🟡 V2 - Treino com traços finos
A versão V2 aplica erosão nas imagens de treino para gerar traços mais finos.

Processamento usado:

- erosão via OpenCV
- geração de imagens suplementares
- concatenação com os dados originais
- treinamento com dados expandidos

Comando:

```bash
python main.py V2
```

Esse pipeline é útil para simular imagens com escrita mais delicada ou mais fina.

---

### 🔵 V3 - Treino com traços robustos
A versão V3 aplica:

- dilatação para engrossar linhas e traços
- blur gaussiano para suavizar bordas
- geração de imagens reforçadas para o treino

Comando:

```bash
python main.py V3
```

Essa versão foi desenhada para lidar melhor com imagens que tenham traços mais grossos ou menos definidos.

---

## 🧪 Modelos treinados

O projeto treina os seguintes algoritmos em cada versão:

### 🌲 Random Forest
- usado em dados vetorizados 784 pixels
- guarda o modelo em:
  - `models/V1/random_forest_mnist.joblib`
  - `models/V2/random_forest_mnist.joblib`
  - `models/V3/random_forest_mnist.joblib`

### 📈 SVM
- kernel RBF
- usado sobre os mesmos vetores 784 pixels
- salvo em:
  - `models/V1/svm_mnist.joblib`
  - `models/V2/svm_mnist.joblib`
  - `models/V3/svm_mnist.joblib`

### 🧠 CNN
- rede convolucional para entradas 28x28x1
- salva o melhor modelo em:
  - `models/V1/cnn_mnist.keras`
  - `models/V2/cnn_mnist.keras`
  - `models/V3/cnn_mnist.keras`

---

## 🚀 Como executar o treinamento completo

No diretório raiz do projeto:

```bash
python main.py V1
```

ou

```bash
python main.py V2
```

ou

```bash
python main.py V3
```

Durante a execução, o programa:

- baixa o dataset MNIST via OpenML
- salva gráficos em `images/<versao>/`
- salva modelos em `models/<versao>/`
- gera `metadata.json` com métricas e informações do treino

---

## 🖥️ Como usar a aplicação desktop

A aplicação desktop permite testar inferência em uma imagem específica.

### 🧭 Exemplos de uso

#### CNN da versão V3
```bash
python app.py --CNN:V3
```

#### Random Forest da versão V2
```bash
python app.py --RF:V2
```

#### SVM da versão V1
```bash
python app.py --SVM:V1
```

### Observação
A estrutura da entrada do argumento é:

```text
--<TIPO_DE_MODELO>:<VERSAO>
```

Onde:

- `--CNN` = CNN
- `--SVM` = SVM
- `--RF` = Random Forest
- `V1`, `V2`, `V3` = versões do pipeline

Exemplos válidos:

```bash
python app.py --CNN:V1
python app.py --RF:V2
python app.py --SVM:V3
```

---

## 📦 Dependências

O projeto verifica e instala automaticamente as bibliotecas necessárias em [src/install.py](src/install.py), incluindo:

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- opencv-python
- Pillow
- PySide6
- seaborn
- joblib

Se quiser rodar manualmente a verificação:

```bash
python -c "from src.install import verificar_e_instalar_bibliotecas; verificar_e_instalar_bibliotecas()"
```

---

## 📤 Saídas geradas

### 🖼️ Pastas de imagens
As imagens processadas e os gráficos ficam em:

- `images/V1/`
- `images/V2/`
- `images/V3/`

Exemplos de arquivos:

- `balanceamento_classes.png`
- `grade_exemplos*.png`
- `comparativo_processamento.png`
- `fase4_matrizes_confusao_comparativas.png`
- `resultado_teste_ood.png`

### 🧩 Pastas de modelos
Os modelos treinados ficam em:

- `models/V1/`
- `models/V2/`
- `models/V3/`

Exemplos de arquivos:

- `random_forest_mnist.joblib`
- `svm_mnist.joblib`
- `cnn_mnist.keras`
- `metadata.json`

---

## 💡 Dicas de uso

- Para a forma mais simples e rápida, use `V1`.
- Para melhorar robustez em traços mais finos, use `V2`.
- Para imagens com escrita mais grossa ou mais intensa, use `V3`.
- Para avaliar o melhor modelo, teste as três versões e compare as métricas salvas em `metadata.json`.

---

## ⚠️ Observações importantes

- O treinamento de CNN e dos modelos mais pesados pode levar algum tempo, principalmente no download do dataset e no ajuste dos modelos.
- O projeto salva os artefatos em pastas específicas por versão, o que facilita comparar desempenho entre `V1`, `V2` e `V3`.
- O teste OOD está presente para avaliar comportamento do modelo em instâncias fora do conjunto de treinamento padrão.

---

## ✅ Resumo rápido

```bash
# Treinar versão base
python main.py V1

# Treinar versão com traços finos
python main.py V2

# Treinar versão com traços robustos
python main.py V3

# Usar modelo treinado na aplicação desktop
python app.py --CNN:V3
python app.py --SVM:V1
python app.py --RF:V2
```