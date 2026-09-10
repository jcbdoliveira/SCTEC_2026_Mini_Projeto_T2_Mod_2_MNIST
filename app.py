# ==============================================================================
# APLICAÇÃO DESKTOP SIMPLIFICADA COM PySide6 E OpenCV (cv2)
# ==============================================================================

import sys
import os
import numpy as np
import cv2
import tensorflow as tf
import joblib  
from src.install import verificar_e_instalar_bibliotecas

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QGridLayout, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QFont

verificar_e_instalar_bibliotecas()

# --- Configuração e Carregamento Dinâmico por Parâmetros (`sys.argv`) ---
argumento_bruto = sys.argv[1] if len(sys.argv) > 1 else "--CNN:V3"
argumento_bruto = argumento_bruto.upper()

# Extrai o modelo (ex: --CNN) e a versão (ex: V3) corretamente
argumento = argumento_bruto.split(":")[0]  
versao_modelo = argumento_bruto.split(":")[1] if ":" in argumento_bruto else "V1"

if argumento == "--RF":
    NOME_MODELO = f"Random Forest ({versao_modelo})"
    CAMINHO_MODELO = os.path.join("models", versao_modelo, "random_forest_mnist.joblib")
elif argumento == "--SVM":
    NOME_MODELO = f"SVM ({versao_modelo})"
    CAMINHO_MODELO = os.path.join("models", versao_modelo, "svm_mnist.joblib")
else:
    NOME_MODELO = f"CNN ({versao_modelo})"
    CAMINHO_MODELO = os.path.join("models", versao_modelo, "cnn_mnist.keras")

# Carrega o modelo globalmente com base no argumento informado
if os.path.exists(CAMINHO_MODELO):
    if CAMINHO_MODELO.endswith(".keras"):
        modelo_carregado = tf.keras.models.load_model(CAMINHO_MODELO)
    else:
        modelo_carregado = joblib.load(CAMINHO_MODELO)
    print(f"[OK] Modelo {NOME_MODELO} carregado com sucesso de '{CAMINHO_MODELO}'!")
else:
    modelo_carregado = None
    print(f"[AVISO] Arquivo do modelo não encontrado em '{CAMINHO_MODELO}'.")

# Elementos globais da interface para manipulação direta
lbl_view_imagem = None
barras_progresso = []
labels_valores = []

def preprocessar_e_predizer_digito(roi_cinza):
    """Processa o recorte do dígito usando o pipeline V3 e realiza a predição."""
    if modelo_carregado is None:
        return 0, 0.0, np.zeros(10)

    # 1. Inversão adaptativa baseada nos cantos da imagem (Binarização Otsu)
    cantos = [roi_cinza[0, 0], roi_cinza[0, -1], roi_cinza[-1, 0], roi_cinza[-1, -1]]
    if np.mean(cantos) > 127:
        _, roi_bin = cv2.threshold(roi_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, roi_bin = cv2.threshold(roi_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Redimensionamento proporcional para caber em uma caixa de 20x20
    h, w = roi_bin.shape
    if h > w:
        novo_h, novo_w = 20, max(1, int(w * (20.0 / h)))
    else:
        novo_w, novo_h = 20, max(1, int(h * (20.0 / w)))
    roi_redim = cv2.resize(roi_bin, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

    # 3. Canvas 28x28 and centralização por Centro de Massa
    canvas = np.zeros((28, 28), dtype=np.uint8)
    M = cv2.moments(roi_redim)
    if M["m00"] != 0:
        pos_x = max(0, min(14 - int(M["m10"] / M["m00"]), 28 - novo_w))
        pos_y = max(0, min(14 - int(M["m01"] / M["m00"]), 28 - novo_h))
    else:
        pos_y, pos_x = (28 - novo_h) // 2, (28 - novo_w) // 2
    
    canvas[pos_y:pos_y + novo_h, pos_x:pos_x + novo_w] = roi_redim
    
    # Suavização Gaussiana final para recriar as bordas esfumaçadas do MNIST original
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # 4. Predição com distinção de formato e métodos de saída
    if argumento == "--SVM":
        tensor = (canvas.astype(np.float32) / 255.0).reshape(1, 784)
        scores = modelo_carregado.decision_function(tensor)[0]
        exp_scores = np.exp(scores - np.max(scores))  
        probs = exp_scores / np.sum(exp_scores)
        
    elif argumento == "--RF":
        tensor = (canvas.astype(np.float32) / 255.0).reshape(1, 784)
        probs = modelo_carregado.predict_proba(tensor)[0]
        
    else:  # --CNN
        tensor = (canvas.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
        probs = modelo_carregado.predict(tensor, verbose=0)[0]
    
    return int(np.argmax(probs)), float(np.max(probs)), probs

def processar_imagem_completa(caminho_arquivo):
    """Aplica o detector OpenCV na imagem carregada e mapeia múltiplos dígitos."""
    img_bgr = cv2.imread(caminho_arquivo)
    if img_bgr is None:
        return

    img_anotada = img_bgr.copy()
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Armazena os resultados para atualizar o painel Softmax com o dígito mais confiável da folha
    maior_confianca, melhor_distribuicao, melhor_digito = 0.0, np.zeros(10), -1

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        # Filtros geométricos para isolar os dígitos manuscritos das suas amostras
        if 200 < (w * h) < 200000 and 0.10 <= (w / h) <= 1.5 and h >= 15 and w >= 5:
            pad = int(max(w, h) * 0.15)
            roi = cinza[max(0, y-pad):min(cinza.shape[0], y+h+pad), max(0, x-pad):min(cinza.shape[1], x+w+pad)]
            
            if roi.size == 0: 
                continue
            
            digito, conf, dist = preprocessar_e_predizer_digito(roi)

            if conf > maior_confianca:
                maior_confianca, melhor_distribuicao, melhor_digito = conf, dist, digito

            # Desenha caixas verdes ao redor de cada dígito detectado no papel/post-it
            cv2.rectangle(img_anotada, (x, y), (x + w, y + h), (40, 180, 80), 3)
            tag = f"{digito} ({conf * 100:.1f}%)"
            cv2.putText(
                img_anotada, tag, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 180, 80), 2, cv2.LINE_AA
            )

    # Converte e exibe a imagem anotada no QLabel do Qt
    img_rgb = cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)
    h_img, w_img, ch = img_rgb.shape
    q_img = QImage(img_rgb.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
    lbl_view_imagem.setPixmap(QPixmap.fromImage(q_img).scaled(650, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # Atualiza as barras de progresso com as probabilidades do dígito de maior destaque
    for i in range(10):
        prob_percentual = float(melhor_distribuicao[i]) * 100
        barras_progresso[i].setValue(int(prob_percentual))
        labels_valores[i].setText(f"{prob_percentual:.1f}%")
        barras_progresso[i].setStyleSheet("QProgressBar::chunk { background-color: #D47B52; }" if i == melhor_digito else "")

def acao_botao_carregar():
    """Gerencia o clique do botão de upload."""
    caminho, _ = QFileDialog.getOpenFileName(None, "Escolher Imagem", "", "Imagens (*.png *.jpg *.jpeg *.bmp)")
    if caminho:
        processar_imagem_completa(caminho)


# --- Bloco de Inicialização da Janela e Layout (Interface) ---
app = QApplication(sys.argv)

janela = QMainWindow()
janela.setWindowTitle(f"Verificador de Dígitos (MNIST) - Modelo: {NOME_MODELO}")
janela.resize(1000, 550)

widget_central = QWidget()
layout_principal = QHBoxLayout(widget_central)

# Lado Esquerdo: Botão e Visualizador da Imagem
layout_esquerdo = QVBoxLayout()
btn_carregar = QPushButton("Carregar Imagem")
btn_carregar.setStyleSheet("background-color: #D47B52; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
btn_carregar.clicked.connect(acao_botao_carregar)

lbl_view_imagem = QLabel("Clique no botão para selecionar uma imagem.")
lbl_view_imagem.setAlignment(Qt.AlignCenter)
lbl_view_imagem.setStyleSheet("background-color: #F5F5F0; border: 2px dashed #E6DFD3; border-radius: 8px;")

layout_esquerdo.addWidget(btn_carregar)
layout_esquerdo.addWidget(lbl_view_imagem, stretch=1)
layout_principal.addLayout(layout_esquerdo, stretch=2)

# Lado Direito: Barras Softmax de Distribuição
grupo_softmax = QGroupBox(f"Probabilidades ({NOME_MODELO})")
layout_softmax = QGridLayout(grupo_softmax)

for i in range(10):
    barra = QProgressBar()
    barra.setRange(0, 100)
    barra.setValue(0)
    barra.setFixedHeight(18)
    
    lbl_val = QLabel("0.0%")
    lbl_val.setFixedWidth(50)
    lbl_val.setFont(QFont("Arial", 10, QFont.Bold))

    lbl_val2 = QLabel(f"Nº {i}:")
    lbl_val2.setFont(QFont("Arial", 10, QFont.Bold))
    lbl_val2.setFixedWidth(40)
    
    layout_softmax.addWidget(lbl_val2, i, 0)
    layout_softmax.addWidget(barra, i, 1)
    layout_softmax.addWidget(lbl_val, i, 2)
    
    barras_progresso.append(barra)
    labels_valores.append(lbl_val)

layout_principal.addWidget(grupo_softmax, stretch=1)

janela.setCentralWidget(widget_central)
janela.show()
sys.exit(app.exec())
