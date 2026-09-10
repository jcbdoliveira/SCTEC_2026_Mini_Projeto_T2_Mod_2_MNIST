import sys
import os
import subprocess
import importlib

def verificar_e_instalar_bibliotecas():
    """
    Realiza o setup automático das bibliotecas do projeto.
    """
    bibliotecas_necessarias = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "console": "console",
        "random": "random",
        "seaborn": "seaborn",
        "joblib": "joblib",
        "json": "json",
        "tensorflow": "tensorflow",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "PySide6": "PySide6", 
        "keras": "keras",
        "webbrowser": "webbrowser",
        "http.server": "http.server",
        "socketserver": "socketserver"
   }

    print("==================================================================")
    print(" [Setup Automático] Passo 1: Verificando bibliotecas ............")
    print("==================================================================")

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    ausentes = []
    for modulo_import, pacote_pip in bibliotecas_necessarias.items():
        try:
            importlib.import_module(modulo_import)
            print(f" -> [OK] Biblioteca '{modulo_import}' já está instalada.")
        except ImportError:
            print(f" -> [XX] Biblioteca '{modulo_import}' não encontrada.")
            ausentes.append(pacote_pip)

    if ausentes:
        print("\n------------------------------------------------------------------")
        print(" Passo 2: Instalando bibliotecas não encontradas..................")
        print("------------------------------------------------------------------")
        for pacote in ausentes:
            print(f"Instalando '{pacote}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])
                print(f" -> [OK] '{pacote}' instalado com sucesso!")
            except Exception as e:
                print("\n------------------------------------------------------------------")
                print(" Passo 3: Relatório de Erro na Instalação")
                print("------------------------------------------------------------------")
                print(f"Aviso: Não foi possível instalar '{pacote}' via pip: {e}")

def verifica_pastas(modelo):    
    os.makedirs(f"models/{modelo}", exist_ok=True)
    os.makedirs(f"images/{modelo}", exist_ok=True)