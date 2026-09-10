#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Visualizador 3D TensorSpace para Redes Convolucionais Keras (MNIST)
=============================================================================
Este script:
1. Carrega o modelo 'models/V2/cnn_mnist.keras' (ou gera um se solicitado).
2. Inspeciona a arquitetura (camadas, formatos, parâmetros).
3. Converte os pesos e a topologia para o formato compatível com TensorSpace/TFJS.
4. Gera a página HTML interativa 3D baseada no TensorSpace Playground - LeNet.
5. Inicia um servidor HTTP local (http.server) na porta 8000 e abre o navegador.
=============================================================================
"""

import sys
import json
import webbrowser
from pathlib import Path
import http.server
import socketserver

# ---------------------------------------------------------------------------
# Configurações do Projeto
# ---------------------------------------------------------------------------
model_path = Path("models/V2/cnn_mnist.keras")
OUTPUT_DIR = Path("tensorspace_output")
SERVER_PORT = 8000
SERVER_HOST = "localhost"

# ---------------------------------------------------------------------------
# Funções de Verificação e Ambiente
# ---------------------------------------------------------------------------
def verificar_ambiente():
    """Verifica versão do Python e importação do TensorFlow."""
    print("=" * 70)
    print("  TensorSpace CNN Visualizer - Inicializando")
    print("=" * 70)
    print(f"[1/6] Verificando versão do Python: {sys.version.split()[0]}...")
    
    if sys.version_info < (3, 9):
        print("[ALERTA] É recomendado Python 3.10 ou superior.")

    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow carregado com sucesso (versão: {tf.__version__})")
        return tf
    except ImportError:
        print("\n" + "!" * 70)
        print("[ERRO FATAL] TensorFlow não está instalado neste ambiente virtual!")
        print("Para instalar as dependências necessárias, execute:")
        print("    pip install -r requirements.txt")
        print("Ou:")
        print("    pip install tensorflow==2.15.0 numpy")
        print("!" * 70 + "\n")
        sys.exit(1)

def verificar_ou_criar_modelo(tf):
    """Verifica se o arquivo do modelo existe. Se não existir, oferece criar um modelo de exemplo."""
    print(f"\n[2/6] Verificando arquivo do modelo: '{model_path}'...")
    
    if model_path.exists():
        print(f"[OK] Arquivo encontrado: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
        return True
    
    print(f"\n[AVISO] O arquivo '{model_path}' não foi encontrado!")
    print("Deseja criar um modelo CNN MNIST de exemplo compatível agora?")
    print("1. Sim, treinar rapidamente uma CNN MNIST leve (1 época, ~15 seg)")
    print("2. Sim, gerar e salvar arquitetura LeNet-5 com pesos pré-inicializados (instantâneo)")
    print("3. Não, vou colocar meu próprio arquivo 'cnn_mnist.keras' na pasta 'models/V2/'")
    
    # Se rodando sem terminal interativo ou entrada padrão
    try:
        opcao = input("\nEscolha uma opção [1/2/3] (padrão: 2): ").strip() or "2"
    except (EOFError, KeyboardInterrupt):
        opcao = "2"
    
    if opcao == "1":
        criar_modelo_treinado(tf)
        return True
    elif opcao == "2":
        criar_modelo_instantaneo(tf)
        return True
    else:
        print(f"\nPor favor, copie o seu arquivo para '{model_path.resolve()}' e execute novamente:")
        print("    python visualizar_modelo.py")
        sys.exit(0)


def criar_modelo_instantaneo(tf):
    """Cria uma CNN MNIST estilo LeNet e salva em models/V2/cnn_mnist.keras imediatamente."""
    print("\n--> Criando arquitetura CNN LeNet para MNIST...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Cria modelo Sequential LeNet-5 adaptado para MNIST (28x28)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1), name="mnist_input"),
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding="same", activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_1"),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding="valid", activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(120, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(84, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(10, activation="softmax", name="output_digits")
    ], name="cnn_mnist_lenet")
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save(model_path)
    print(f"[OK] Modelo Sequential gerado e salvo com sucesso em: {model_path}")


def criar_modelo_treinado(tf):
    """Treina rapidamente uma CNN no dataset MNIST e salva o modelo."""
    print("\n--> Baixando MNIST e realizando treino rápido (1 época)...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:10000].astype("float32") / 255.0
    x_train = x_train[..., None]
    y_train = y_train[:10000]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1), name="mnist_input"),
        tf.keras.layers.Conv2D(6, (5, 5), padding="same", activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_1"),
        tf.keras.layers.Conv2D(16, (5, 5), padding="valid", activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(120, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(84, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(10, activation="softmax", name="output_digits")
    ], name="cnn_mnist_trained")
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
    
    model.save(model_path)
    print(f"[OK] Modelo treinado e salvo com sucesso em: {model_path}")


# ---------------------------------------------------------------------------
# Inspeção Detalhada do Modelo Keras
# ---------------------------------------------------------------------------
def inspecionar_modelo(tf, model):
    """Inspeciona as camadas do modelo, exibe tabela detalhada e detecta compatibilidade com TensorSpace."""
    print("\n[3/6] Inspecionando arquitetura do modelo...")
    print("-" * 75)
    model.summary()
    print("-" * 75)
    
    print("\nTABELA DETALHADA DE CAMADAS:")
    print(f"{'#':<3} | {'Nome da Camada':<18} | {'Tipo':<16} | {'Formato Saída':<16} | {'Params':<8} | {'TensorSpace'}")
    print("-" * 88)
    
    info_camadas = []
    camadas_ignoradas = []
    
    # Camadas diretamente renderizáveis no TensorSpace
    tipos_suportados = {
        "Conv2D": "Conv2d",
        "MaxPooling2D": "Pooling2d (Max)",
        "AveragePooling2D": "Pooling2d (Avg)",
        "Dense": "Dense",
        "Flatten": "Flatten (Transição)",
        "InputLayer": "GreyscaleInput",
        "Reshape": "Reshape",
        "Softmax": "Output1d"
    }
    
    # Camadas não renderizadas visualmente em 3D
    tipos_nao_visuais = {
        "Dropout": "Ignorada visualmente (ativa apenas no treino)",
        "BatchNormalization": "Ignorada visualmente (não altera topologia 3D)",
        "Activation": "Fundida na camada anterior"
    }

    ordem = 1
    for layer in model.layers:
        nome_tipo = layer.__class__.__name__
        qtd_params = layer.count_params()
        
        try:
            formato_saida = str(layer.output_shape)
        except Exception:
            formato_saida = "N/A"
            
        compativel = "Renderizável"
        if nome_tipo in tipos_suportados:
            status_ts = f"OK: {tipos_suportados[nome_tipo]}"
        elif nome_tipo in tipos_nao_visuais:
            status_ts = f"IGNORAR: {tipos_nao_visuais[nome_tipo]}"
            camadas_ignoradas.append((layer.name, nome_tipo, tipos_nao_visuais[nome_tipo]))
            compativel = "Ignorada"
        else:
            status_ts = f"AVISO: Não suportada nativamente"
            camadas_ignoradas.append((layer.name, nome_tipo, "Sem suporte direto no TensorSpace"))
            compativel = "Incompatível"

        print(f"{ordem:<3} | {layer.name:<18} | {nome_tipo:<16} | {formato_saida:<16} | {qtd_params:<8} | {status_ts}")
        
        info_camadas.append({
            "order": ordem,
            "name": layer.name,
            "type": nome_tipo,
            "output_shape": layer.output_shape if hasattr(layer, "output_shape") else None,
            "params": qtd_params,
            "config": layer.get_config(),
            "compatible": compativel
        })
        ordem += 1

    print("-" * 88)
    if camadas_ignoradas:
        print("\n[NOTA DE COMPATIBILIDADE TENSORSPACE]")
        for nome, tipo, motivo in camadas_ignoradas:
            print(f"  - Camada '{nome}' ({tipo}): {motivo}.")
        print("  * A omissão visual dessas camadas NÃO afeta a inferência matemática dos pesos.")
    
    return info_camadas, camadas_ignoradas


# ---------------------------------------------------------------------------
# Conversão do Modelo para TensorFlow.js / TensorSpace
# ---------------------------------------------------------------------------
def converter_modelo_para_tensorspace(tf, model, output_dir):
    """
    Converte o modelo Keras para os arquivos requeridos pelo TensorSpace:
    - model.json (topologia e manifesto de pesos compatível com TFJS)
    - weights.bin (pesos binários em Float32 little-endian)
    - model_spec.json (metadados estruturados para adaptação dinâmica)
    """
    print(f"\n[4/6] Convertendo modelo para formato compatível com TensorSpace...")
    model_export_dir = output_dir / "converted_model"
    model_export_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Tentar conversor oficial tensorflowjs se disponível
    usou_tfjs_oficial = False
    try:
        import tensorflowjs as tfjs
        print("--> Pacote 'tensorflowjs' detectado! Executando conversor oficial...")
        tfjs.converters.save_keras_model(model, str(model_export_dir))
        usou_tfjs_oficial = True
        print("[OK] Modelo convertido com sucesso usando tensorflowjs!")
    except Exception as e:
        print(f"--> Conversor oficial tfjs não disponível ou incompatível ({e}).")
        print("--> Executando conversor nativo de alta fidelidade Keras -> TFJS...")
    
    # 2. Conversor nativo puro (garante funcionamento independente de conflitos de versão tfjs)
    if not usou_tfjs_oficial or not (model_export_dir / "model.json").exists():
        exportar_tfjs_nativo(model, model_export_dir)

    # 2.5. Ajuste crítico de compatibilidade com TensorFlow.js 1.7.4 (TensorSpace):
    # - Keras 2.6+ grava o grafo como "Functional" -> converte para "Model"
    # - Keras 3 / TF 2.16+ omite batchInputShape em InputLayer -> enriquece com batchInputShape / inputShape
    model_json_path = model_export_dir / "model.json"
    if model_json_path.exists():
        try:
            with open(model_json_path, "r", encoding="utf-8") as f:
                dados_json = json.load(f)
            if "modelTopology" in dados_json:
                dados_json["modelTopology"] = sanitizar_topologia_tfjs(dados_json["modelTopology"])
            with open(model_json_path, "w", encoding="utf-8") as f:
                json.dump(dados_json, f, indent=2)
            print("[OK] Topologia e camadas sanitizadas para compatibilidade total com TFJS 1.7.4.")
        except Exception as e:
            print(f"[AVISO] Não foi possível verificar model.json: {e}")

    # 3. Exportar especificação JSON da arquitetura para a renderização dinâmica
    spec = gerar_especificacao_arquitetura(model)
    with open(model_export_dir / "model_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
        
    print(f"[OK] Arquivos gerados em '{model_export_dir}':")
    for f in model_export_dir.iterdir():
        print(f"     - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


def sanitizar_topologia_tfjs(model_topology):
    """
    Ajusta recursivamente a topologia do Keras para o formato estritamente esperado pelo TensorFlow.js 1.7.4:
    - Converte 'Functional' para 'Model'
    - Garante que InputLayer receba EXCLUSIVAMENTE 'batchInputShape', NUNCA 'inputShape' junto.
      (TFJS lança: 'Only provide the inputShape OR batchInputShape argument to inputLayer, not both at the same time.')
    - Converte 'inbound_nodes' de objetos Keras 3 ({"args": [...]}) para matrizes Keras 2 ([[[layer_name, 0, 0, {}]]]).
      (TFJS lança: 'Corrupted configuration, expected array for nodeData: [object Object]')
    - Garante que input_layers e output_layers sejam matrizes válidas [[nome, 0, 0]].
    """
    if not isinstance(model_topology, dict):
        return model_topology
        
    # 1. Substituir "Functional" por "Model" para compatibilidade com TFJS
    if model_topology.get("class_name") == "Functional":
        model_topology["class_name"] = "Model"
        
    config = model_topology.get("config", {})
    if isinstance(config, dict):
        layers = config.get("layers", [])
        prev_layer_name = None

        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                continue
            cls_name = layer.get("class_name")
            cfg = layer.get("config", {})
            if not isinstance(cfg, dict):
                cfg = {}
                layer["config"] = cfg

            layer_name = layer.get("name") or cfg.get("name")
            
            b_shape = cfg.get("batch_input_shape") or cfg.get("batch_shape") or cfg.get("batchInputShape") or cfg.get("batchShape")
            in_shape = cfg.get("input_shape") or cfg.get("inputShape") or cfg.get("shape")
            
            if cls_name == "InputLayer" or idx == 0:
                if b_shape and isinstance(b_shape, (list, tuple)):
                    final_batch_shape = list(b_shape)
                elif in_shape and isinstance(in_shape, (list, tuple)):
                    final_batch_shape = [None] + list(in_shape)
                else:
                    final_batch_shape = [None, 28, 28, 1]
                
                cfg["batchInputShape"] = final_batch_shape
                # Remover terminantemente inputShape e variantes para não acionar o erro de colisão no TFJS:
                for k in ["inputShape", "input_shape", "shape", "batch_shape", "batch_input_shape", "batchShape"]:
                    cfg.pop(k, None)
                layer["inbound_nodes"] = []
                prev_layer_name = layer_name
                continue
            else:
                if cfg.get("batchInputShape") and cfg.get("inputShape"):
                    cfg.pop("inputShape", None)
                    cfg.pop("input_shape", None)

            # Sanitizar inbound_nodes para Keras 2 / TFJS 1.7.4
            raw_nodes = layer.get("inbound_nodes")
            clean_nodes = []

            if isinstance(raw_nodes, list) and len(raw_nodes) > 0:
                for node in raw_nodes:
                    node_entries = []
                    if isinstance(node, dict):
                        # Formato Keras 3: {"args": [...], "kwargs": {...}}
                        args = node.get("args", [])
                        for arg in args:
                            if isinstance(arg, dict):
                                c = arg.get("config", {}) if isinstance(arg.get("config"), dict) else {}
                                hist = c.get("keras_history") or arg.get("keras_history")
                                if isinstance(hist, (list, tuple)) and len(hist) >= 1:
                                    node_entries.append([str(hist[0]), int(hist[1]) if len(hist) > 1 else 0, int(hist[2]) if len(hist) > 2 else 0, {}])
                                elif "name" in c:
                                    node_entries.append([str(c["name"]), 0, 0, {}])
                            elif isinstance(arg, (list, tuple)) and len(arg) >= 1:
                                node_entries.append([str(arg[0]), int(arg[1]) if len(arg) > 1 else 0, int(arg[2]) if len(arg) > 2 else 0, {}])
                            elif isinstance(arg, str):
                                node_entries.append([arg, 0, 0, {}])
                    elif isinstance(node, (list, tuple)):
                        for item in node:
                            if isinstance(item, (list, tuple)):
                                if len(item) > 0 and isinstance(item[0], (list, tuple)):
                                    for sub in item:
                                        node_entries.append([str(sub[0]), int(sub[1]) if len(sub) > 1 else 0, int(sub[2]) if len(sub) > 2 else 0, {}])
                                else:
                                    kw = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
                                    node_entries.append([str(item[0]), int(item[1]) if len(item) > 1 else 0, int(item[2]) if len(item) > 2 else 0, kw])
                            elif isinstance(item, dict):
                                c = item.get("config", {}) if isinstance(item.get("config"), dict) else {}
                                hist = c.get("keras_history") or item.get("keras_history")
                                if isinstance(hist, (list, tuple)) and len(hist) >= 1:
                                    node_entries.append([str(hist[0]), int(hist[1]) if len(hist) > 1 else 0, int(hist[2]) if len(hist) > 2 else 0, {}])
                            elif isinstance(item, str):
                                node_entries.append([item, 0, 0, {}])

                    if node_entries:
                        clean_nodes.append(node_entries)

            if not clean_nodes and prev_layer_name:
                clean_nodes = [[[prev_layer_name, 0, 0, {}]]]

            layer["inbound_nodes"] = clean_nodes
            prev_layer_name = layer_name

        # Sanitizar input_layers e output_layers
        if layers:
            first_name = layers[0].get("name") or layers[0].get("config", {}).get("name")
            last_name = layers[-1].get("name") or layers[-1].get("config", {}).get("name")
            if first_name:
                config["input_layers"] = [[first_name, 0, 0]]
            if last_name:
                config["output_layers"] = [[last_name, 0, 0]]
                        
    return model_topology


def exportar_tfjs_nativo(model, target_dir):
    """
    Exporta os pesos e a topologia diretamente para a especificação TFJS:
    Gera 'model.json' e 'group1-shard1of1.bin' contendo Float32 little-endian.
    """
    weights_data = bytearray()
    weights_manifest_entries = []
    
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
            
        weight_names = []
        if len(weights) == 1:
            weight_names = [f"{layer.name}/kernel"]
        elif len(weights) == 2:
            weight_names = [f"{layer.name}/kernel", f"{layer.name}/bias"]
        elif len(weights) == 4: # BatchNorm
            weight_names = [f"{layer.name}/gamma", f"{layer.name}/beta", f"{layer.name}/moving_mean", f"{layer.name}/moving_variance"]
        else:
            weight_names = [f"{layer.name}/param_{i}" for i in range(len(weights))]
            
        for name, w_array in zip(weight_names, weights):
            flat_w = w_array.astype("float32").flatten()
            byte_length = len(flat_w) * 4
            weights_data.extend(flat_w.tobytes())
            
            weights_manifest_entries.append({
                "name": name,
                "shape": list(w_array.shape),
                "dtype": "float32"
            })
            
    bin_filename = "group1-shard1of1.bin"
    with open(target_dir / bin_filename, "wb") as f:
        f.write(weights_data)
        
    # Salvar model.json no formato oficial TFJS
    is_seq = (model.__class__.__name__ == "Sequential")
    top_class_name = "Sequential" if is_seq else "Model"
    
    topology = sanitizar_topologia_tfjs({
        "class_name": top_class_name,
        "config": model.get_config()
    })
    
    model_json = {
        "format": "layers-model",
        "generatedBy": "TensorSpace-Keras-Native-Converter",
        "convertedBy": "visualizar_modelo.py",
        "modelTopology": topology,
        "weightsManifest": [
            {
                "paths": [bin_filename],
                "weights": weights_manifest_entries
            }
        ]
    }
    
    with open(target_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)


def gerar_especificacao_arquitetura(model):
    """Extrai informações precisas de cada camada para a visualização TensorSpace."""
    layers_summary = []
    for layer in model.layers:
        cfg = layer.get_config()
        tipo = layer.__class__.__name__
        
        info = {
            "name": layer.name,
            "type": tipo,
            "params": layer.count_params()
        }
        
        if hasattr(layer, "output_shape"):
            # Tratar tuplas de formatos
            shp = layer.output_shape
            if isinstance(shp, list) and len(shp) > 0:
                shp = shp[0]
            info["output_shape"] = list(shp) if shp else None
            
        if tipo == "Conv2D":
            info["filters"] = cfg.get("filters", 16)
            info["kernel_size"] = cfg.get("kernel_size", [5, 5])
            info["strides"] = cfg.get("strides", [1, 1])
            info["padding"] = cfg.get("padding", "valid")
            info["activation"] = cfg.get("activation", "relu")
        elif tipo in ["MaxPooling2D", "AveragePooling2D"]:
            info["pool_size"] = cfg.get("pool_size", [2, 2])
            info["strides"] = cfg.get("strides", [2, 2])
        elif tipo == "Dense":
            info["units"] = cfg.get("units", 10)
            info["activation"] = cfg.get("activation", "linear")
            
        layers_summary.append(info)
        
    return {
        "model_name": getattr(model, "name", "cnn_mnist"),
        "total_params": model.count_params(),
        "layers": layers_summary
    }


# ---------------------------------------------------------------------------
# Geração do Arquivo HTML com TensorSpace e Explicações das Camadas
# ---------------------------------------------------------------------------
def gerar_metadados_didaticos_camadas(info_camadas):
    """Gera metadados pedagógicos completos (nome, importância, papel e matemática) para cada camada."""
    metadados = []
    
    # 1. Entrada MNIST
    metadados.append({
        "id": "mnist_input",
        "name": "mnist_input",
        "type": "GreyscaleInput (InputLayer)",
        "display_type": "Camada de Entrada (Greyscale 28×28)",
        "shape": "[28, 28, 1]",
        "params": 0,
        "activation": "Normalização linear [0.0, 1.0]",
        "importance": "Porta de entrada dos dados na rede. Preserva integralmente a geometria 2D e a vizinhança espacial dos pixels desenhados pelo usuário, condição indispensável para a visão computacional.",
        "role": "Recebe a imagem 28×28 desenhada no canvas, converte os valores de intensidade de cinza (0 a 255) para o intervalo [0.0, 1.0] e estrutura a matriz como um tensor 3D de formato [28, 28, 1]. Mantém a continuidade das bordas e curvas para alimentar os filtros convolucionais subsequentes.",
        "color": "#0ea5e9"
    })
    
    total = len(info_camadas)
    for idx, c in enumerate(info_camadas):
        nome = c["name"]
        tipo = c["type"]
        cfg = c.get("config", {})
        params = c.get("params", 0)
        shape_val = c.get("output_shape")
        shape_str = str(shape_val) if shape_val else "[28, 28, 1]"
        eh_ultima = (idx == total - 1)
        
        if tipo == "Conv2D":
            filtros = cfg.get("filters", 16)
            k = cfg.get("kernel_size", [5, 5])
            k_str = f"{k[0]}×{k[1]}" if isinstance(k, (list, tuple)) else f"{k}×{k}"
            ativ = cfg.get("activation", "relu").upper()
            metadados.append({
                "id": nome,
                "name": nome,
                "type": f"Convolucional 2D ({filtros} filtros {k_str})",
                "display_type": f"Convolução 2D ({filtros} filtros)",
                "shape": shape_str,
                "params": params,
                "activation": ativ,
                "importance": "O núcleo central da rede convolucional. Detecta traços e padrões visuais fundamentais (arestas verticais, horizontais, curvas e cantos) de forma invariante ao deslocamento.",
                "role": f"Aplica {filtros} pequenos filtros matemáticos (kernels de tamanho {k_str}) que varrem a imagem calculando multiplicações locais. Cada filtro é treinado para acender intensamente diante de traços específicos (ex: arco do dígito 0 ou traço reto do 1), gerando {filtros} mapas de características (feature maps). Em seguida, a ativação {ativ} elimina valores negativos, inserindo não-linearidade essencial.",
                "color": "#38bdf8"
            })
        elif tipo in ["MaxPooling2D", "AveragePooling2D"]:
            p = cfg.get("pool_size", [2, 2])
            p_str = f"{p[0]}×{p[1]}" if isinstance(p, (list, tuple)) else f"{p}×{p}"
            tipo_pool = "Max Pooling" if "Max" in tipo else "Average Pooling"
            metadados.append({
                "id": nome,
                "name": nome,
                "type": f"Subamostragem ({tipo_pool} {p_str})",
                "display_type": f"Agrupamento ({tipo_pool})",
                "shape": shape_str,
                "params": params,
                "activation": "Nenhuma (Redução de Resolução)",
                "importance": "Concede 'invariância à translação' (o número continua sendo reconhecido mesmo desenhado levemente fora de centro) e reduz o volume de dados em até 75%, evitando memorização excessiva e acelerando o processamento.",
                "role": f"Divide cada mapa de características em blocos {p_str} e retém apenas o maior valor ('max') de cada região. Isso descarta ruídos e pequenas variações na grossura do traço, preservando as características mais dominantes com resolução espacial reduzida pela metade.",
                "color": "#818cf8"
            })
        elif tipo == "Flatten":
            metadados.append({
                "id": nome,
                "name": nome,
                "type": "Achatamento de Tensores (Flatten)",
                "display_type": "Achatamento (Flatten)",
                "shape": shape_str,
                "params": 0,
                "activation": "Nenhuma (Reorganização Geométrica)",
                "importance": "Atua como ponte indispensável entre a visão espacial bidimensional (mapas de pixels 2D) e a lógica de decisão vetorial das camadas densas totalmente conectadas.",
                "role": "Reorganiza a matriz tridimensional de mapas de características em um único vetor linear contínuo (unidimensional), permitindo que todos os pontos detectados sejam alimentados simultaneamente aos neurônios classificadores.",
                "color": "#a855f7"
            })
        elif tipo == "Dropout":
            taxa = int(cfg.get("rate", 0.25) * 100)
            metadados.append({
                "id": nome,
                "name": nome,
                "type": f"Regularização Dropout ({taxa}%)",
                "display_type": f"Dropout ({taxa}%)",
                "shape": shape_str,
                "params": 0,
                "activation": "Máscara Estocástica (Treinamento)",
                "importance": "Principal mecanismo de proteção contra 'overfitting' (quando o modelo decora os dados de treino e erra em novos desenhos feitos pelo usuário).",
                "role": f"Durante o treino, desliga aleatoriamente {taxa}% dos neurônios a cada ciclo. Isso força os neurônios restantes a aprender representações ricas e independentes, impedindo que a rede dependa de conexões isoladas para reconhecer um número.",
                "color": "#f59e0b"
            })
        elif tipo == "Dense":
            unidades = cfg.get("units", 10)
            ativ = cfg.get("activation", "relu").upper()
            if eh_ultima or unidades == 10:
                metadados.append({
                    "id": nome,
                    "name": nome,
                    "type": f"Camada de Saída (Classificador Softmax - {unidades} dígitos)",
                    "display_type": "Saída (Classificação Final)",
                    "shape": shape_str,
                    "params": params,
                    "activation": "Softmax",
                    "importance": "A camada de decisão e veredito final da rede neural. Converte os padrões abstratos processados em probabilidades explícitas e calibradas de 0% a 100% para cada dígito de 0 a 9.",
                    "role": "Possui exatamente 10 neurônios (um para cada dígito de 0 a 9). Cada neurônio avalia a afinidade do vetor de características com a silhueta típica do dígito correspondente. A função Softmax normaliza os valores de modo que a soma total das probabilidades resulte rigorosamente em 100%, elegendo o neurônio de maior valor como a previsão final.",
                    "color": "#10b981"
                })
            else:
                metadados.append({
                    "id": nome,
                    "name": nome,
                    "type": f"Camada Densa Oculta ({unidades} neurônios)",
                    "display_type": f"Densa Oculta ({unidades} neurônios)",
                    "shape": shape_str,
                    "params": params,
                    "activation": ativ,
                    "importance": "Camada de síntese e raciocínio de alto nível. Combina os traços locais identificados pelas camadas convolucionais anteriores para montar a percepção visual holística do dígito.",
                    "role": f"Cada um dos seus {unidades} neurônios está totalmente conectado a todas as saídas do vetor anterior. Cada neurônio calcula uma soma ponderada dos pesos, adiciona um viés (bias) e aplica ativação {ativ}, permitindo relacionar curvas e retas complexas.",
                    "color": "#6366f1"
                })
        else:
            metadados.append({
                "id": nome,
                "name": nome,
                "type": tipo,
                "display_type": tipo,
                "shape": shape_str,
                "params": params,
                "activation": cfg.get("activation", "N/A"),
                "importance": f"Camada estrutural da arquitetura ({tipo}). Contribui para a transformação dos dados ao longo do pipeline de inferência.",
                "role": f"Processa os tensores recebidos de acordo com as especificações da camada {tipo}.",
                "color": "#64748b"
            })
            
    return metadados


def gerar_codigo_js_tensorspace(info_camadas):
    """Gera as chamadas JavaScript para construir dinamicamente o modelo TensorSpace."""
    js_lines = []
    
    # 1. Entrada MNIST (28x28x1)
    js_lines.append("    // 1. Camada de entrada (GreyscaleInput para MNIST 28x28x1)")
    js_lines.append("    const inLayer = new TSP.layers.GreyscaleInput({ shape: [28, 28, 1], name: 'mnist_input' });")
    js_lines.append("    model.add(inLayer);")
    js_lines.append("    window.tspCamadasMapa['mnist_input'] = inLayer;")
    
    for camada in info_camadas:
        tipo = camada["type"]
        nome = camada["name"]
        cfg = camada["config"]
        
        if tipo == "Conv2D":
            filters = cfg.get("filters", 16)
            ksize = cfg.get("kernel_size", [5, 5])
            ksize_val = ksize[0] if isinstance(ksize, (list, tuple)) else ksize
            strides = cfg.get("strides", [1, 1])
            strides_val = strides[0] if isinstance(strides, (list, tuple)) else strides
            
            js_lines.append(f"    // Conv2D: {nome} ({filters} filtros, kernel {ksize_val}x{ksize_val})")
            js_lines.append(f"    const layer_{nome} = new TSP.layers.Conv2d({{")
            js_lines.append(f"        name: '{nome}',")
            js_lines.append(f"        kernelSize: {ksize_val},")
            js_lines.append(f"        filters: {filters},")
            js_lines.append(f"        strides: {strides_val}")
            js_lines.append(f"    }});")
            js_lines.append(f"    model.add(layer_{nome});")
            js_lines.append(f"    window.tspCamadasMapa['{nome}'] = layer_{nome};")
            
        elif tipo in ["MaxPooling2D", "AveragePooling2D"]:
            pool_size = cfg.get("pool_size", [2, 2])
            pool_val = pool_size[0] if isinstance(pool_size, (list, tuple)) else pool_size
            strides = cfg.get("strides", [2, 2])
            strides_val = strides[0] if isinstance(strides, (list, tuple)) else strides
            
            js_lines.append(f"    // Pooling: {nome} (pool {pool_val}x{pool_val}, strides {strides_val})")
            js_lines.append(f"    const layer_{nome} = new TSP.layers.Pooling2d({{")
            js_lines.append(f"        name: '{nome}',")
            js_lines.append(f"        poolSize: [{pool_val}, {pool_val}],")
            js_lines.append(f"        strides: [{strides_val}, {strides_val}]")
            js_lines.append(f"    }});")
            js_lines.append(f"    model.add(layer_{nome});")
            js_lines.append(f"    window.tspCamadasMapa['{nome}'] = layer_{nome};")
            
        elif tipo == "Dense":
            units = cfg.get("units", 10)
            is_last = (camada == info_camadas[-1]) or (units == 10)
            
            if is_last:
                js_lines.append(f"    // Output1d: {nome} (10 classes 0-9)")
                js_lines.append(f"    const layer_{nome} = new TSP.layers.Output1d({{")
                js_lines.append(f"        name: '{nome}',")
                js_lines.append(f"        units: {units},")
                js_lines.append(f"        outputs: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']")
                js_lines.append(f"    }});")
                js_lines.append(f"    model.add(layer_{nome});")
                js_lines.append(f"    window.tspCamadasMapa['{nome}'] = layer_{nome};")
            else:
                js_lines.append(f"    // Dense: {nome} ({units} unidades)")
                js_lines.append(f"    const layer_{nome} = new TSP.layers.Dense({{")
                js_lines.append(f"        name: '{nome}',")
                js_lines.append(f"        units: {units}")
                js_lines.append(f"    }});")
                js_lines.append(f"    model.add(layer_{nome});")
                js_lines.append(f"    window.tspCamadasMapa['{nome}'] = layer_{nome};")
                
        elif tipo in ["Dropout", "BatchNormalization", "Activation"]:
            js_lines.append(f"    // Camada '{nome}' ({tipo}) omitida na renderização 3D direta (não possui representação geométrica direta no TensorSpace)")
            
    return "\n".join(js_lines)


def gerar_arquivo_html(info_camadas, camadas_ignoradas, output_dir):
    """Gera o arquivo index.html completo e autônomo com TensorSpace Playground LeNet adaptado e inspeção detalhada de camadas."""
    print(f"\n[5/6] Gerando arquivo HTML da visualização 3D...")
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    
    js_layers_code = gerar_codigo_js_tensorspace(info_camadas)
    metadados_camadas = gerar_metadados_didaticos_camadas(info_camadas)
    metadados_json = json.dumps(metadados_camadas, ensure_ascii=False)
    
    # Gerar chips HTML das camadas para a barra de navegação
    chips_html_list = []
    idx = 0
    for item in metadados_camadas:
        idx = idx + 1
        cid = item["id"]
        cname = item["name"]
        cdisp = item["display_type"]
        ccolor = item.get("color", "#38bdf8")
        chips_html_list.append(
            f'<button class="layer-chip" style="--chip-color: {ccolor};" onclick="abrirDetalhesCamada(\'{cid}\')">'
            f'<span class="chip-dot" style="background: {ccolor};"></span>'
            f'<span class="chip-text">{idx} - {cdisp}</span>'
            f'</button>'
        )
    chips_bar_html = "\n      ".join(chips_html_list)
    
    # Montar avisos de camadas ignoradas em HTML
    avisos_html = ""
    if camadas_ignoradas:
        avisos_html = "<div class='warning-box'><strong>Camadas não-visuais omitidas na geometria 3D:</strong><ul>"
        for n, t, m in camadas_ignoradas:
            avisos_html += f"<li><code>{n}</code> ({t}): {m}</li>"
        avisos_html += "</ul><small>A inferência matemática permanece idêntica.</small></div>"
        
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Visualização 3D da rede CNN treinada</title>
  
  <!-- Dependências Oficiais do TensorSpace: Three.js r98, Tween.js, TrackballControls e TFJS -->
  <script src="https://cdn.jsdelivr.net/npm/three@0.98.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.98.0/examples/js/controls/TrackballControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.98.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tweenjs/tween.js@18.6.4/dist/tween.umd.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.7.4/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/tensorspace@0.6.1/dist/tensorspace.min.js"></script>

  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: #0b0f19;
      color: #e2e8f0;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}

    /* Container 3D onde o TensorSpace renderiza o modelo */
    #tensorspace-container {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }}

    /* Painel superior com título e status */
    #header {{
      position: absolute;
      top: 16px;
      left: 20px;
      z-index: 10;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(10px);
      padding: 14px 20px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      max-width: 440px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    #header h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    #header p {{
      font-size: 0.82rem;
      color: #94a3b8;
      margin-top: 4px;
      line-height: 1.4;
    }}

    /* Barra Horizontal de Camadas (Topo Centralizado) */
    #layer-nav-bar {{
      position: absolute;
      top: 16px;
      left: 480px;
      right: 280px;
      z-index: 10;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(10px);
      padding: 10px 16px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      align-items: center;
      gap: 10px;
      overflow-x: auto;
      white-space: nowrap;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }}
    #layer-nav-bar::-webkit-scrollbar {{
      height: 4px;
    }}
    #layer-nav-bar::-webkit-scrollbar-thumb {{
      background: #334155;
      border-radius: 4px;
    }}
    .nav-label {{
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
      display: flex;
      align-items: center;
      gap: 6px;
      flex-shrink: 0;
    }}
    .layer-chips-wrap {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .layer-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 20px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #cbd5e1;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      white-space: nowrap;
    }}
    .layer-chip:hover {{
      background: rgba(51, 65, 85, 0.9);
      border-color: var(--chip-color, #38bdf8);
      color: #ffffff;
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}
    .chip-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      flex-shrink: 0;
    }}

    /* Painel interativo de desenho do dígito (MNIST 28x28) */
    #canvas-card {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      z-index: 10;
      background: rgba(15, 23, 42, 0.92);
      backdrop-filter: blur(10px);
      padding: 16px;
      border-radius: 14px;
      border: 1px solid rgba(56, 189, 248, 0.3);
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
      width: 250px;
    }}
    #canvas-card h3 {{
      font-size: 0.9rem;
      font-weight: 600;
      color: #f1f5f9;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .canvas-wrapper {{
      width: 200px;
      height: 200px;
      margin: 0 auto;
      background: #000000;
      border-radius: 8px;
      border: 2px solid #334155;
      cursor: crosshair;
      touch-action: none;
    }}
    #drawing-canvas {{
      width: 200px;
      height: 200px;
      image-rendering: pixelated;
    }}
    .button-group {{
      display: flex;
      gap: 8px;
      margin-top: 12px;
    }}
    .btn {{
      flex: 1;
      padding: 8px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-primary {{
      background: #0284c7;
      color: #ffffff;
    }}
    .btn-primary:hover {{
      background: #0369a1;
    }}
    .btn-secondary {{
      background: #334155;
      color: #cbd5e1;
    }}
    .btn-secondary:hover {{
      background: #475569;
    }}

    /* Painel de Resultados */
    #prediction-badge {{
      margin-top: 10px;
      background: #1e293b;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 0.85rem;
      text-align: center;
    }}
    #prediction-badge span {{
      font-size: 1.3rem;
      font-weight: 800;
      color: #38bdf8;
      margin-left: 6px;
    }}

    /* Controles de Câmera 3D e Informações no canto superior direito */
    #controls-card {{
      position: absolute;
      top: 16px;
      right: 20px;
      z-index: 10;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(8px);
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      font-size: 0.8rem;
      color: #94a3b8;
      width: 240px;
    }}
    #controls-card ul {{
      list-style: none;
      margin-top: 6px;
    }}
    #controls-card li {{
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    #controls-card kbd {{
      background: #334155;
      padding: 2px 6px;
      border-radius: 4px;
      color: #f1f5f9;
      font-size: 0.72rem;
    }}

    /* Modal / Drawer Flutuante de Detalhes da Camada */
    #layer-modal {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: blur(6px);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s ease;
    }}
    #layer-modal.active {{
      opacity: 1;
      pointer-events: auto;
    }}
    .modal-card {{
      background: #0f172a;
      border: 1px solid rgba(56, 189, 248, 0.4);
      border-radius: 16px;
      width: 90%;
      max-width: 560px;
      max-height: 88vh;
      overflow-y: auto;
      padding: 24px;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.8), 0 0 30px rgba(56, 189, 248, 0.15);
      transform: translateY(16px);
      transition: transform 0.25s ease;
    }}
    #layer-modal.active .modal-card {{
      transform: translateY(0);
    }}
    .modal-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      border-bottom: 1px solid #334155;
      padding-bottom: 14px;
      margin-bottom: 16px;
    }}
    .modal-title-wrap {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .modal-badge {{
      display: inline-block;
      align-self: flex-start;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}
    .modal-header h2 {{
      font-size: 1.25rem;
      font-weight: 700;
      color: #f8fafc;
    }}
    .modal-close {{
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 1.6rem;
      line-height: 1;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 6px;
      transition: all 0.15s;
    }}
    .modal-close:hover {{
      color: #ffffff;
      background: #334155;
    }}

    /* Grid de estatísticas rápidas da camada */
    .info-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }}
    .info-item {{
      background: #1e293b;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .info-label {{
      display: block;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
      margin-bottom: 4px;
    }}
    .info-value {{
      font-size: 0.88rem;
      font-weight: 600;
      color: #f1f5f9;
    }}
    .code-font {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: #38bdf8;
    }}

    /* Seções explicativas */
    .section-box {{
      background: rgba(30, 41, 59, 0.6);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 14px;
      border-left: 4px solid #38bdf8;
    }}
    .section-box.importance-box {{
      border-left-color: #f59e0b;
      background: rgba(245, 158, 11, 0.06);
    }}
    .section-box.role-box {{
      border-left-color: #10b981;
      background: rgba(16, 185, 129, 0.06);
    }}
    .section-box h4 {{
      font-size: 0.88rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .section-box p {{
      font-size: 0.84rem;
      line-height: 1.55;
      color: #cbd5e1;
    }}

    .modal-actions {{
      display: flex;
      gap: 10px;
      margin-top: 18px;
    }}

    /* Avisos de compatibilidade */
    .warning-box {{
      margin-top: 8px;
      padding: 8px;
      background: rgba(245, 158, 11, 0.1);
      border-left: 3px solid #f59e0b;
      font-size: 0.75rem;
      color: #fbbf24;
      border-radius: 4px;
    }}
    .warning-box ul {{
      margin-left: 16px;
      margin-top: 4px;
    }}

    /* Loading Overlay */
    #loading {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      z-index: 20;
      background: rgba(15, 23, 42, 0.95);
      padding: 24px 32px;
      border-radius: 12px;
      text-align: center;
      border: 1px solid #38bdf8;
      box-shadow: 0 0 30px rgba(56, 189, 248, 0.2);
    }}
    .spinner {{
      border: 3px solid rgba(56, 189, 248, 0.2);
      border-top: 3px solid #38bdf8;
      border-radius: 50%;
      width: 32px;
      height: 32px;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 12px auto;
    }}
    @keyframes spin {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>

  <!-- Container principal onde o modelo 3D é desenhado -->
  <div id="tensorspace-container"></div>

  <!-- Cabeçalho -->
  <div id="header">
    <h1>SCTec - Análise Preditiva com Python [T2] Módulo 2</h1>
    <p>Visualização 3D interativa da rede CNN Keras (MNIST).</p>
    {avisos_html}
  </div>

  <!-- Barra de Navegação Horizontal das Camadas -->
  <div id="layer-nav-bar">
    <span class="nav-label">Camadas da CNN:</span>
    <div class="layer-chips-wrap">
      {chips_bar_html}
    </div>
  </div>

  <!-- Instruções de Navegação 3D -->
  <div id="controls-card">
    <strong style="color: #f1f5f9;">Navegação 3D:</strong>
    <ul>
      <li><span>Girar Câmera</span> <kbd>Botão Esquerdo</kbd></li>
      <li><span>Zoom in / out</span> <kbd>Scroll / Roda</kbd></li>
      <li><span>Panorâmica (Pan)</span> <kbd>Botão Direito</kbd></li>
      <li><span>Inspecionar Camada</span> <kbd>Clique na Camada</kbd></li>
    </ul>
  </div>

  <!-- Painel de Desenho do Dígito MNIST -->
  <div id="canvas-card">
    <h3>
      <span>Desenhar Dígito</span>
      <small style="color: #64748b; font-size: 0.75rem;">28×28 px</small>
    </h3>
    <div class="canvas-wrapper">
      <canvas id="drawing-canvas" width="28" height="28"></canvas>
    </div>
    <div class="button-group">
      <button class="btn btn-primary" id="btn-predict">Classificar</button>
      <button class="btn btn-secondary" id="btn-clear">Limpar</button>
    </div>
    <div id="prediction-badge">
      Previsão: <span id="pred-digit">-</span> <small id="pred-prob" style="color: #94a3b8;"></small>
    </div>
  </div>

  <!-- Modal / Gaveta de Detalhes da Camada Clicada -->
  <div id="layer-modal">
    <div class="modal-card">
      <div class="modal-header">
        <div class="modal-title-wrap">
          <span class="modal-badge" id="modal-badge">Conv2D</span>
          <h2 id="modal-title">conv2d_1</h2>
        </div>
        <button class="modal-close" id="modal-close-btn" title="Fechar">&times;</button>
      </div>
      
      <div class="info-grid">
        <div class="info-item">
          <span class="info-label">Tipo Arquitetural</span>
          <span class="info-value" id="modal-type">-</span>
        </div>
        <div class="info-item">
          <span class="info-label">Formato do Tensor (Shape)</span>
          <span class="info-value code-font" id="modal-shape">-</span>
        </div>
        <div class="info-item">
          <span class="info-label">Parâmetros Aprendíveis</span>
          <span class="info-value" id="modal-params">0</span>
        </div>
        <div class="info-item">
          <span class="info-label">Função de Ativação</span>
          <span class="info-value" id="modal-activation">-</span>
        </div>
      </div>

      <div class="section-box importance-box">
        <h4>⭐ Importância no Modelo</h4>
        <p id="modal-importance"></p>
      </div>

      <div class="section-box role-box">
        <h4>🧠 O que ela faz no modelo</h4>
        <p id="modal-role"></p>
      </div>

      <div class="modal-actions">
        <button class="btn btn-primary" id="btn-toggle-3d-layer">
          <span id="btn-toggle-3d-text">Expandir / Recolher na Cena 3D</span>
        </button>
        <button class="btn btn-secondary" id="btn-modal-close-action">Fechar</button>
      </div>
    </div>
  </div>

  <!-- Overlay de Carregamento -->
  <div id="loading">
    <div class="spinner"></div>
    <div id="loading-text">Carregando TensorSpace e pesos do modelo...</div>
  </div>

  <script>
    // -----------------------------------------------------------------------
    // VARIÁVEIS GLOBAIS SEGURAS (Declaradas no topo para evitar ReferenceError)
    // -----------------------------------------------------------------------
    var model = null;
    var fallbackTfModel = null;
    var modoAlternativoAtivo = false;
    var camadaAbertaAtiva = null;
    var metadadosCamadas = {metadados_json};
    window.tspCamadasMapa = {{}};
    var threePlacasCamadas = [];

    // -----------------------------------------------------------------------
    // 1. Configuração do Canvas de Desenho MNIST (28x28)
    // -----------------------------------------------------------------------
    const rawCanvas = document.getElementById("drawing-canvas");
    const ctx = rawCanvas.getContext("2d", {{ willReadFrequently: true }});
    let isDrawing = false;

    function resetCanvas() {{
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, 28, 28);
      document.getElementById("pred-digit").innerText = "-";
      document.getElementById("pred-prob").innerText = "";
      if (window.indicador3DUltimaCamada) {{
        const sc = (model && model.scene) || window.activeThreeScene;
        if (sc) sc.remove(window.indicador3DUltimaCamada);
        window.indicador3DUltimaCamada = null;
      }}
      if (window.outputBars) {{
        window.outputBars.forEach(b => {{
          b.scale.set(1, 1, 1);
          b.position.y = -2.7;
          b.material.color.setHex(0x334155);
          if (b.material.emissive) b.material.emissive.setHex(0x000000);
        }});
      }}
      const navBtn = document.getElementById("nav-dense_1");
      if (navBtn) {{
        navBtn.style.borderColor = "";
        navBtn.style.boxShadow = "";
        navBtn.innerText = "dense_1";
      }}
      if (typeof model !== "undefined" && model && typeof model.clear === "function") {{
        try {{
          model.clear();
        }} catch (e) {{
          console.warn("[TensorSpace] Aviso ao limpar cena:", e);
        }}
      }}
    }}
    resetCanvas();

    function getCanvasCoords(evt) {{
      const rect = rawCanvas.getBoundingClientRect();
      const scaleX = rawCanvas.width / rect.width;
      const scaleY = rawCanvas.height / rect.height;
      const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
      const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
      return {{
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
      }};
    }}

    function startDraw(e) {{
      isDrawing = true;
      draw(e);
    }}
    function stopDraw() {{
      isDrawing = false;
      ctx.beginPath();
    }}
    function draw(e) {{
      if (!isDrawing) return;
      e.preventDefault();
      const pos = getCanvasCoords(e);
      ctx.lineWidth = 2.4;
      ctx.lineCap = "round";
      ctx.strokeStyle = "#ffffff";
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
    }}

    rawCanvas.addEventListener("mousedown", startDraw);
    rawCanvas.addEventListener("mouseup", stopDraw);
    rawCanvas.addEventListener("mousemove", draw);
    rawCanvas.addEventListener("mouseleave", stopDraw);
    rawCanvas.addEventListener("touchstart", startDraw);
    rawCanvas.addEventListener("touchend", stopDraw);
    rawCanvas.addEventListener("touchmove", draw);
    document.getElementById("btn-clear").addEventListener("click", resetCanvas);

    // -----------------------------------------------------------------------
    // 2. Inspeção Interativa das Camadas (Modal & Destaque)
    // -----------------------------------------------------------------------
    const layerModal = document.getElementById("layer-modal");
    
    function abrirDetalhesCamada(camadaId) {{
      if (!camadaId) return;
      const layerData = metadadosCamadas.find(c => c.id === camadaId || c.name === camadaId);
      if (!layerData) return;

      camadaAbertaAtiva = layerData.id;
      document.getElementById("modal-badge").innerText = layerData.display_type;
      document.getElementById("modal-badge").style.borderColor = layerData.color;
      document.getElementById("modal-badge").style.color = layerData.color;
      document.getElementById("modal-title").innerText = layerData.name;
      document.getElementById("modal-type").innerText = layerData.type;
      document.getElementById("modal-shape").innerText = layerData.shape;
      document.getElementById("modal-params").innerText = layerData.params.toLocaleString("pt-BR") + " parâmetros";
      document.getElementById("modal-activation").innerText = layerData.activation;
      document.getElementById("modal-importance").innerText = layerData.importance;
      document.getElementById("modal-role").innerText = layerData.role;

      // Verificar estado no TensorSpace
      const tspLayer = window.tspCamadasMapa[layerData.id];
      if (tspLayer && typeof tspLayer.openLayer === "function") {{
        document.getElementById("btn-toggle-3d-layer").style.display = "inline-flex";
        document.getElementById("btn-toggle-3d-text").innerText = tspLayer.isOpen ? "Recolher Camada na Cena 3D" : "Expandir Camada na Cena 3D";
      }} else {{
        document.getElementById("btn-toggle-3d-layer").style.display = "none";
      }}

      layerModal.classList.add("active");
    }}

    function fecharModalCamada() {{
      layerModal.classList.remove("active");
    }}

    document.getElementById("modal-close-btn").addEventListener("click", fecharModalCamada);
    document.getElementById("btn-modal-close-action").addEventListener("click", fecharModalCamada);
    layerModal.addEventListener("click", function(e) {{
      if (e.target === layerModal) fecharModalCamada();
    }});
    window.addEventListener("keydown", function(e) {{
      if (e.key === "Escape") fecharModalCamada();
    }});

    // Ação do botão de expansão 3D
    document.getElementById("btn-toggle-3d-layer").addEventListener("click", function() {{
      if (!camadaAbertaAtiva) return;
      const tspLayer = window.tspCamadasMapa[camadaAbertaAtiva];
      if (tspLayer && typeof tspLayer.openLayer === "function") {{
        if (tspLayer.isOpen) {{
          tspLayer.closeLayer();
          tspLayer.isOpen = false;
          document.getElementById("btn-toggle-3d-text").innerText = "Expandir Camada na Cena 3D";
        }} else {{
          tspLayer.openLayer();
          tspLayer.isOpen = true;
          document.getElementById("btn-toggle-3d-text").innerText = "Recolher Camada na Cena 3D";
        }}
      }}
    }});

    // -----------------------------------------------------------------------
    // 3. Construção da Rede TensorSpace com base na arquitetura real
    // -----------------------------------------------------------------------
    const container = document.getElementById("tensorspace-container");

    // Polyfill crítico de compatibilidade: Registra 'Functional' e intercepta fetch para sanitizar model.json
    (function() {{
      if (typeof tf !== "undefined" && tf.serialization) {{
        try {{
          const ModelClass = tf.LayersModel || tf.Model;
          if (ModelClass) {{
            class Functional extends ModelClass {{
              static get className() {{ return "Functional"; }}
            }}
            tf.serialization.registerClass(Functional);
            console.log("[TFJS Compat] Classe 'Functional' registrada com sucesso.");
          }}
        }} catch (e) {{
          console.warn("[TFJS Compat] Aviso ao registrar classe:", e);
        }}
      }}

      // Interceptor transparente de Fetch para garantir que InputLayer tenha EXCLUSIVAMENTE batchInputShape
      const origFetch = window.fetch;
      window.fetch = async function(...args) {{
        const response = await origFetch.apply(this, args);
        try {{
          const url = typeof args[0] === "string" ? args[0] : (args[0] && args[0].url ? args[0].url : "");
          if (url.includes("model.json")) {{
            const clone = response.clone();
            const data = await clone.json();
            if (data && data.modelTopology) {{
              if (data.modelTopology.class_name === "Functional") {{
                data.modelTopology.class_name = "Model";
              }}
              const layers = (data.modelTopology.config && data.modelTopology.config.layers) || [];
              let prevName = null;
              layers.forEach((l, idx) => {{
                if (!l) return;
                const cls = l.class_name;
                const c = l.config || {{}};
                l.config = c;
                const name = l.name || c.name;

                const bShape = c.batchInputShape || c.batch_input_shape || c.batch_shape || c.batchShape;
                const inShape = c.inputShape || c.input_shape || c.shape;

                if (cls === "InputLayer" || idx === 0) {{
                  let finalBatch = [null, 28, 28, 1];
                  if (bShape && Array.isArray(bShape)) {{
                    finalBatch = bShape;
                  }} else if (inShape && Array.isArray(inShape)) {{
                    finalBatch = [null, ...inShape];
                  }}
                  c.batchInputShape = finalBatch;

                  delete c.inputShape;
                  delete c.input_shape;
                  delete c.shape;
                  delete c.batch_shape;
                  delete c.batch_input_shape;
                  delete c.batchShape;

                  l.inbound_nodes = [];
                  prevName = name;
                  return;
                }} else {{
                  if (c.batchInputShape && c.inputShape) {{
                    delete c.inputShape;
                    delete c.input_shape;
                  }}
                }}

                // Sanitizar inbound_nodes para evitar 'Corrupted configuration, expected array for nodeData: [object Object]'
                const raw = l.inbound_nodes;
                let converted = [];

                if (Array.isArray(raw) && raw.length > 0) {{
                  for (const node of raw) {{
                    let conns = [];
                    if (node && typeof node === "object" && !Array.isArray(node)) {{
                      const args = node.args || [];
                      for (const arg of args) {{
                        if (arg && typeof arg === "object" && !Array.isArray(arg)) {{
                          const hist = (arg.config && arg.config.keras_history) || arg.keras_history;
                          if (Array.isArray(hist) && hist.length >= 1) {{
                            conns.push([String(hist[0]), Number(hist[1] || 0), Number(hist[2] || 0), {{}}]);
                          }} else if (arg.name) {{
                            conns.push([String(arg.name), 0, 0, {{}}]);
                          }}
                        }} else if (Array.isArray(arg) && arg.length >= 1) {{
                          conns.push([String(arg[0]), Number(arg[1] || 0), Number(arg[2] || 0), {{}}]);
                        }} else if (typeof arg === "string") {{
                          conns.push([arg, 0, 0, {{}}]);
                        }}
                      }}
                    }} else if (Array.isArray(node)) {{
                      for (const item of node) {{
                        if (Array.isArray(item)) {{
                          if (item.length > 0 && Array.isArray(item[0])) {{
                            for (const sub of item) {{
                              conns.push([String(sub[0]), Number(sub[1] || 0), Number(sub[2] || 0), {{}}]);
                            }}
                          }} else {{
                            conns.push([String(item[0]), Number(item[1] || 0), Number(item[2] || 0), item[3] || {{}}]);
                          }}
                        }} else if (item && typeof item === "object") {{
                          const hist = (item.config && item.config.keras_history) || item.keras_history;
                          if (Array.isArray(hist) && hist.length >= 1) {{
                            conns.push([String(hist[0]), Number(hist[1] || 0), Number(hist[2] || 0), {{}}]);
                          }}
                        }} else if (typeof item === "string") {{
                          conns.push([item, 0, 0, {{}}]);
                        }}
                      }}
                    }}
                    if (conns.length > 0) {{
                      converted.push(conns);
                    }}
                  }}
                }}

                if (converted.length === 0 && prevName) {{
                  converted = [ [ [prevName, 0, 0, {{}}] ] ];
                }}

                l.inbound_nodes = converted;
                prevName = name;
              }});

              if (layers.length > 0 && data.modelTopology.config) {{
                const firstLayer = layers[0].name || (layers[0].config && layers[0].config.name);
                const lastLayer = layers[layers.length - 1].name || (layers[layers.length - 1].config && layers[layers.length - 1].config.name);
                if (firstLayer) data.modelTopology.config.input_layers = [[firstLayer, 0, 0]];
                if (lastLayer) data.modelTopology.config.output_layers = [[lastLayer, 0, 0]];
              }}
              console.log("[TFJS Compat] model.json sanitizado dinamicamente pelo navegador.");
              return new Response(JSON.stringify(data), {{
                status: response.status,
                statusText: response.statusText,
                headers: response.headers
              }});
            }}
          }}
        }} catch (e) {{
          console.warn("[TFJS Compat] Aviso fetch interceptor:", e);
        }}
        return response;
      }};
    }})();

    function initTensorSpaceModel() {{
      try {{
        // Inicializar Sequential Container oficial do TensorSpace
        model = new TSP.models.Sequential(container);

{js_layers_code}

        console.log("[TensorSpace] Topologia configurada. Carregando pesos...");

        // Carregar pesos convertidos do diretório relativo 'converted_model'
        model.load({{
          type: "tfjs",
          url: "./converted_model/model.json",
          onProgress: function(fraction) {{
            document.getElementById("loading-text").innerText = 
              "Baixando tensores: " + Math.round(fraction * 100) + "%";
          }},
          onComplete: function() {{
            console.log("[TensorSpace] Pesos carregados com sucesso!");
          }},
          onError: function(err) {{
            console.warn("[TensorSpace] Erro ao carregar pesos no TensorSpace:", err);
            iniciarModoAlternativo(err);
          }}
        }});

        // Inicializar a renderização 3D do TensorSpace
        model.init(function() {{
          console.log("[TensorSpace] Inicialização 3D concluída!");
          document.getElementById("loading").style.display = "none";
          configurarRaycasterClique3D();
        }});

        // Timeout de proteção caso CDN ou inicialização WebGL demorem
        setTimeout(function() {{
          const ld = document.getElementById("loading");
          if (ld && ld.style.display !== "none" && !modoAlternativoAtivo) {{
            console.warn("[TensorSpace] Timeout de carregamento atingido, ativando renderizador direto.");
            iniciarModoAlternativo("Timeout de carregamento");
          }}
        }}, 6000);

      }} catch (err) {{
        console.warn("[TensorSpace] Fallback para renderização direta via Three.js / TFJS:", err);
        iniciarModoAlternativo(err);
      }}
    }}

    // -----------------------------------------------------------------------
    // 4. Detecção de Clique 3D na Cena (Raycasting)
    // -----------------------------------------------------------------------
    function configurarRaycasterClique3D() {{
      let downX = 0, downY = 0;
      container.addEventListener("mousedown", function(e) {{
        downX = e.clientX;
        downY = e.clientY;
      }});

      container.addEventListener("click", function(e) {{
        // Apenas processa se não foi arrasto da câmera (pan/rotação)
        if (Math.abs(e.clientX - downX) > 6 || Math.abs(e.clientY - downY) > 6) return;
        if (e.target.tagName !== "CANVAS" && e.target !== container) return;

        // Se houver cena Three.js do TensorSpace ou placas registradas
        let camera = null, scene = null;
        if (model && model.camera && model.scene) {{
          camera = model.camera;
          scene = model.scene;
        }} else if (window.activeThreeCamera && window.activeThreeScene) {{
          camera = window.activeThreeCamera;
          scene = window.activeThreeScene;
        }}

        if (!camera || !scene) return;

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const rect = container.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(scene.children, true);

        if (intersects && intersects.length > 0) {{
          for (let hit of intersects) {{
            let cur = hit.object;
            while (cur) {{
              if (cur.userData && cur.userData.layerId) {{
                abrirDetalhesCamada(cur.userData.layerId);
                return;
              }}
              // Checar se o nome do mesh bate com alguma camada
              if (cur.name) {{
                const achou = metadadosCamadas.find(c => cur.name.includes(c.id) || cur.name.includes(c.name));
                if (achou) {{
                  abrirDetalhesCamada(achou.id);
                  return;
                }}
              }}
              cur = cur.parent;
            }}
          }}
        }}
      }});
    }}

    // -----------------------------------------------------------------------
    // 5. Fallback Seguro caso o CDN do TensorSpace falhe ou tenha incompatibilidade
    // -----------------------------------------------------------------------
    async function iniciarModoAlternativo(erroOriginal) {{
      if (modoAlternativoAtivo) return;
      modoAlternativoAtivo = true;
      document.getElementById("loading-text").innerText = "Inicializando renderizador 3D integrado...";
      try {{
        // Carregar modelo diretamente com TFJS para inferência garantida
        fallbackTfModel = await tf.loadLayersModel("./converted_model/model.json");
        console.log("[TFJS] Modelo carregado com sucesso no modo autônomo!");
        document.getElementById("loading").style.display = "none";
        
        // Criar visualizador Three.js interativo estilizado
        criarVisualizador3DIntegrado();
      }} catch (tfErr) {{
        console.error("Erro ao carregar modelo:", tfErr);
        document.getElementById("loading-text").innerHTML = 
          "<span style='color:#f87171;'>Erro ao carregar modelo:</span><br><small>" + tfErr.message + "</small>";
      }}
    }}

    function criarVisualizador3DIntegrado() {{
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0f19);
      const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(0, 14, 48);

      window.activeThreeCamera = camera;
      window.activeThreeScene = scene;

      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      container.appendChild(renderer.domElement);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
      scene.add(ambientLight);
      const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
      dirLight.position.set(20, 40, 30);
      scene.add(dirLight);

      // Renderizar placas 3D representando cada camada da CNN
      const gridHelper = new THREE.GridHelper(70, 35, 0x1e293b, 0x0f172a);
      gridHelper.position.y = -8;
      scene.add(gridHelper);

      // Criar placas 3D identificadas para cada camada
      const camadasRenderizaveis = metadadosCamadas.filter(c => !c.type.includes("Dropout") && !c.type.includes("Achatamento"));
      const stepZ = 7;
      const startZ = -((camadasRenderizaveis.length - 1) * stepZ) / 2;

      camadasRenderizaveis.forEach((cmd, i) => {{
        const zPos = startZ + (i * stepZ);

        // Se for a última camada (dense_1 / classificação), renderizar os 10 neurônios verticais com identificação 0 a 9
        if (cmd.id === "dense_1" || i === camadasRenderizaveis.length - 1) {{
          const groupOutput = new THREE.Group();
          groupOutput.position.set(0, 0, zPos);
          groupOutput.userData = {{ layerId: cmd.id }};
          groupOutput.name = "layer_mesh_" + cmd.id;

          // Pedestal base da camada de saída
          const basePlat = new THREE.Mesh(
            new THREE.BoxGeometry(9.6, 0.25, 2.0),
            new THREE.MeshPhongMaterial({{ color: 0x1e293b, shininess: 80 }})
          );
          basePlat.position.y = -3.2;
          groupOutput.add(basePlat);

          window.outputBars = [];
          for (let d = 0; d < 10; d++) {{
            const bx = -4.05 + d * 0.9;
            const barGeo = new THREE.BoxGeometry(0.68, 0.8, 0.68);
            const barMat = new THREE.MeshPhongMaterial({{
              color: 0x334155,
              emissive: 0x000000,
              shininess: 90
            }});
            const barMesh = new THREE.Mesh(barGeo, barMat);
            barMesh.position.set(bx, -2.8, 0);
            groupOutput.add(barMesh);
            window.outputBars.push(barMesh);

            // Borda da barra
            const barEdge = new THREE.LineSegments(
              new THREE.EdgesGeometry(barGeo),
              new THREE.LineBasicMaterial({{ color: 0x64748b }})
            );
            barMesh.add(barEdge);

            // Badge circular do dígito 0 a 9 abaixo da barra
            const tc = document.createElement("canvas");
            tc.width = 96; tc.height = 96;
            const tctx = tc.getContext("2d");
            tctx.beginPath();
            tctx.arc(48, 48, 44, 0, Math.PI * 2);
            tctx.fillStyle = "#0f172a";
            tctx.fill();
            tctx.strokeStyle = "#38bdf8";
            tctx.lineWidth = 4;
            tctx.stroke();

            tctx.fillStyle = "#ffffff";
            tctx.font = "bold 52px sans-serif";
            tctx.textAlign = "center";
            tctx.textBaseline = "middle";
            tctx.fillText(d.toString(), 48, 48);

            const spTex = new THREE.CanvasTexture(tc);
            const sp = new THREE.Sprite(new THREE.SpriteMaterial({{ map: spTex, depthTest: false }}));
            sp.scale.set(0.72, 0.72, 1);
            sp.position.set(bx, -3.85, 0.6);
            groupOutput.add(sp);
          }}

          scene.add(groupOutput);
          return;
        }}

        const geo = new THREE.BoxGeometry(6, 6, 0.4);
        const mat = new THREE.MeshPhongMaterial({{
          color: cmd.color || 0x38bdf8,
          transparent: true,
          opacity: 0.85,
          shininess: 90
        }});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(0, 0, zPos);
        mesh.userData = {{ layerId: cmd.id }};
        mesh.name = "layer_mesh_" + cmd.id;
        scene.add(mesh);

        // Borda destacada
        const edges = new THREE.EdgesGeometry(geo);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0xffffff, linewidth: 2 }}));
        mesh.add(line);
      }});

      configurarRaycasterClique3D();

      // Loop de renderização
      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();

      window.addEventListener("resize", () => {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }});
    }}

    // -----------------------------------------------------------------------
    // 5.1 Indicador 3D Explícito do Dígito Previsto na Última Camada
    // -----------------------------------------------------------------------
    function atualizarIndicadorUltimaCamada(predIdx, maxProb, allProbs) {{
      const scene = (model && model.scene) || window.activeThreeScene;
      if (!scene) return;

      // Remover banner anterior se existir
      if (window.indicador3DUltimaCamada) {{
        scene.remove(window.indicador3DUltimaCamada);
        window.indicador3DUltimaCamada = null;
      }}

      // Localizar posição da última camada
      let targetX = 0;
      let targetY = 5.0;
      let targetZ = 18;

      let lastMesh = scene.getObjectByName("layer_mesh_dense_1") || scene.getObjectByName("dense_1");
      if (!lastMesh) {{
        scene.traverse(function(obj) {{
          if (obj.name && (obj.name.includes("dense_1") || obj.name.includes("output") || obj.name.includes("Output1d"))) {{
            lastMesh = obj;
          }}
        }});
      }}

      if (lastMesh) {{
        const bbox = new THREE.Box3().setFromObject(lastMesh);
        targetX = (bbox.min.x + bbox.max.x) / 2;
        targetY = bbox.max.y + 3.0;
        targetZ = (bbox.min.z + bbox.max.z) / 2;
      }}

      // Se houver as 10 barras da última camada renderizadas
      if (window.outputBars && window.outputBars.length === 10) {{
        for (let d = 0; d < 10; d++) {{
          const bar = window.outputBars[d];
          const probVal = allProbs ? (allProbs[d] || 0.02) : (d === predIdx ? maxProb : 0.02);
          const h = Math.max(0.4, probVal * 6.5);
          bar.scale.set(1, h / 0.8, 1);
          bar.position.y = -3.2 + h / 2;
          if (d === predIdx) {{
            bar.material.color.setHex(0x00f2fe);
            if (bar.material.emissive) {{
              bar.material.emissive.setHex(0x0284c7);
              bar.material.emissiveIntensity = 1.2;
            }}
            targetX = -4.05 + d * 0.9;
            targetY = -3.2 + h + 2.4;
          }} else {{
            bar.material.color.setHex(probVal > 0.1 ? 0x38bdf8 : 0x1e293b);
            if (bar.material.emissive) {{
              bar.material.emissive.setHex(0x000000);
              bar.material.emissiveIntensity = 0;
            }}
          }}
        }}
      }}

      // Criar Sprite 3D flutuante de alta visibilidade com seta indicando a classe prevista
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 200;
      const c = canvas.getContext("2d");
      if (c) {{
        // Fundo com acabamento neon
        c.fillStyle = "rgba(11, 15, 25, 0.95)";
        c.strokeStyle = "#00f2fe";
        c.lineWidth = 6;
        if (typeof c.roundRect === "function") {{
          c.roundRect(8, 8, 496, 130, 20);
        }} else {{
          c.strokeRect(8, 8, 496, 130);
        }}
        c.fill();
        c.stroke();

        // Seta apontando para baixo (para o neurônio do dígito previsto)
        c.beginPath();
        c.moveTo(226, 138);
        c.lineTo(256, 186);
        c.lineTo(286, 138);
        c.closePath();
        c.fillStyle = "#00f2fe";
        c.fill();

        // Títulos e identificação clara do dígito
        c.textAlign = "center";
        c.fillStyle = "#38bdf8";
        c.font = "bold 24px monospace, sans-serif";
        c.fillText("ÚLTIMA CAMADA (SAÍDA SOFTMAX)", 256, 42);

        c.fillStyle = "#ffffff";
        c.font = "bold 44px sans-serif";
        c.fillText("★ DÍGITO PREVISTO: " + predIdx + " ★", 256, 92);

        c.fillStyle = "#4ade80";
        c.font = "bold 22px monospace, sans-serif";
        c.fillText("(" + (maxProb * 100).toFixed(1) + "% de certeza)", 256, 124);

        const tex = new THREE.CanvasTexture(canvas);
        const mat = new THREE.SpriteMaterial({{ map: tex, depthTest: false }});
        const sprite = new THREE.Sprite(mat);
        sprite.scale.set(7.5, 3.0, 1);
        sprite.position.set(targetX, targetY, targetZ);
        scene.add(sprite);
        window.indicador3DUltimaCamada = sprite;
      }}

      // Destacar o botão da camada na barra de navegação
      const navBtn = document.getElementById("nav-dense_1");
      if (navBtn) {{
        navBtn.style.borderColor = "#00f2fe";
        navBtn.style.boxShadow = "0 0 14px rgba(0, 242, 254, 0.6)";
        navBtn.innerText = "★ dense_1: Dígito " + predIdx + " (" + (maxProb * 100).toFixed(1) + "%)";
      }}
    }}

    // -----------------------------------------------------------------------
    // 6. Inferência Interativa ao Clicar em "Classificar"
    // -----------------------------------------------------------------------
    async function classificarDigito() {{
      const imgData = ctx.getImageData(0, 0, 28, 28);
      const inputArr = new Float32Array(28 * 28);
      
      // Normalizar pixels para [0.0, 1.0]
      for (let i = 0; i < 28 * 28; i++) {{
        inputArr[i] = imgData.data[i * 4] / 255.0;
      }}

      // 1. Tentar inferência no TensorSpace com animação 3D de ativações
      if (model && typeof model.predict === "function") {{
        try {{
          model.predict(inputArr, function(output) {{
            if (!output || output.length === 0) return;
            let maxProb = -1;
            let predIdx = 0;
            for (let i = 0; i < output.length; i++) {{
              if (output[i] > maxProb) {{
                maxProb = output[i];
                predIdx = i;
              }}
            }}
            document.getElementById("pred-digit").innerText = predIdx;
            document.getElementById("pred-prob").innerText = "(" + (maxProb * 100).toFixed(1) + "%)";
            atualizarIndicadorUltimaCamada(predIdx, maxProb, output);
          }});
          return;
        }} catch (tspErr) {{
          console.warn("[TensorSpace] Falha ao atualizar animação 3D, usando fallback TFJS:", tspErr);
        }}
      }}

      // 2. Fallback direto TFJS caso o TensorSpace ainda não tenha carregado ou falhe
      try {{
        if (!fallbackTfModel && typeof tf !== "undefined" && tf.loadLayersModel) {{
          fallbackTfModel = await tf.loadLayersModel("./converted_model/model.json");
        }}
        if (fallbackTfModel) {{
          const tensor = tf.tensor4d(inputArr, [1, 28, 28, 1]);
          const preds = fallbackTfModel.predict(tensor);
          const data = await preds.data();
          let maxProb = -1;
          let predIdx = 0;
          for (let i = 0; i < data.length; i++) {{
            if (data[i] > maxProb) {{
              maxProb = data[i];
              predIdx = i;
            }}
          }}
          document.getElementById("pred-digit").innerText = predIdx;
          document.getElementById("pred-prob").innerText = "(" + (maxProb * 100).toFixed(1) + "%)";
          atualizarIndicadorUltimaCamada(predIdx, maxProb, data);
          tensor.dispose();
          preds.dispose();
        }}
      }} catch (e) {{
        console.error("Erro na predição TFJS:", e);
      }}
    }}

    document.getElementById("btn-predict").addEventListener("click", classificarDigito);

    // Iniciar após carregamento do DOM
    window.addEventListener("DOMContentLoaded", initTensorSpaceModel);
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"[OK] Arquivo gerado com sucesso: {html_path.resolve()} ({html_path.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Servidor HTTP Local com Tratamento de CORS
# ---------------------------------------------------------------------------
def obter_handler_servidor(diretorio_base):
    """Cria um handler HTTP customizado que adiciona cabeçalhos CORS e serve a pasta correta."""
    class ServidorTensorspace(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(diretorio_base), **kwargs)

        def do_GET(self):
            # Responder requisições automáticas do Chrome DevTools silenciosamente sem gerar 404
            if self.path.startswith("/.well-known/"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            super().do_GET()
            
        def end_headers(self):
            # Adicionar cabeçalhos CORS para permitir carregar os shards .bin e model.json
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            super().end_headers()
            
        def log_message(self, format, *args):
            # Manter logs limpos no terminal e omitir requisições automáticas de devtools
            if len(args) > 0 and ".well-known" in str(args[0]):
                return
            if len(args) > 1 and "200" not in str(args[1]):
                sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")

    return ServidorTensorspace


def iniciar_servidor_local(output_dir, porta=SERVER_PORT):
    """Inicia o servidor HTTP local na porta especificada e abre a página no navegador."""
    handler_class = obter_handler_servidor(output_dir)
    
    # Encontrar porta disponível caso 8000 esteja em uso
    porta_ativa = porta
    tentativas = 0
    httpd = None
    
    while tentativas < 10:
        try:
            httpd = socketserver.TCPServer((SERVER_HOST, porta_ativa), handler_class)
            break
        except OSError:
            porta_ativa += 1
            tentativas += 1
            
    if httpd is None:
        print(f"[ERRO] Não foi possível vincular o servidor HTTP às portas entre {porta} e {porta + 10}.")
        sys.exit(1)
        
    url = f"http://{SERVER_HOST}:{porta_ativa}/"
    
    print("\n" + "=" * 75)
    print(f"  [6/6] SERVIDOR LOCAL TENSORSPACE EM EXECUÇÃO!")
    print("=" * 75)
    print(f"  URL de Acesso: {url}")
    print(f"  Pasta Servida: {output_dir.resolve()}")
    print("-" * 75)
    print("  -> Pressione [Ctrl + C] a qualquer momento para encerrar o servidor.")
    print("=" * 75 + "\n")
    
    # Tentar abrir o navegador padrão automaticamente
    try:
        print(f"Abrindo navegador em: {url}...")
        webbrowser.open(url)
    except Exception as e:
        print(f"[AVISO] Não foi possível abrir o navegador automaticamente ({e}).")
        print(f"Abra manualmente o link no seu navegador: {url}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Encerrando servidor HTTP local por solicitação do usuário...")
    finally:
        httpd.server_close()
        print("[OK] Servidor finalizado com sucesso. Até logo!")


# ---------------------------------------------------------------------------
# Fluxo Principal de Execução
# ---------------------------------------------------------------------------
def main(modelo):
    """Função orquestradora que executa todas as etapas do fluxo."""

    model_path = Path("models/" + modelo + "/cnn_mnist.keras")

    # Etapa 1: Verificar ambiente Python e TensorFlow
    tf = verificar_ambiente()
    
    # Etapa 2: Verificar ou criar o arquivo do modelo models/V2/cnn_mnist.keras
    verificar_ou_criar_modelo(tf)
    
    # Criar pasta de saída
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Etapa 3: Carregar o modelo Keras
    print(f"\nCarregando modelo a partir de '{model_path}'...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("[OK] Modelo Keras carregado com sucesso!")
    except Exception as e:
        print(f"\n[ERRO FATAL] Falha ao carregar o modelo '{model_path}': {e}")
        print("Certifique-se de que o arquivo foi salvo com TensorFlow 2.x ou Keras 3 compatível.")
        sys.exit(1)
        
    # Etapa 4: Inspecionar arquitetura
    info_camadas, camadas_ignoradas = inspecionar_modelo(tf, model)
    
    # Etapa 5: Converter modelo para TensorSpace / TFJS
    converter_modelo_para_tensorspace(tf, model, OUTPUT_DIR)
    
    # Etapa 6: Gerar o arquivo index.html adaptado
    gerar_arquivo_html(info_camadas, camadas_ignoradas, OUTPUT_DIR)
    
    # Etapa 7: Iniciar o servidor HTTP local
    iniciar_servidor_local(OUTPUT_DIR, SERVER_PORT)
