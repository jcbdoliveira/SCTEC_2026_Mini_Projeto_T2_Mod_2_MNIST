#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Visualizador 3D TensorSpace para Redes Convolucionais Keras (MNIST) - Keras 3 Fix
=============================================================================
Este módulo é importado pelo app3D.py e expõe a função main(modelo).
"""

import os
# Força o Keras a usar o TensorFlow como backend para evitar conflitos de versão
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import json
import webbrowser
from pathlib import Path
import http.server
import socketserver
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Configurações Globais do Projeto
# ---------------------------------------------------------------------------
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
    print(f"Verificando versão do Python: {sys.version.split()}...")
    
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
        print("    pip install tensorflow tensorflowjs numpy")
        print("!" * 70 + "\n")
        sys.exit(1)

def verificar_ou_criar_modelo(tf, model_path):
    """Verifica se o arquivo do modelo existe. Se não existir, oferece criar um de exemplo."""
    print(f"\nVerificando arquivo do modelo: '{model_path}'...")
    
    if model_path.exists():
        print(f"[OK] Arquivo encontrado: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
        return True
    
    print(f"\n[AVISO] O arquivo '{model_path}' não foi encontrado!")
    print("Deseja criar um modelo CNN MNIST de exemplo compatível agora?")
    print("1. Sim, treinar rapidamente uma CNN MNIST leve (1 época, ~15 seg)")
    print("2. Sim, gerar e salvar arquitetura LeNet-5 com pesos pré-inicializados (instantâneo)")
    print("3. Não, vou colocar meu próprio arquivo 'cnn_mnist.keras' na pasta do modelo correspondente")
    
    try:
        opcao = input("\nEscolha uma opção [1/2/3] (padrão: 2): ").strip() or "2"
    except (EOFError, KeyboardInterrupt):
        opcao = "2"
    
    if opcao == "1":
        criar_modelo_treinado(tf, model_path)
        return True
    elif opcao == "2":
        criar_modelo_instantaneo(tf, model_path)
        return True
    else:
        print(f"\nPor favor, copie o seu arquivo para '{model_path.resolve()}' e execute novamente.")
        sys.exit(0)


def criar_modelo_instantaneo(tf, model_path):
    """Cria uma CNN MNIST estilo LeNet e salva no caminho especificado."""
    print(f"\n--> Criando arquitetura CNN LeNet para MNIST em {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
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


def criar_modelo_treinado(tf, model_path):
    """Treina rapidamente uma CNN no dataset MNIST e salva o modelo no caminho especificado."""
    print(f"\n--> Baixando MNIST e realizando treino rápido (1 época) em {model_path}...")
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
    """Inspeciona as camadas do modelo, exibe tabela detalhada e detecta compatibilidade."""
    print("\nInspecionando arquitetura do modelo...")
    print("-" * 75)
    model.summary()
    print("-" * 75)
    
    print("\nTABELA DETALHADA DE CAMADAS:")
    print(f"{'#':<3} | {'Nome da Camada':<18} | {'Tipo':<16} | {'Formato Saída':<16} | {'Params':<8} | {'TensorSpace'}")
    print("-" * 88)
    
    info_camadas = []
    camadas_ignoradas = []
    
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
    
    tipos_nao_visuais = {
        "Dropout": "Ignorada visualmente",
        "BatchNormalization": "Ignorada visualmente",
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
    return info_camadas, camadas_ignoradas


def obter_formato_entrada(model):
    """Infere HxWxC da entrada Keras, sem assumir MNIST/28x28."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0] if shape else None
    if not shape or len(shape) < 3:
        raise ValueError(f"Não foi possível inferir o formato de entrada: {shape}")
    dims = [int(v) if v is not None else None for v in shape[-3:]]
    height, width, channels = dims
    if not height or not width:
        raise ValueError(f"Altura/largura da entrada precisam ser fixas: {shape}")
    return {"height": height, "width": width, "channels": channels or 1, "shape": [height, width, channels or 1]}


def obter_unidades_saida(model):
    """Obtém o número de classes/unidades da saída do modelo."""
    shape = getattr(model, "output_shape", None)
    if isinstance(shape, list):
        shape = shape[-1] if shape else None
    units = shape[-1] if shape and len(shape) else None
    if units is None:
        for layer in reversed(model.layers):
            if hasattr(layer, "units"):
                units = layer.units
                break
    return int(units) if units else 10


def nome_rotulo_saida(units):
    return [str(i) for i in range(int(units))]

# ---------------------------------------------------------------------------
# Conversão do Modelo para TensorFlow.js / TensorSpace (Keras 3 Adaptado)
# ---------------------------------------------------------------------------
def converter_modelo_para_tensorspace(tf, model, output_dir):
    """Orquestra a exportação usando o conversor nativo purificado."""
    print(f"\nConvertendo modelo para formato compatível com TensorSpace...")
    model_export_dir = output_dir / "converted_model"
    model_export_dir.mkdir(parents=True, exist_ok=True)
    
    # Executa o exportador nativo que filtra as camadas de normalização/dropout de forma segura
    exportar_tfjs_nativo(model, model_export_dir)

    # Exportar especificação JSON da arquitetura para a renderização dinâmica
    spec = gerar_especificacao_arquitetura(model)
    with open(model_export_dir / "model_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)        
        
    print(f"[OK] Arquivos gerados em '{model_export_dir}':")
    for f in model_export_dir.iterdir():
        print(f"     - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


def sanitizar_topologia_tfjs(model_topology):
    """Filtra completamente camadas não visuais (BN/Dropout) para manter o alinhamento com o TensorSpace."""
    if not isinstance(model_topology, dict):
        return model_topology
        
    if model_topology.get("class_name") == "Functional":
        model_topology["class_name"] = "Model"
        
    config = model_topology.get("config", {})
    if isinstance(config, dict):
        layers = config.get("layers", [])
        prev_layer_name = None
        cleaned_layers = []

        tipos_validos = ["Conv2D", "MaxPooling2D", "AveragePooling2D", "Flatten", "Dense", "Softmax"]

        for layer in layers:
            if not isinstance(layer, dict):
                continue
            cls_name = layer.get("class_name")
            
            # Pula completamente camadas que o TensorSpace não renderiza
            if cls_name not in tipos_validos and cls_name != "InputLayer":
                continue
                
            cfg = layer.get("config", {}) or {}
            layer_name = layer.get("name") or cfg.get("name")
            
            if cls_name == "InputLayer":
                continue
                
            properties_to_keep = [
                "name", "trainable", "filters", "kernel_size", "strides", 
                "padding", "data_format", "activation", "use_bias", 
                "pool_size", "units"
            ]
            
            cleaned_layer_config = {}
            for prop in properties_to_keep:
                if prop in cfg:
                    cleaned_layer_config[prop] = cfg[prop]
            
            layer["config"] = cleaned_layer_config

            if not cleaned_layers:
                layer["config"]["batchInputShape"] = [None, 28, 28, 1]
                layer["inbound_nodes"] = []
                prev_layer_name = layer_name
                cleaned_layers.append(layer)
                continue

            # Reconecta os inbound_nodes para pular as camadas de Dropout e BatchNormalization excluídas
            layer["inbound_nodes"] = [[[prev_layer_name, 0, 0, {}]]]
            prev_layer_name = layer_name
            cleaned_layers.append(layer)

        config["layers"] = cleaned_layers

        if cleaned_layers:
            config["input_layers"] = [[cleaned_layers[0].get("name"), 0, 0]]
            config["output_layers"] = [[cleaned_layers[-1].get("name"), 0, 0]]
                        
    return model_topology


def exportar_tfjs_nativo(model, target_dir):
    """Exporta um modelo funcional multi-saída para o TensorSpace.

    O TensorSpace 0.6.1 precisa receber um tensor por camada visualizada.
    Portanto, a topologia exportada transforma cada camada renderizável em
    uma saída do modelo, mantendo os mesmos pesos e nomes Keras.
    """
    import numpy as np

    tipos_validos = ["Conv2D", "MaxPooling2D", "AveragePooling2D", "Flatten", "Dense"]
    camadas = [layer for layer in model.layers
               if layer.__class__.__name__ in tipos_validos]
    if not camadas:
        raise ValueError("O modelo não possui camadas compatíveis com TensorSpace.")

    weights_data = bytearray()
    weights_manifest_entries = []
    for layer in camadas:
        weights = layer.get_weights()
        for index, w_array in enumerate(weights):
            w_array = np.asarray(w_array, dtype="float32")
            suffix = "kernel" if index == 0 else "bias" if index == 1 else f"param_{index}"
            weights_data.extend(w_array.tobytes())
            weights_manifest_entries.append({
                "name": f"{layer.name}/{suffix}",
                "shape": list(w_array.shape),
                "dtype": "float32"
            })

    bin_filename = "group1-shard1of1.bin"
    (target_dir / bin_filename).write_bytes(weights_data)
    entrada = obter_formato_entrada(model)
    input_name = "tensorspace_input"

    input_layer = {
        "name": input_name,
        "class_name": "InputLayer",
        "config": {
            "batch_input_shape": [None, *entrada["shape"]],
            "dtype": "float32",
            "sparse": False,
            "name": input_name
        },
        "inbound_nodes": []
    }
    topology_layers = [input_layer]
    previous = input_name
    for layer in camadas:
        config = dict(layer.get_config())
        config.pop("batch_input_shape", None)
        topology_layers.append({
            "name": layer.name,
            "class_name": layer.__class__.__name__,
            "config": config,
            "inbound_nodes": [] if previous is None else [[[previous, 0, 0, {}]]]
        })
        previous = layer.name

    outputs = [[layer.name, 0, 0] for layer in camadas]
    topology = {
        "class_name": "Model",
        "config": {
            "name": getattr(model, "name", "tensorspace_model"),
            "layers": topology_layers,
            "input_layers": [[input_name, 0, 0]],
            "output_layers": outputs
        }
    }
    model_json = {
        "format": "layers-model",
        "generatedBy": "TensorFlow.js Converter 1.7.4 + TensorSpace dynamic exporter",
        "convertedBy": "TensorSpace dynamic exporter",
        "modelTopology": topology,
        "weightsManifest": [{"paths": [bin_filename], "weights": weights_manifest_entries}]
    }
    with open(target_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)


def gerar_especificacao_arquitetura(model):
    """Extrai arquitetura, parâmetros e dimensões sem valores MNIST fixos."""
    layers_summary = []
    tipos_validos = ["Conv2D", "MaxPooling2D", "AveragePooling2D", "Flatten", "Dense"]
    for layer in model.layers:
        tipo = layer.__class__.__name__
        if tipo not in tipos_validos:
            continue
        cfg = layer.get_config()
        try:
            shape = layer.output.shape.as_list()
        except Exception:
            shape = None
        info = {
            "name": layer.name,
            "type": tipo,
            "params": int(layer.count_params()),
            "output_shape": shape,
            "config": cfg,
            "compatible": "Renderizável"
        }
        layers_summary.append(info)
    return {
        "model_name": getattr(model, "name", "tensorspace_model"),
        "input": obter_formato_entrada(model),
        "output_units": obter_unidades_saida(model),
        "total_params": int(model.count_params()),
        "layers": layers_summary
    }


def gerar_metadados_didaticos_camadas(info_camadas, input_info=None, output_units=10):
    """Gera metadados didáticos baseados na arquitetura real do modelo."""
    input_info = input_info or {"height": 28, "width": 28, "channels": 1}
    h, w, c = input_info["height"], input_info["width"], input_info["channels"]
    metadados = [{
        "id": "tensorspace_input", "name": "tensorspace_input",
        "type": "GreyscaleInput" if c == 1 else "RGBInput",
        "display_type": f"Entrada ({h}x{w}x{c})", "shape": f"[{h}, {w}, {c}]",
        "params": 0, "activation": "Normalização linear [0, 1]",
        "importance": "Recebe a imagem/desenho de entrada em tempo real.",
        "role": "Converte os pixels normalizados em dados para a primeira camada.",
        "color": "#0ea5e9"
    }]
    for cinfo in info_camadas:
        nome, tipo = cinfo["name"], cinfo["type"]
        cfg, params, shape = cinfo.get("config", {}) or {}, cinfo.get("params", 0), str(cinfo.get("output_shape") or "N/A")
        if tipo == "Conv2D":
            filters = cfg.get("filters", 1)
            metadados.append({"id": nome, "name": nome, "type": f"Convolucional 2D ({filters} filtros)", "display_type": f"Convolução ({filters} filtros)", "shape": shape, "params": params, "activation": str(cfg.get("activation", "linear")).upper(), "importance": "Extrai características locais da entrada.", "role": "Aplica filtros deslizantes para destacar bordas, formas e padrões.", "color": "#38bdf8"})
        elif tipo in ("MaxPooling2D", "AveragePooling2D"):
            avg = tipo == "AveragePooling2D"
            metadados.append({"id": nome, "name": nome, "type": "Average Pooling 2D" if avg else "Max Pooling 2D", "display_type": "AveragePooling (Redutor)" if avg else "MaxPooling (Redutor)", "shape": shape, "params": params, "activation": "Nenhuma", "importance": "Reduz a resolução preservando informação relevante.", "role": "Resume regiões da ativação e reduz o custo espacial.", "color": "#818cf8"})
        elif tipo == "Flatten":
            metadados.append({"id": nome, "name": nome, "type": "Flatten (Achatamento)", "display_type": "Flatten (Vetorizador)", "shape": shape, "params": params, "activation": "Nenhuma", "importance": "Converte mapas em vetor.", "role": "Prepara as características para as camadas densas.", "color": "#a855f7"})
        elif tipo == "Dense":
            units = int(cfg.get("units", output_units))
            final = units == int(output_units)
            metadados.append({"id": nome, "name": nome, "type": f"Dense ({units} saídas)" if final else f"Dense Oculta ({units} neurônios)", "display_type": "Saída (Veredito Final)" if final else f"Densa ({units} neurônios)", "shape": shape, "params": params, "activation": str(cfg.get("activation", "linear")).upper(), "importance": "Produz as probabilidades finais." if final else "Combina características de alto nível.", "role": "Calcula a decisão da rede." if final else "Aprende combinações não lineares de características.", "color": "#10b981" if final else "#6366f1"})
    return metadados


def gerar_codigo_js_tensorspace(info_camadas, input_info=None, output_units=10):
    """Gera camadas TensorSpace e parâmetros derivados do Keras."""
    input_info = input_info or {"height": 28, "width": 28, "channels": 1}
    h, w, c = input_info["height"], input_info["width"], input_info["channels"]
    input_type = "GreyscaleInput" if c == 1 else "RGBInput"
    labels = json.dumps(nome_rotulo_saida(output_units), ensure_ascii=False)
    lines = [
        "    model = new TSP.models.Sequential(container, { animeTime: 200, grid: { color: 0x30363d }, loader: 'tfjs' });",
        f"    model.add(new TSP.layers.{input_type}({{shape: [{h}, {w}, {c}], name: 'tensorspace_input'}}));"
    ]
    for cinfo in info_camadas:
        tipo, nome, cfg = cinfo["type"], cinfo["name"], cinfo.get("config", {}) or {}
        safe_name = json.dumps(nome)
        if tipo == "Conv2D":
            k, st = cfg.get("kernel_size", [3, 3])[0], cfg.get("strides", [1, 1])[0]
            lines.append(f"    model.add(new TSP.layers.Conv2d({{ name: {safe_name}, kernelSize: {k}, filters: {int(cfg.get('filters', 1))}, strides: {st}, padding: {json.dumps(cfg.get('padding', 'same'))} }}));")
        elif tipo in ("MaxPooling2D", "AveragePooling2D"):
            pool, st = cfg.get("pool_size", [2, 2])[0], cfg.get("strides", [2, 2])[0]
            lines.append(f"    model.add(new TSP.layers.Pooling2d({{ name: {safe_name}, poolSize: {pool}, strides: {st} }}));")
        elif tipo == "Flatten":
            lines.append(f"    model.add(new TSP.layers.Flatten({{ name: {safe_name} }}));")
        elif tipo == "Dense":
            units = int(cfg.get("units", output_units))
            if units == int(output_units):
                lines.append(f"    model.add(new TSP.layers.Output1d({{ name: {safe_name}, units: {units}, outputs: {labels} }}));")
            else:
                lines.append(f"    model.add(new TSP.layers.Dense({{ name: {safe_name}, units: {units} }}));")
    return "\n".join(lines)


def gerar_arquivo_html(info_camadas, camadas_ignoradas, output_dir):
    """Gera o index.html interativo na pasta de saída."""
    html_path = output_dir / "index.html"
    
    spec_path = output_dir / "converted_model" / "model_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    input_info = spec.get("input") or {"height": 28, "width": 28, "channels": 1}
    output_units = int(spec.get("output_units") or 10)
    js_layers_code = gerar_codigo_js_tensorspace(info_camadas, input_info, output_units)
    metadados_camadas = gerar_metadados_didaticos_camadas(info_camadas, input_info, output_units)
    metadados_json = json.dumps(metadados_camadas, ensure_ascii=False)
    input_h, input_w, input_c = input_info["height"], input_info["width"], input_info["channels"]
    model_version = int((output_dir / "converted_model" / "model.json").stat().st_mtime)
    
    chips_html_list = []
    for idx, item in enumerate(metadados_camadas, start=1):
        chip_html = f"""<button class="layer-chip" style="--chip-color: {item.get('color', '#38bdf8')};" onclick="abrirDetalhesCamada('{item['id']}')">
            <span class="chip-dot" style="background: {item.get('color', '#38bdf8')};"></span>
            <span class="chip-text">{idx} - {item['display_type']}</span>
        </button>"""
        chips_html_list.append(chip_html)
        
    chips_bar_html = "\n      ".join(chips_html_list)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CNN 3D Visualizer - TensorSpace</title>
  
  <script src="https://cdn.jsdelivr.net/npm/three@0.98.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.98.0/examples/js/controls/TrackballControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.98.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tweenjs/tween.js@18.6.4/dist/tween.umd.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.7.4/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/tensorspace@0.6.1/dist/tensorspace.min.js"></script>

  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #0b0f19;
      color: #e2e8f0;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #tensorspace-container {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }}
    #header {{
      position: absolute;
      top: 16px;
      left: 20px;
      z-index: 10;
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(10px);
      padding: 14px 20px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      max-width: 400px;
    }}
    #header h1 {{ font-size: 1.1rem; color: #38bdf8; }}
    #header p {{ font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }}
    
    #layer-nav-bar {{
      position: absolute;
      top: 16px;
      left: 440px;
      right: 20px;
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
      font-size: 0.75rem;
      cursor: pointer;
    }}
    .chip-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

    #canvas-card {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      z-index: 10;
      background: rgba(15, 23, 42, 0.9);
      padding: 16px;
      border-radius: 14px;
      border: 1px solid rgba(56, 189, 248, 0.3);
      width: 240px;
    }}
    .canvas-wrapper {{
      width: 150px;
      height: 150px;
      margin: 10px auto;
      background: #000;
      border: 2px solid #334155;
      border-radius: 8px;
    }}
    #drawing-canvas {{ width: 100%; height: 100%; cursor: crosshair; }}
    .btn {{
      padding: 8px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      width: 48%;
    }}
    .btn-primary {{ background: #0284c7; color: white; }}
    .btn-secondary {{ background: #334155; color: #cbd5e1; }}
    
    #prediction-badge {{
      margin-top: 10px;
      background: #1e293b;
      padding: 8px;
      border-radius: 6px;
      text-align: center;
      font-size: 0.85rem;
    }}
    #pred-digit {{ font-size: 1.4rem; font-weight: 800; color: #38bdf8; }}

    #layer-modal {{
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      z-index: 100;
      background: rgba(0,0,0,0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0; pointer-events: none;
      transition: opacity 0.2s ease;
    }}
    #layer-modal.active {{ opacity: 1; pointer-events: auto; }}
    .modal-card {{
      background: #0f172a;
      border: 1px solid #38bdf8;
      border-radius: 16px;
      width: 90%; max-width: 500px;
      padding: 20px;
    }}
    .modal-header {{ display: flex; justify-content: space-between; margin-bottom: 15px; }}
    .modal-close {{ background: none; border: none; color: #94a3b8; font-size: 1.5rem; cursor: pointer; }}
    .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }}
    .info-item {{ background: #1e293b; padding: 8px; border-radius: 6px; }}
    .info-label {{ font-size: 0.7rem; color: #94a3b8; }}
    .info-value {{ font-size: 0.85rem; font-weight: 600; }}
    .section-box {{ background: rgba(30, 41, 59, 0.5); padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #38bdf8; }}
    
    #loading {{
      position: absolute;
      top: 50%; left: 50%; transform: translate(-50%, -50%);
      background: rgba(15, 23, 42, 0.95);
      padding: 30px; border-radius: 12px; border: 1px solid #38bdf8;
      text-align: center; z-index: 20;
    }}
  </style>
</head>
<body>

  <div id="tensorspace-container"></div>

  <div id="header">
    <h1>Visualizador 3D - {spec.get("model_name", "Modelo Keras")}</h1>
    <p>Arquitetura e ativações em tempo real.</p>
  </div>

  <div id="layer-nav-bar">
    {chips_bar_html}
  </div>

  <div id="canvas-card">
    <h3 style="text-align:center;">Desenhe Aqui</h3>
    <div class="canvas-wrapper">
      <canvas id="drawing-canvas" width="{input_w}" height="{input_h}"></canvas>
    </div>
    <div style="display:flex; justify-content:space-between;">
      <button class="btn btn-primary" id="btn-predict">Prever</button>
      <button class="btn btn-secondary" id="btn-clear">Limpar</button>
    </div>
    <div id="prediction-badge">
      Previsão: <span id="pred-digit">-</span>
    </div>
  </div>

  <div id="layer-modal">
    <div class="modal-card">
      <div class="modal-header">
        <div>
          <h2 id="modal-title">Camada</h2>
        </div>
        <button class="modal-close" onclick="fecharModalCamada()">&times;</button>
      </div>
      <div class="info-grid">
        <div class="info-item">
          <span class="info-label">Tipo</span>
          <span class="info-value" id="modal-type">-</span>
        </div>
        <div class="info-item">
          <span class="info-label">Shape</span>
          <span class="info-value" id="modal-shape">-</span>
        </div>
      </div>
      <div class="section-box">
        <h4 style="color:#38bdf8; margin-bottom:4px;">Importância</h4>
        <p id="modal-importance" style="font-size:0.8rem;"></p>
      </div>
      <div class="section-box" style="border-left-color: #10b981;">
        <h4 style="color:#10b981; margin-bottom:4px;">O que faz</h4>
        <p id="modal-role" style="font-size:0.8rem;"></p>
      </div>
    </div>
  </div>

  <div id="loading">
    <div style="color:#38bdf8; margin-bottom:10px;">Carregando Modelo 3D...</div>
    <small style="color:#94a3b8;">Isso pode levar alguns segundos.</small>
  </div>

  <script>
    var model = null;
    var metadadosCamadas = {metadados_json};

    // --- CANVAS DRAWING ---
    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas.getContext("2d", {{ willReadFrequently: true }});
    let isDrawing = false;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function startDraw(e) {{ isDrawing = true; draw(e); }}
    function stopDraw() {{ isDrawing = false; ctx.beginPath(); }}
    function draw(e) {{
      if (!isDrawing) return;
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const point = e.touches ? e.touches[0] : e;
      const x = point.clientX - rect.left;
      const y = point.clientY - rect.top;
      
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.strokeStyle = "white";
      ctx.lineTo(x * (canvas.width/rect.width), y * (canvas.height/rect.height));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x * (canvas.width/rect.width), y * (canvas.height/rect.height));
    }}

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("touchstart", startDraw);
    canvas.addEventListener("touchend", stopDraw);
    canvas.addEventListener("touchmove", draw);

    document.getElementById("btn-clear").addEventListener("click", () => {{
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      document.getElementById("pred-digit").innerText = "-";
      if (model) model.clear();
    }});

    // --- TENSORSPACE INITIALIZATION ---
    const container = document.getElementById("tensorspace-container");
    
{js_layers_code}

    model.load({{
        type: "tfjs",
        url: "./converted_model/model.json?v={model_version}"
    }});

    model.init(function() {{
        document.getElementById("loading").style.display = "none";
        console.log("TensorSpace Inicializado!");
    }});

    // --- PREDICTION ---
    document.getElementById("btn-predict").addEventListener("click", function() {{
        const imgData = ctx.getImageData(0, 0, 28, 28);
        const inputData = [];
        for (let i = 0; i < imgData.data.length; i += 4) {{
            inputData.push(imgData.data[i] / 255.0);
        }}
        
        if (model) {{
            model.predict(inputData, function(output) {{
                let maxIndex = 0;
                let maxVal = output[0];
                for (let i = 1; i < output.length; i++) {{
                    if (output[i] > maxVal) {{
                        maxVal = output[i];
                        maxIndex = i;
                    }}
                }}
                document.getElementById("pred-digit").innerText = maxIndex;
            }});
        }}
    }});

    // --- MODAL INFO ---
    function abrirDetalhesCamada(id) {{
      const info = metadadosCamadas.find(c => c.id === id);
      if (!info) return;
      document.getElementById("modal-title").innerText = info.display_type;
      document.getElementById("modal-type").innerText = info.type;
      document.getElementById("modal-shape").innerText = info.shape;
      document.getElementById("modal-importance").innerText = info.importance;
      document.getElementById("modal-role").innerText = info.role;
      document.getElementById("layer-modal").classList.add("active");
    }}

    function fecharModalCamada() {{
      document.getElementById("layer-modal").classList.remove("active");
    }}
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] index.html gerado com sucesso em '{html_path}'")


# ---------------------------------------------------------------------------
# Servidor HTTP Local e Inicialização
# ---------------------------------------------------------------------------
def rodar_servidor_local():
    """Inicia o servidor HTTP local na pasta de saída e abre o navegador."""
    print(f"\nIniciando servidor web local na porta {SERVER_PORT}...")
    os.chdir(OUTPUT_DIR)
    
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", SERVER_PORT), Handler) as httpd:
            url = f"http://{SERVER_HOST}:{SERVER_PORT}"
            print("=" * 70)
            print(f"  🚀 SERVIDOR RODANDO COM SUCESSO!")
            print(f"  Acesse no seu navegador: {url}")
            print("  Para fechar o servidor, pressione: CTRL + C no terminal")
            print("=" * 70)
            
            webbrowser.open(url)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[OK] Servidor encerrado pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERRO] Não foi possível iniciar o servidor na porta {SERVER_PORT}: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Função Principal de Entrada (Chamada pelo app3D.py)
# ---------------------------------------------------------------------------
def main(modelo="V2"):
    """
    Função principal que orquestra todo o processo de carregamento, 
    inspeção, conversão e inicialização do servidor 3D.
    """
    model_path = Path(f"models/{modelo}/cnn_mnist.keras")
    
    tf = verificar_ambiente()
    verificar_ou_criar_modelo(tf, model_path)
    
    try:
        print(f"\n--> Carregando modelo '{model_path}'...")
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o arquivo do modelo: {e}")
        sys.exit(1)
        
    info_camadas, camadas_ignoradas = inspecionar_modelo(tf, model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    converter_modelo_para_tensorspace(tf, model, OUTPUT_DIR)
    gerar_arquivo_html(info_camadas, camadas_ignoradas, OUTPUT_DIR)
    rodar_servidor_local()


if __name__ == "__main__":
    main("V2")
