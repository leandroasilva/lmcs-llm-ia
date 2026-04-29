# GPU Training for LMCS LLM

## Visão Geral

Esta pasta contém a implementação de treinamento com GPU acelerado por PyTorch para o modelo LMCS Transformer.

### Dispositivos Suportados

| Dispositivo | Plataforma | Backend | Status |
|-------------|-----------|---------|--------|
| **AMD RX 550** | macOS | Metal/MPS | ✅ Suportado |
| **Apple Silicon (M1/M2/M3)** | macOS | MPS | ✅ Suportado |
| **AMD ROCm** | Linux | ROCm/CUDA | ⚠️ Experimental |
| **CPU** | Qualquer | CPU | ✅ Fallback |

## Instalação

```bash
# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Uso

### Treinamento Automático (Recomendado)

Detecta automaticamente a melhor GPU disponível:

```bash
python train_gpu.py --device auto --epochs 300 --batch-size 16
```

### Forçar Device Específico

```bash
# Usar GPU Apple Silicon (MPS)
python train_gpu.py --device mps --epochs 300

# Usar CPU
python train_gpu.py --device cpu --epochs 100
```

### Com Configuração Personalizada

```bash
python train_gpu.py \
  --device auto \
  --epochs 300 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --d-model 512 \
  --n-layers 6 \
  --n-heads 8 \
  --max-seq-len 256
```

### Resumir Treinamento

```bash
python train_gpu.py --resume ./checkpoints/model.pt --epochs 500
```

### Exportar para Go

Após treinamento, exporte o modelo para formato JSON compatível com Go:

```bash
python train_gpu.py --export-go ./checkpoints/model.pt --output-path ../config.trained.json
```

## Estrutura de Arquivos

```
training_gpu/
├── train_gpu.py          # Script principal de treinamento
├── requirements.txt      # Dependências Python
├── config.train.json     # Configuração do treinamento
├── src/                  # Código fonte Python
├── kernels/              # Kernels OpenCL (se necessário)
├── configs/              # Configurações de treinamento
├── checkpoints/          # Checkpoints salvos
│   ├── model.pt          # Checkpoint atual
│   ├── model_best.pt     # Melhor modelo
│   └── model_epoch_X.pt  # Checkpoints por epoch
└── docs/                 # Documentação
```

## Integração com Projeto Go

### Fluxo de Trabalho

1. **Treinar com GPU (Python)**
   ```bash
   cd training_gpu
   python train_gpu.py --device auto --epochs 300
   ```

2. **Exportar para Go**
   ```bash
   python train_gpu.py --export-go ./checkpoints/model_best.pt \
                       --output-path ../config.trained.json
   ```

3. **Usar no Go**
   ```bash
   cd ..
   ./lmcs-llm serve --config config.trained.json
   ```

### Formato de Exportação

O modelo exportado é um JSON contendo:
- `config`: Configurações do modelo
- `weights`: Todos os pesos convertidos para listas Python/JSON

## Performance Esperada

### AMD RX 550 (4GB VRAM)

| Configuração | Batch Size | Tempo/Epoch | VRAM Usage |
|--------------|------------|-------------|------------|
| Small (256)  | 16         | ~30s        | ~1.5GB     |
| Medium (512) | 8          | ~45s        | ~2.5GB     |
| Large (768)  | 4          | ~60s        | ~3.5GB     |

### Apple Silicon M1/M2/M3

| Configuração | Batch Size | Tempo/Epoch | VRAM Usage |
|--------------|------------|-------------|------------|
| Small (256)  | 32         | ~15s        | ~1GB       |
| Medium (512) | 16         | ~25s        | ~2GB       |
| Large (768)  | 8          | ~40s        | ~3GB       |

## Troubleshooting

### MPS não disponível

Se receber erro "MPS backend not available":

```bash
# Verificar versão do PyTorch
python -c "import torch; print(torch.__version__)"

# Deve ser >= 2.0.0
# Se não for, reinstale:
pip install --upgrade torch
```

### Out of Memory

Se receber erro de memória:

```bash
# Reduzir batch size
python train_gpu.py --batch-size 8

# Ou usar gradiente accumulation
python train_gpu.py --batch-size 8 --grad-accum 4
```

### AMD GPU no Linux

Para AMD GPUs no Linux, use ROCm:

```bash
# Instalar PyTorch com ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6

# Ou usar via Docker
docker run --device=/dev/kfd --device=/dev/dri \
  -v $(pwd):/workspace \
  rocm/pytorch:latest \
  python train_gpu.py --device cuda
```

## Métricas e Logging

### TensorBoard

Para visualizar métricas:

```bash
tensorboard --logdir ./logs
```

Acesse http://localhost:6006

### Logs em Arquivo

Os logs são salvos em:
- `./logs/training.log` - Log completo
- `./logs/metrics.json` - Métricas por epoch

## Próximos Passos

- [ ] Suporte a Mixed Precision (FP16)
- [ ] Distributed Training
- [ ] Export ONNX para inferência mais rápida
- [ ] Quantização para deploy
- [ ] Integração com Weights & Biases

## Links Úteis

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [AMD ROCm](https://rocm.docs.amd.com/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
