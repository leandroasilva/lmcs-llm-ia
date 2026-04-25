# LMCS LLM IA - Assistente Conversacional com Transformer

Um modelo de linguagem **Transformer Small** implementado do zero em Go, treinado com dataset enriquecido de atendimento ao cliente brasileiro, com interface de chat moderna estilo ChatGPT/DeepSeek.

## 🚀 Funcionalidades

- **Arquitetura Transformer**: Modelo Small com multi-head self-attention, positional encoding e feed-forward networks
- **Dataset Enriquecido**: Metadados completos (intent, sentiment, sector) para contexto rico
- **Tokenização por Palavras**: Vocabulário de 3,600+ palavras vs 47 caracteres (LSTM antigo)
- **Contexto Longo**: 256 tokens de contexto vs 50 caracteres
- **Interface Moderna**: Chat estilo ChatGPT com design escuro e responsivo
- **API REST**: Endpoints JSON para integração com outros sistemas
- **Treinamento Incremental**: Pause, teste e continue de onde parou
- **Checkpoint Automático**: Modelo salva progresso automaticamente
- **100% Go**: Sem dependências Python, scripts em Go puro
- **Configurável**: Controle de temperatura, top-k sampling e hiperparâmetros do Transformer

## 📁 Estrutura do Projeto

```
lmcs-llm-ia/
├── main.go                      # Ponto de entrada da aplicação
├── config/
│   └── config.go               # Gerenciamento de configurações
├── model/
│   ├── transformer.go          # Implementação Transformer completa
│   ├── model.go                # Modelo legado (softmax regression)
│   └── utils.go                # Funções auxiliares (Softmax, CrossEntropyLoss)
├── api/
│   └── handlers.go             # Handlers HTTP da API
├── dataset/
│   ├── downloader.go           # Download dataset simples
│   ├── download_enriched.go    # Download dataset COM metadados
│   └── data/
│       ├── train.txt           # Dataset simples (legado)
│       └── train_enriched.txt  # Dataset enriquecido (RECOMENDADO)
├── static/
│   ├── index.html              # Frontend - Interface de chat
│   ├── style.css               # Estilos CSS modernos
│   └── app.js                  # JavaScript (prompt conversacional)
├── config.json                 # Configuração atual do projeto
├── config.example.json         # Exemplo de configuração
├── go.mod                      # Módulo Go e dependências
└── lmcs-llm                    # Binário compilado
```

## 🛠️ Instalação e Execução

### Pré-requisitos

- Go 1.22 ou superior
- gonum/mat (operações matriciais otimizadas com BLAS)
- Conexão com internet (para download do dataset)

### 1. Compilar o Projeto

```bash
cd /Volumes/Dock/workspace/lmcs/lmcs-llm-ia

# Compilar
go build -o lmcs-llm
```

### 2. Baixar Dataset Enriquecido (Recomendado)

```bash
# Baixar dataset COM metadados (intent, sentiment, sector)
./lmcs-llm --download-enriched

# OU dataset simples (sem metadados)
./lmcs-llm --download-dataset
```

**Dataset Enriquecido inclui:**
- 755 conversas reais de atendimento
- 10 categorias de intent (compra, suporte, reclamação, etc.)
- 3 sentiments balanceados (positive, negative, neutral)
- 8 sectors (telecom, saúde, financeiro, etc.)
- Formato: `[INTENT:x] [SENTIMENT:y] [SECTOR:z]`

### 3. Configurar

```bash
# Configuração já está otimizada em config.json
# Para personalizar:
nano config.json
```

### 4. Executar e Treinar

```bash
# Executar (treina automaticamente se não houver modelo)
./lmcs-llm

# Treinar mais épocas (incremental)
./lmcs-llm --train

# Ver opções
./lmcs-llm --help
```

### 5. Acessar

- **Frontend Chat**: http://localhost:8080
- **API Health**: http://localhost:8080/api/health
- **API Generate**: POST http://localhost:8080/api/ask

## 📝 Configuração (config.json)

```json
{
  "training": {
    "epochs": 30,
    "learning_rate": 0.001,
    "batch_size": 16,
    "temperature": 0.7,
    "top_k": 30,
    "d_model": 128,
    "n_heads": 4,
    "num_layers": 2,
    "max_seq_len": 256,
    "ff_hidden": 256,
    "max_vocab": 5000
  },
  "server": {
    "port": ":8080",
    "host": "localhost"
  },
  "paths": {
    "model_path": "lmcs-model.bin",
    "input_file": "dataset/data/train_enriched.txt"
  }
}
```

### Parâmetros do Transformer

| Parâmetro | Descrição | Valor Padrão |
|-----------|-----------|--------------|
| `d_model` | Dimensão do modelo (embeddings) | 128 |
| `n_heads` | Número de attention heads | 4 |
| `num_layers` | Camadas do Transformer | 2 |
| `max_seq_len` | Tamanho máximo da sequência | 256 |
| `ff_hidden` | Hidden size do feed-forward | 256 |
| `max_vocab` | Tamanho máximo do vocabulário | 5000 |

## 🎨 Interface de Chat

### Recursos do Frontend

- **Sidebar**: Histórico de conversas salvas no localStorage
- **Configurações**: Controle deslizante de temperatura e top-k
- **Chat Responsivo**: Interface estilo ChatGPT com tema escuro
- **Sugestões**: Botões de prompts para começar rapidamente
- **Auto-save**: Conversas salvas automaticamente

### Como Usar

1. Abra http://localhost:8080 no navegador
2. Clique em "Novo Chat" ou use uma sugestão
3. Digite sua mensagem e pressione Enter
4. Ajuste a temperatura para mais/menos criatividade
5. Alterne entre conversas no histórico

## 🔌 API Endpoints

### Health Check

```bash
GET /api/health

Response:
{
  "status": "ok",
  "model": "Transformer",
  "vocab": 3613,
  "d_model": 128,
  "heads": 4,
  "layers": 2,
  "epochs": 30
}
```

### Gerar Texto

```bash
POST /api/ask
Content-Type: application/json

{
  "question": "Oi, tudo bem?",
  "temperature": 0.7,
  "top_k": 30
}

Response:
{
  "answer": "Olá! Tudo bem sim! Como posso ajudar você hoje?",
  "model": "Transformer",
  "elapsed_ms": 45,
  "vocab_size": 3613,
  "d_model": 128,
  "temperature": 0.7,
  "top_k": 30
}
```

### Exemplos de Uso

```bash
# Via curl
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quero contratar um plano", "temperature": 0.7}'

# Via JavaScript
fetch('/api/ask', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    question: 'Como funciona o suporte?',
    temperature: 0.7,
    top_k: 30
  })
})
.then(res => res.json())
.then(data => console.log(data.answer));
```

## 🧠 Sobre o Modelo

### Arquitetura Transformer Small

**Componentes Principais:**

1. **Embedding Layer**
   - Token Embedding: [vocab_size, d_model]
   - Positional Encoding: [max_seq_len, d_model]

2. **Multi-Head Self-Attention**
   - 4 attention heads em paralelo
   - Scaled dot-product attention
   - Q, K, V projections
   - Residual connections

3. **Feed-Forward Network**
   - Two-layer MLP: 128 → 256 → 128
   - ReLU activation
   - Layer normalization

4. **Output Layer**
   - Linear projection: [d_model, vocab_size]
   - Softmax para probabilidades

**Hiperparâmetros:**
- **Tipo**: Word-level Transformer Language Model
- **Vocabulário**: 3,613 palavras
- **Contexto**: 256 tokens
- **Parâmetros**: ~500K pesos treináveis
- **Inicialização**: Xavier/Glorot uniform

### Dataset Enriquecido

**Fonte:** HuggingFace - Brazilian Customer Service Conversations

**Estatísticas:**
- **755 conversas** completas
- **5,900 diálogos** (usuário + assistente)
- **873 KB** de dados de treinamento
- **146,792 tokens** após tokenização

**Metadados Incluídos:**

| Metadado | Valores | Distribuição |
|----------|---------|--------------|
| **Intent** | 10 categorias | compra, suporte, reclamação, etc. |
| **Sentiment** | 3 classes | positive (33%), negative (33%), neutral (33%) |
| **Sector** | 8 domínios | telecom, saúde, financeiro, etc. |

**Formato do Dataset:**

```
[INTENT:compra] [SENTIMENT:negative] [SECTOR:telecom]
Usuário: Oi, vc pode me ajudar? Quero contratar um plano de internet
Assistente: Beleza, posso te ajudar com isso! Qual é o problema?
Usuário: Não consigo avançar na compra, da erro
Assistente: Entendi, vamos tentar resolver isso...
```

**Benefícios dos Metadados:**
- ✅ Modelo aprende padrões por **intenção**
- ✅ Entende **tom emocional** da conversa
- ✅ Conhece o **domínio** específico
- ✅ Gera respostas mais **contextualizadas**
- ✅ Melhor **generalização** para novos cenários

### Download do Dataset

```bash
# Dataset enriquecido (RECOMENDADO)
./lmcs-llm --download-enriched

# Dataset simples
./lmcs-llm --download-dataset
```

**O download:**
1. Conecta na API do HuggingFace
2. Baixa 755 conversas com metadados
3. Processa e formata em português
4. Gera `train_enriched.txt` com contexto rico
5. Exibe estatísticas detalhadas

### Modo Conversacional

**Como Funciona:**

1. Usuário digita: "Quero contratar um plano"
2. Modelo processa com contexto de 256 tokens
3. Attention mechanism identifica padrões relevantes
4. Gera resposta token por token
5. Retorna resposta completa

**Detecção Inteligente:**
- Para ao detectar mudança de turno
- Remove tokens especiais automaticamente
- Limpa formatação excessiva
- Retorna apenas a resposta do assistente

### Treinamento Incremental

```bash
# Ver épocas treinadas
./lmcs-llm
# Output: "Época 30/30 - Loss: 2.345"

# Treinar mais épocas
./lmcs-llm --train
```

**Como Funciona:**
- Modelo guarda `EpochsTrained` no checkpoint
- Salvamento automático após cada sessão
- Learning rate decay exponencial (0.95^epoch)
- Épocas acumulativas

### Performance Esperada

**Treinamento:**
- **30 épocas**: ~15-30 minutos
- **Velocidade**: ~30-60 segundos/época
- **Memória**: ~500MB-1GB
- **CPU**: 800% (8 cores em uso total)

**Inferência:**
- **Latência**: <100ms para 100 tokens
- **Qualidade**: Respostas conversacionais coerentes
- **Temperatura ótima**: 0.6-0.8

### Temperatura e Sampling

| Temperatura | Comportamento | Uso Recomendado |
|-------------|---------------|-----------------|
| 0.1-0.5 | Conservador, repetitivo | Fatos, dados |
| **0.6-0.8** | **Equilibrado** | **Conversas (ideal)** |
| 0.9-1.2 | Criativo, variado | Brainstorming |
| 1.3-2.0 | Muito diversificado | Arte, poesia |

**Top-K Sampling:**
- Seleciona apenas dos K tokens mais prováveis
- Reduz nonsense e melhora coerência
- Valor recomendado: 30-40

## 🏗️ Arquitetura do Código

### Pacotes Go

```
main.go              → Entry point, CLI, training loop
config/config.go     → Config management, validation
model/transformer.go → Transformer architecture
model/utils.go       → Math utilities (Softmax, Loss)
api/handlers.go      → HTTP handlers, routing
dataset/*.go         → Dataset downloaders
```

### Boas Práticas Go

- ✅ **Separação de responsabilidades**: Pacotes organizados
- ✅ **Tratamento de erros**: Verificação e propagação
- ✅ **Thread-safety**: Mutex onde necessário
- ✅ **Interfaces implícitas**: Polimorfismo Go-style
- ✅ **Documentação**: Comentários em código exportado
- ✅ **100% Go**: Sem Python ou scripts externos

## 📊 Comparação: LSTM vs Transformer

| Feature | LSTM (Antigo) | Transformer (Atual) |
|---------|---------------|---------------------|
| **Tipo** | Character-level | Word-level |
| **Vocabulário** | 47 caracteres | 3,613 palavras |
| **Contexto** | 50 chars | 256 tokens |
| **Metadados** | ❌ Não | ✅ Sim (intent, sentiment, sector) |
| **Arquitetura** | LSTM gates | Multi-head attention |
| **Paralelização** | Sequencial | Paralela |
| **Qualidade** | Baixa | Alta |
| **Coerência** | Fraca | Forte |

## 🐛 Troubleshooting

### Porta já em uso

```bash
# Matar processo na porta 8080
lsof -ti:8080 | xargs kill -9

# Reiniciar
./lmcs-llm
```

### Modelo corrompido

```bash
# Remover e retreinar
rm -f lmcs-model.bin
./lmcs-llm
```

### Dataset não encontrado

```bash
# Baixar dataset enriquecido
./lmcs-llm --download-enriched

# Verificar arquivo
ls -lh dataset/data/train_enriched.txt
```

### Testar API

```bash
# Testar via curl
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Oi, tudo bem?", "temperature": 0.7}'

# Health check
curl http://localhost:8080/api/health
```

## 🎯 Roadmap Futuro

- [ ] Implementar backpropagation completa (BPTT)
- [ ] Adicionar gradient clipping
- [ ] Suporte a multi-turn conversations
- [ ] Fine-tuning por domínio específico
- [ ] Exportar modelo para ONNX
- [ ] Adicionar mais datasets multilíngues
- [ ] Implementar beam search decoding
- [ ] Adicionar caching de attention para inferência mais rápida

## 📄 Licença

MIT License

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add feature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📧 Contato

Projeto desenvolvido como demonstração de modelo Transformer em Go puro.

---

⭐ **Se este projeto foi útil, dê uma estrela no repositório!**
