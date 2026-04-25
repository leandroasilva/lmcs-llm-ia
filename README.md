# LMCS LLM IA - Chat Conversacional

Um modelo de linguagem LSTM (Long Short-Term Memory) em nível de caractere com interface de chat moderna estilo ChatGPT/DeepSeek, treinado com datasets conversacionais em português do HuggingFace.

## 🚀 Funcionalidades

- **Modelo LSTM Avançado**: Arquitetura LSTM com gonum/mat para operações matriciais otimizadas (BLAS)
- **Modo Conversacional**: Responde perguntas como assistente (não completa texto)
- **Dataset HuggingFace**: Brazilian Customer Service + datasets portugueses
- **Coletor Automático**: Baixa e combina múltiplos datasets automaticamente
- **Interface Moderna**: Chat estilo ChatGPT com design escuro e responsivo
- **Configurável**: Controle de temperatura, top-k sampling e tamanho de geração
- **Histórico**: Salva conversas no localStorage do navegador
- **API REST**: Endpoints JSON para integração com outros sistemas
- **Multi-threading**: Treinamento paralelo com 8 workers (GOMAXPROCS)
- **Treinamento Incremental**: Pause, teste e continue de onde parou
- **Checkpoint**: Modelo salva progresso automaticamente (épocas acumuladas)

## 📁 Estrutura do Projeto

```
lmcs-llm-ia/
├── main.go                 # Ponto de entrada da aplicação
├── config/
│   └── config.go          # Gerenciamento de configurações
├── model/
│   ├── model.go           # Estrutura e lógica do modelo LLM (legado)
│   ├── lstm.go            # Implementação LSTM com gonum/mat
│   └── utils.go           # Funções auxiliares (Softmax, Sample, PreprocessText)
├── api/
│   └── handlers.go        # Handlers HTTP da API (Modelo + LSTM)
├── static/
│   ├── index.html         # Frontend - Interface de chat
│   ├── style.css          # Estilos CSS modernos
│   └── app.js             # JavaScript (prompt conversacional)
├── dataset/
│   └── train.json         # Dataset combinado (JSON)
├── download_dataset.py    # Baixa dataset do HuggingFace
├── collect_portuguese_datasets_fast.py  # Coletor de datasets PT
├── start_conversational_training.sh     # Inicia treinamento
├── CONVERSATIONAL_MODE.md # Guia modo conversacional
├── config.json            # Configuração do projeto
└── livro.txt              # Dataset de treinamento (texto)
```

## 🛠️ Instalação e Execução

### Pré-requisitos

- Go 1.22 ou superior
- gonum/mat (dependência para operações matriciais LSTM)
- Python 3 (para baixar datasets do HuggingFace)
- Arquivo `livro.txt` com texto para treinamento (gerado automaticamente)

### 1. Clonar o repositório

```bash
cd /Volumes/Dock/workspace/lmcs/lmcs-llm-ia
```

### 2. Baixar Datasets (Opcional)

```bash
# Baixar dataset Brazilian Customer Service
python3 download_dataset.py

# OU coletar múltiplos datasets em português
python3 collect_portuguese_datasets_fast.py

# Os scripts geram automaticamente:
# - conversas.txt (dataset conversacional)
# - livro.txt (para treinamento)
# - dataset/train.json (formato estruturado)
```

### 3. Configurar

```bash
# Editar configurações (já existe config.json)
nano config.json
```

### 4. Executar

```bash
# Compilar (recomendado)
go build -o lmcs-llm

# Executar (carrega modelo existente ou treina novo)
./lmcs-llm

# OU usar script de início rápido
./start_conversational_training.sh

# Treinar mais épocas (modo incremental)
./lmcs-llm --train
```

### 5. Acessar

- **Frontend**: http://localhost:8080
- **API Health**: http://localhost:8080/api/health
- **API Generate**: http://localhost:8080/api/ask (POST)

## 📝 Configuração (config.json)

```json
{
  "training": {
    "epochs": 300,
    "learning_rate": 0.005,
    "batch_size": 256,
    "temperature": 0.7,
    "context_size": 200,
    "top_k": 40,
    "hidden_size": 256,
    "num_layers": 3,
    "use_lstm": true
  },
  "server": {
    "port": ":8080",
    "host": "localhost"
  },
  "paths": {
    "model_path": "lmcs-model.bin",
    "input_file": "livro.txt"
  }
}
```

## 🎨 Interface de Chat

### Recursos do Frontend

- **Sidebar**: Histórico de conversas salvas
- **Configurações**: Controle deslizante de temperatura e tamanho máximo
- **Chat**: Interface responsiva estilo ChatGPT
- **Sugestões**: Botões de sugestão para começar rapidamente
- **Auto-save**: Conversas salvas automaticamente no navegador

### Como Usar

1. Abra http://localhost:8080 no navegador
2. Clique em "Novo Chat" ou use uma sugestão
3. Digite sua mensagem e pressione Enter
4. Ajuste a temperatura na sidebar para mais/menos criatividade
5. Alterne entre conversas no histórico

## 🔌 API Endpoints

### Health Check

```bash
GET /api/health

Response:
{
  "status": "ok",
  "model": "LSTM (gonum): vocab=47, hidden=256, context=200, layers=3, epochs_trained=300, params=323375"
}
```

### Gerar Texto (Formato Conversacional)

```bash
POST /api/ask
Content-Type: application/json

{
  "seed": "Usuário: Qual é a capital do Brasil?\nAssistente: ",
  "length": 200,
  "temperature": 0.7,
  "top_k": 40
}

Response:
{
  "success": true,
  "result": "A capital do Brasil é Brasília, localizada na região Centro-Oeste do país."
}
```

### Exemplos de Uso

```bash
# curl (formato conversacional)
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"seed": "Usuário: Olá\nAssistente: ", "length": 100, "temperature": 0.7}'

# JavaScript
fetch('/api/ask', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    seed: 'Usuário: Como funciona LSTM?\nAssistente: ',
    length: 200,
    temperature: 0.7
  })
})
.then(res => res.json())
.then(data => console.log(data.result));
```

## 🧠 Sobre o Modelo

### Arquitetura LSTM

- **Tipo**: Character-level LSTM Language Model
- **Framework**: gonum/mat (BLAS-optimized matrix operations)
- **Camadas**: LSTM com gates (Input, Forget, Output, Cell)
- **Hidden Size**: 256 unidades (configurável)
- **Context Size**: 200 caracteres (configurável até 500)
- **Função de Ativação**: Sigmoid (gates), Tanh (cell), Softmax (output)
- **Otimização**: Stochastic Gradient Descent (SGD) paralelo
- **Loss**: Cross-entropy
- **Parâmetros**: ~323,375 pesos treináveis

### Dataset Conversacional

- **Fonte**: HuggingFace (Brazilian Customer Service + datasets PT)
- **Total**: 755+ conversas reais de atendimento
- **Diálogos**: 2.953 mensagens usuário + 2.947 respostas assistente
- **Formato**: "Usuário: [pergunta]\nAssistente: [resposta]"
- **Pré-processamento**: Formatação conversacional automática

### Coletor de Datasets

O projeto inclui scripts para baixar automaticamente datasets em português:

```bash
# Baixar dataset específico
python3 download_dataset.py

# Coletar TOP 20 datasets portugueses
python3 collect_portuguese_datasets_fast.py
```

Os scripts:
1. Buscam datasets em português na API do HuggingFace
2. Navegam automaticamente entre páginas
3. Baixam via datasets-server API
4. Extraem conversas no formato Usuário/Assistente
5. Combinam tudo em um único arquivo `dataset/train.json`
6. Geram `livro.txt` para treinamento

### Modo Conversacional (GPT/DeepSeek Style)

O modelo foi adaptado para funcionar como um assistente conversacional:

**Como funciona:**
1. Usuário digita: "Qual é a capital do Brasil?"
2. Frontend formata: "Usuário: Qual é a capital do Brasil?\nAssistente: "
3. Modelo gera resposta: "A capital do Brasil é Brasília..."
4. Código remove prompt e retorna apenas a resposta
5. Chat exibe resposta do assistente

**Detecção inteligente:**
- Modelo para ao detectar "\nUsuário:" (fim da resposta)
- Remove automaticamente o prompt/seed
- Limpa whitespace excessivo
- Retorna apenas conteúdo da resposta

### Treinamento Incremental

O modelo suporta **treinamento parcial e contínuo**:

```bash
# Ver épocas treinadas
./lmcs-llm
# Output: "📊 Status: 120 épocas treinadas"

# Treinar mais épocas (incremental)
./lmcs-llm --train
```

**Como funciona:**
- Modelo guarda contador de `EpochsTrained`
- Salvamento automático após cada sessão
- Flag `--train` ativa modo incremental
- Épocas são acumuladas: `total = existentes + novas`

### Performance Atual

- **Épocas**: 300
- **Context Size**: 200 caracteres (4x maior que antes)
- **Dataset**: ~6.000 mensagens conversacionais
- **Tempo Estimado**: ~30-45 minutos
- **Workers**: 8 threads paralelos (GOMAXPROCS)

### Temperatura

- **Baixa (0.1-0.5)**: Texto mais conservador e repetitivo
- **Média (0.6-1.0)**: Equilíbrio entre coerência e criatividade (recomendado)
- **Alta (1.1-2.0)**: Texto mais diversificado e imprevisível

## 🏗️ Melhores Práticas Go

O projeto segue as melhores práticas da linguagem Go:

- ✅ **Separação de responsabilidades**: Pacotes organizados (config, model, api)
- ✅ **Tratamento de erros**: Todos os erros verificados e propagados
- ✅ **Thread-safety**: Mutex para proteção de dados compartilhados
- ✅ **Interfaces implícitas**: Go way de polimorfismo
- ✅ **Documentação**: Comentários em tipos e funções exportados
- ✅ **Nomes exportados**: Convenção de nomenclatura Go

## 📊 Performance

### Treinamento

- **Dataset**: 6.000+ mensagens conversacionais (~829KB)
- **Velocidade**: ~5-8 segundos/época (8 workers, batch 256)
- **300 épocas**: ~30-45 minutos
- **Memória**: ~200-300MB durante treinamento
- **CPU**: 800% (8 cores em uso total)

### Inferência

- **Latência**: <50ms para gerar 100 caracteres
- **Qualidade**: Respostas conversacionais de 1-3 frases
- **Temperatura ótima**: 0.6-0.8 para conversas

### Evolução do Projeto

| Versão | Dataset | Context | Tipo | Parâmetros | Uso |
|--------|---------|---------|------|------------|-----|
| v1.0 | 643 KB livros | 15 chars | Text completion | 98,739 | Gerar texto |
| v2.0 | 5.26 MB livros | 15 chars | Text completion | 98,739 | Gerar texto |
| v3.0 | 36 MB livros | 50 chars | Text completion | 329,780 | Gerar texto |
| **v4.0** | **6K conversas** | **200 chars** | **Chat GPT-style** | **323,375** | **Conversar** |

## 🐛 Troubleshooting

### Porta já em uso

```bash
# Matar processo usando a porta
lsof -ti:8080 | xargs kill -9

# Reiniciar servidor
./lmcs-llm
```

### Modelo não carrega

```bash
# Deletar modelo corrompido e treinar novo
rm -f lmcs-model.bin
./lmcs-llm
```

### Coletar novos datasets

```bash
# Baixar mais datasets em português
python3 collect_portuguese_datasets_fast.py

# Ver dataset coletado
ls -lh dataset/train.json livro.txt

# Retreinar com novo dataset
./lmcs-llm --train
```

### Testar modelo treinado

```bash
# Via navegador
http://localhost:8080

# Ou via API (formato conversacional)
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"seed": "Usuário: Olá, como vai?\nAssistente: ", "length": 150, "temperature": 0.7}'
```

## 📄 Licença

MIT License

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📧 Contato

Projeto desenvolvido como demonstração de modelo de linguagem conversacional em Go.

---

⭐ Se este projeto foi útil, dê uma estrela no repositório!
