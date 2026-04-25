# LMCS LLM IA - Chat Interface

Um modelo de linguagem LSTM (Long Short-Term Memory) em nível de caractere com interface de chat moderna estilo ChatGPT, treinado com dataset expandido de 36MB em português.

## 🚀 Funcionalidades

- **Modelo LSTM Avançado**: Arquitetura LSTM com gonum/mat para operações matriciais otimizadas (BLAS)
- **Dataset Expandido**: 36MB de texto em português (27 PDFs + livros)
- **Interface Moderna**: Chat estilo ChatGPT com design escuro e responsivo
- **Configurável**: Controle de temperatura, top-k sampling e tamanho de geração
- **Histórico**: Salva conversas no localStorage do navegador
- **API REST**: Endpoints JSON para integração com outros sistemas
- **Multi-threading**: Treinamento paralelo com 8 workers (GOMAXPROCS)
- **Auto-detect**: Carrega modelo existente ou treina automaticamente

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
│   └── app.js             # JavaScript da aplicação
├── livros/                # Pasta para PDFs de treinamento
├── extract_pdfs.sh        # Script para extração de PDFs para txt
├── test_model.sh          # Script para testar geração de texto
├── config.json            # Configuração do projeto
└── livro.txt              # Dataset combinado com todos PDFs
```

## 🛠️ Instalação e Execução

### Pré-requisitos

- Go 1.22 ou superior
- gonum/mat (dependência para operações matriciais LSTM)
- Arquivo `livro.txt` com texto para treinamento (opcional, mas recomendado)
- pdftotext (poppler) para extração de PDFs: `brew install poppler`

### 1. Clonar o repositório

```bash
cd /Volumes/Dock/workspace/lmcs/lmcs-llm-ia
```

### 2. Preparar Dataset (Opcional)

```bash
# Extrair texto de PDFs na pasta livros/
chmod +x extract_pdfs.sh
./extract_pdfs.sh

# O script combina todos os PDFs + livro.txt existente
# Resultado: livro.txt com ~36MB de texto em português
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

# Executar
./lmcs-llm

# O modelo será:
# - Carregado de lmcs-model.bin (se existir)
# - Ou treinado automaticamente com livro.txt
```

### 5. Acessar

- **Frontend**: http://localhost:8080
- **API Health**: http://localhost:8080/api/health
- **API Generate**: http://localhost:8080/api/ask (POST)

## 📝 Configuração (config.json)

```json
{
  "training": {
    "epochs": 2000,           # Número de épocas de treinamento
    "learning_rate": 0.005,   # Taxa de aprendizado
    "batch_size": 512,        # Tamanho do batch
    "temperature": 0.7,       # Temperatura padrão (0.1-2.0)
    "context_size": 50,       # Tamanho do contexto (caracteres anteriores)
    "top_k": 40,              # Top-K sampling (diversidade controlada)
    "hidden_size": 256,       # Tamanho da camada oculta LSTM
    "num_layers": 3,          # Número de camadas LSTM (registro)
    "use_lstm": true          # Usar arquitetura LSTM (true) ou legado (false)
  },
  "server": {
    "port": ":8080",          # Porta do servidor
    "host": "localhost"       # Host
  },
  "paths": {
    "model_path": "lmcs-model.bin",  # Arquivo do modelo
    "input_file": "livro.txt"        # Arquivo de treinamento
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
  "model": "LMCS LLM"
}
```

### Gerar Texto

```bash
POST /api/ask
Content-Type: application/json

{
  "seed": "o rato roeu",    # Texto inicial (prompt)
  "length": 100,            # Número de caracteres a gerar
  "temperature": 0.7,       # Criatividade (0.1-2.0)
  "top_k": 40               # Top-K sampling
}

Response:
{
  "success": true,
  "result": "o rato roeu a roupa..."
}
```

### Exemplos de Uso

```bash
# curl
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"seed": "a", "length": 50, "temperature": 1.0}'

# JavaScript
fetch('/api/ask', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    seed: 'o',
    length: 100,
    temperature: 0.8
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
- **Context Size**: 50 caracteres de entrada
- **Função de Ativação**: Sigmoid (gates), Tanh (cell), Softmax (output)
- **Otimização**: Stochastic Gradient Descent (SGD) paralelo
- **Loss**: Cross-entropy
- **Parâmetros**: ~330,000 pesos treináveis

### Dataset

- **Fonte**: livros em português (PDFs) + texto original
- **Tamanho**: 36MB brutos → 32MB pós-processamento
- **Idioma**: 100% português (com acentos e ç)
- **Pré-processamento**: Normalização, filtragem de caracteres, lowercase

### Treinamento

O modelo LSTM é treinado para prever o próximo caractere em uma sequência:

1. **Input**: Texto pré-processado (livro.txt)
2. **Processamento**: Mini-batch gradient descent com 8 workers paralelos
3. **Arquitetura**: LSTM com gates para memória de longo prazo
4. **Output**: Distribuição de probabilidade sobre vocabulário (52 chars)
5. **Persistência**: Modelo salvo em formato binário (gob)

### Performance Atual

- **Épocas**: 2000
- **Loss Inicial**: ~2.85
- **Loss Final Esperado**: ~2.0-2.2 (após 2000 épocas)
- **Tempo Estimado**: ~4-5 horas (dependendo do hardware)
- **Workers**: 8 threads paralelos (GOMAXPROCS)

### Temperatura

- **Baixa (0.1-0.5)**: Texto mais conservador e repetitivo
- **Média (0.6-1.0)**: Equilíbrio entre coerência e criatividade
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

- **Dataset**: 32MB pós-processamento (~36M caracteres)
- **Velocidade**: ~8 segundos/época (8 workers, batch 512)
- **2000 épocas**: ~4.5 horas estimado
- **Memória**: ~200-300MB durante treinamento
- **CPU**: 800% (8 cores em uso total)

### Inferência

- **Latência**: <50ms para gerar 100 caracteres
- **Qualidade**: Texto coerente por 15-30 caracteres
- **Temperatura ótima**: 0.5-0.7 para português

### Evolução do Projeto

| Versão | Dataset | Hidden | Context | Parâmetros | Loss Final |
|--------|---------|--------|---------|------------|------------|
| v1.0 | 643 KB | 128 | 15 | 98,739 | 2.580 |
| v2.0 | 5.26 MB | 128 | 15 | 98,739 | 2.393 |
| **v3.0** | **36 MB** | **256** | **50** | **329,780** | **~2.0** (em treino) |

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

### Dataset pequeno ou em outro idioma

```bash
# Extrair PDFs da pasta livros/
chmod +x extract_pdfs.sh
./extract_pdfs.sh

# Verificar tamanho do dataset
wc -c livro.txt
# Deve mostrar ~36MB
```

### Testar modelo treinado

```bash
# Executar script de teste
chmod +x test_model.sh
./test_model.sh

# Ou testar manualmente
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"seed": "era uma vez", "length": 100, "temperature": 0.6}'
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

Projeto desenvolvido como demonstração de modelo de linguagem em Go.

---

⭐ Se este projeto foi útil, dê uma estrela no repositório!
