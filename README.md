# LMCS LLM IA - Chat Interface

Um modelo de linguagem em nível de caractere com interface de chat moderna estilo ChatGPT.

## 🚀 Funcionalidades

- **Modelo de Linguagem**: Rede neural em nível de caractere treinada com backpropagation
- **Interface Moderna**: Chat estilo ChatGPT com design escuro e responsivo
- **Configurável**: Controle de temperatura e tamanho máximo de geração
- **Histórico**: Salva conversas no localStorage do navegador
- **API REST**: Endpoints JSON para integração com outros sistemas
- **Multi-threading**: Treinamento otimizado com mini-batches

## 📁 Estrutura do Projeto

```
lmcs-llm-ia/
├── main.go                 # Ponto de entrada da aplicação
├── config/
│   └── config.go          # Gerenciamento de configurações
├── model/
│   ├── model.go           # Estrutura e lógica do modelo LLM
│   └── utils.go           # Funções auxiliares (Softmax, Sample)
├── api/
│   └── handlers.go        # Handlers HTTP da API
├── static/
│   ├── index.html         # Frontend - Interface de chat
│   ├── style.css          # Estilos CSS modernos
│   └── app.js             # JavaScript da aplicação
├── config.example.json    # Exemplo de configuração
├── config.json            # Configuração local (criar a partir do exemplo)
└── livro.txt              # Texto para treinamento
```

## 🛠️ Instalação e Execução

### Pré-requisitos

- Go 1.22 ou superior
- Arquivo `livro.txt` com texto para treinamento (opcional)

### 1. Clonar o repositório

```bash
cd /Volumes/Dock/workspace/lmcs/lmcs-llm-ia
```

### 2. Configurar

```bash
# Copiar exemplo de configuração
cp config.example.json config.json

# Editar configurações conforme necessário
nano config.json
```

### 3. Executar

```bash
# Usando go run (desenvolvimento)
CGO_ENABLED=0 go run main.go

# Ou compilar e executar (produção)
CGO_ENABLED=0 go build -o lmcs-llm-server
./lmcs-llm-server
```

### 4. Acessar

- **Frontend**: http://localhost:8080
- **API Health**: http://localhost:8080/api/health
- **API Generate**: http://localhost:8080/api/ask (POST)

## 📝 Configuração (config.json)

```json
{
  "training": {
    "epochs": 50,           # Número de épocas de treinamento
    "learning_rate": 0.01,  # Taxa de aprendizado
    "batch_size": 32,       # Tamanho do batch
    "temperature": 0.8      # Temperatura padrão (0.1-2.0)
  },
  "server": {
    "port": ":8080",        # Porta do servidor
    "host": "localhost"     # Host
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
  "seed": "o",           # Caractere inicial
  "length": 100,         # Número de caracteres a gerar
  "temperature": 0.8     # Criatividade (0.1-2.0)
}

Response:
{
  "success": true,
  "result": "o rato roeu..."
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

### Arquitetura

- **Tipo**: Character-level Language Model
- **Camadas**: Single-layer neural network
- **Função de Ativação**: Softmax
- **Otimização**: Stochastic Gradient Descent (SGD)
- **Loss**: Cross-entropy

### Treinamento

O modelo é treinado para prever o próximo caractere em uma sequência:

1. **Input**: Texto bruto (ex: livro.txt)
2. **Processamento**: Mini-batch gradient descent
3. **Output**: Matriz de pesos caractere → caractere
4. **Persistência**: Modelo salvo em formato binário

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

- **Treinamento**: ~1.5s por época (643K caracteres, 50 épocas = ~75s)
- **Inferência**: <10ms para gerar 100 caracteres
- **Memória**: ~50MB para modelo com 132 caracteres únicos

## 🐛 Troubleshooting

### Porta já em uso

```bash
# Matar processo usando a porta
lsof -ti:8080 | xargs kill -9
```

### Erro de compilação no macOS

```bash
# Desabilitar CGO
CGO_ENABLED=0 go run main.go
```

### Modelo não carrega

Verifique se o arquivo `lmcs-model.bin` existe e não está corrompido. Delete-o para criar um novo.

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
