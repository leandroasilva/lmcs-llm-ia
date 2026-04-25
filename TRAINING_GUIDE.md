# 🎯 Treinamento Incremental - Guia de Uso

## ✨ O que é Treinamento Incremental?

Você pode treinar seu modelo **parcialmente**, testar, e **continuar de onde parou**!

## 🚀 Fluxo de Trabalho

### 1️⃣ Primeiro Treinamento (40 épocas)
#### Incrementar treinamento sempre de 10 em 10 épocas.

```bash
# Treinar 40 épocas iniciais
./train_incremental.sh 40
```

**O que acontece:**
- Modelo é criado do zero
- Treina 40 épocas
- Salva em `lmcs-model.bin`
- Mostra loss final

---

### 2️⃣ Testar o Modelo

```bash
# Testar modelo atual
./train_incremental.sh
# Escolha opção 1
```

**O que acontece:**
- Carrega modelo existente
- Inicia servidor temporário
- Faz 2 testes de geração
- Mostra resultados

---

### 3️⃣ Continuar Treinamento (+40 épocas)

```bash
# Adicionar mais 40 épocas
./train_incremental.sh 40
# Escolha opção 2
```

**O que acontece:**
- Carrega modelo existente (40 épocas)
- **Continua** treinamento (não recomeça!)
- Adiciona +40 épocas
- Salva modelo (agora com 80 épocas totais)

---

### 4️⃣ Adicionar Mais Dados ao Dataset

```bash
# 1. Adicionar novos PDFs na pasta livros/
cp novos_livros/*.pdf livros/

# 2. Re-extrair texto
./extract_pdfs.sh

# 3. Continuar treinamento com dataset expandido
./train_incremental.sh 100
# Escolha opção 2
```

**O que acontece:**
- Dataset agora tem mais conteúdo
- Modelo aprende novos padrões
- Knowledge do modelo é expandido

---

## 📊 Comandos Rápidos

### Modo Manual (sem script)

```bash
# Carregar modelo e iniciar servidor (sem treinar)
./lmcs-llm

# Carregar modelo e treinar mais épocas
./lmcs-llm --train

# Carregar modelo e treinar mais épocas (atalho)
./lmcs-llm -t
```

### Monitorar Progresso

```bash
# Ver épocas treinadas
./lmcs-llm
# Output: "📊 Status: 120 épocas treinadas"

# Ver informações do modelo via API
curl http://localhost:8080/api/health | jq
```

---

## 🎯 Estratégias Recomendadas

### 📈 Estratégia Conservadora
```bash
./train_incremental.sh 40   # 40 épocas
./train_incremental.sh 40   # +40 = 80 total
./train_incremental.sh 40   # +40 = 120 total
# Testar após cada bloco
```

### 🚀 Estratégia Agressiva
```bash
./train_incremental.sh 200  # 200 épocas de uma vez
```

### 🔄 Estratégia Iterativa
```bash
# Ciclo: Treinar → Testar → Ajustar → Repetir
./train_incremental.sh 50   # Treinar
# Testar no navegador http://localhost:8080
# Adicionar mais dados se necessário
./train_incremental.sh 50   # Continuar
./train_incremental.sh      # Testar (opção 1)
```

---

## 📉 Como Saber Quando Parar?

### Sinais de Boa Convergência:
- ✅ Loss diminuindo consistentemente
- ✅ Texto gerado fica mais coerente
- ✅ Menos repetições estranhas

### Sinais de Overfitting:
- ⚠️ Loss para de diminuir (estabiliza)
- ⚠️ Texto gerado fica muito repetitivo
- ⚠️ Perde diversidade

### Ponto Ideal:
- Loss entre **1.8 - 2.2** para dataset de 36MB
- **200-500 épocas** normalmente é suficiente
- Teste visual é o melhor indicador!

---

## 🛠️ Resumo das Funcionalidades

| Funcionalidade | Comando | Descrição |
|---------------|---------|-----------|
| **Treinar do zero** | `./train_incremental.sh 100` | Cria novo modelo |
| **Testar modelo** | `./train_incremental.sh` → Opção 1 | Gera texto de teste |
| **Treinar mais** | `./train_incremental.sh 50` → Opção 2 | Adiciona épocas |
| **Resetar** | `./train_incremental.sh` → Opção 3 | Deleta e recomeça |
| **Servidor** | `./lmcs-llm` | Carrega modelo |
| **Treinar + Servidor** | `./lmcs-llm --train` | Treina depois serve |

---

## 💡 Dicas Pro

1. **Sempre teste após treinar** - Loss baixa ≠ texto bom
2. **Salve backups** - Copie `lmcs-model.bin` antes de treinar mais
3. **Ajuste temperatura** - 0.5-0.7 é bom para português
4. **Adicione dados gradualmente** - Melhor que tudo de uma vez
5. **Monitore a loss** - Deve diminuir a cada bloco de épocas

---

## 🔍 Exemplo Completo

```bash
# 1. Primeiro treinamento
./train_incremental.sh 50
# Resultado: Loss 2.65, 50 épocas

# 2. Testar
./train_incremental.sh
# Escolher 1, ver resultados no navegador

# 3. Texto ok, mas pode melhorar - treinar mais
./train_incremental.sh 100
# Resultado: Loss 2.35, 150 épocas totais

# 4. Adicionar mais livros
cp mais_livros/*.pdf livros/
./extract_pdfs.sh

# 5. Continuar com dataset maior
./train_incremental.sh 200
# Resultado: Loss 2.15, 350 épocas totais

# 6. Teste final
./train_incremental.sh
# Escolher 1, celebrar! 🎉
```

---

## ⚙️ Como Funciona Internamente

1. **Campo `EpochsTrained`** - Modelo guarda quantas épocas já fez
2. **Salvamento automático** - Após cada sessão, modelo é salvo
3. **Carregamento inteligente** - Detecta modelo existente
4. **Flag `--train`** - Ativa modo de treinamento adicional
5. **Acúmulo** - `EpochsTrained += novas_epocas`

**Exemplo:**
```
Modelo: 50 épocas treinadas
Comando: ./lmcs-llm --train (config: 40 épocas)
Resultado: 50 + 40 = 90 épocas totais
```

---

## 🎓 Conceitos

- **Treinamento Incremental**: Continuar de onde parou, não recomeçar
- **Fine-tuning**: Ajustar modelo existente com mais dados/épocas
- **Checkpoint**: Salvar estado do modelo para continuar depois
- **Convergência**: Quando a loss estabiliza (ponto ideal para parar)

---

**Pronto para treinar de forma inteligente! 🚀**
