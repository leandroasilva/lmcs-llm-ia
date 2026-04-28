// Estado da aplicação
let conversations = [];
let currentConversationId = null;
let isGenerating = false;

// Elementos DOM
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistory = document.getElementById('chatHistory');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperatureValue');
const maxLengthInput = document.getElementById('maxLength');
const statusIndicator = document.getElementById('statusIndicator');

// Inicialização
document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    checkHealth();
    setupEventListeners();
});

function setupEventListeners() {
    // Ajustar altura do textarea automaticamente
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    });

    // Atalho Enter para enviar
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('chatForm').dispatchEvent(new Event('submit'));
        }
    });

    // Slider de temperatura
    temperatureSlider.addEventListener('input', function() {
        temperatureValue.textContent = parseFloat(this.value).toFixed(1);
    });
}

// Verificar saúde do servidor
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            statusIndicator.textContent = 'Online';
            statusIndicator.classList.add('online');
        } else {
            throw new Error('Server error');
        }
    } catch (error) {
        statusIndicator.textContent = 'Offline';
        statusIndicator.classList.add('offline');
        console.error('Health check failed:', error);
    }
}

// Criar nova conversa
function newChat() {
    const conversation = {
        id: Date.now(),
        title: 'Nova Conversa',
        messages: [],
        createdAt: new Date()
    };
    
    conversations.unshift(conversation);
    currentConversationId = conversation.id;
    
    saveConversations();
    renderChatHistory();
    renderMessages();
}

// Usar sugestão
function useSuggestion(text) {
    messageInput.value = text;
    messageInput.focus();
}

// Enviar mensagem
async function sendMessage(event) {
    event.preventDefault();
    
    const text = messageInput.value.trim();
    if (!text || isGenerating) return;
    
    // Criar conversa se não existir
    if (!currentConversationId) {
        newChat();
    }
    
    const conversation = conversations.find(c => c.id === currentConversationId);
    if (!conversation) return;
    
    // Adicionar mensagem do usuário
    conversation.messages.push({
        role: 'user',
        content: text,
        timestamp: new Date()
    });
    
    // Atualizar título da conversa
    if (conversation.messages.length === 1) {
        conversation.title = text.substring(0, 30) + (text.length > 30 ? '...' : '');
    }
    
    renderMessages();
    saveConversations();
    renderChatHistory();
    
    // Limpar input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Gerar resposta da IA
    await generateAIResponse(conversation);
}

// Gerar resposta da IA com streaming
async function generateAIResponse(conversation) {
    isGenerating = true;
    sendBtn.disabled = true;
    
    // Adicionar mensagem de loading
    const loadingId = addLoadingMessage();
    
    try {
        const userMessage = conversation.messages[conversation.messages.length - 1].content;
        const temperature = parseFloat(temperatureSlider.value) || 0.7;
        const topK = 40;
        
        // Remover loading e adicionar mensagem vazia para streaming
        removeLoadingMessage(loadingId);
        const aiMessageId = addStreamingMessage();
        
        // Usar EventSource para SSE
        const response = await fetch('/api/ask/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: userMessage,
                temperature: temperature,
                top_k: topK
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Ler stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            // Decodificar chunk
            buffer += decoder.decode(value, { stream: true });
            
            // Processar eventos SSE
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Manter linha incompleta no buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        
                        if (data.done) {
                            // Streaming completado
                            conversation.messages.push({
                                role: 'ai',
                                content: fullResponse.trim(),
                                timestamp: new Date(),
                                elapsed_ms: data.elapsed_ms
                            });
                            updateStreamingMessage(aiMessageId, '', true);
                            saveConversations();
                        } else if (data.token) {
                            // Verificar se é erro
                            if (data.token.startsWith('[ERRO]')) {
                                throw new Error(data.token);
                            }
                            
                            // Adicionar token ao texto
                            fullResponse += data.token;
                            updateStreamingMessage(aiMessageId, fullResponse);
                        }
                    } catch (e) {
                        console.error('Erro ao parsear SSE:', e);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Error generating response:', error);
        addErrorMessage('Erro ao gerar resposta. Verifique se o servidor está rodando.');
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// Formatar prompt conversacional com histórico
function formatConversationPrompt(userMessage, conversationHistory) {
    let prompt = "";
    
    // Adicionar histórico (últimas 3 mensagens = 6 items)
    const recentMessages = conversationHistory.slice(-6);
    
    for (const msg of recentMessages) {
        if (msg.role === 'user') {
            prompt += `Usuário: ${msg.content}\n`;
        } else if (msg.role === 'ai') {
            prompt += `Assistente: ${msg.content}\n\n`;
        }
    }
    
    // Adicionar mensagem atual
    prompt += `Usuário: ${userMessage}\n`;
    prompt += `Assistente: `;
    
    return prompt;
}

// Extrair seed da mensagem (manter para compatibilidade)
function extractSeed(text) {
    // Pegar primeiras 2-3 palavras ou 10 caracteres como contexto
    const words = text.split(/\s+/).slice(0, 3).join(' ');
    if (words.length >= 5) {
        return words.substring(0, 10);
    }
    return 'o ';
}

// Adicionar mensagem de loading
function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ai';
    messageDiv.id = id;
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="loading">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return id;
}

// Adicionar mensagem para streaming
function addStreamingMessage() {
    const id = 'streaming-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ai';
    messageDiv.id = id;
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <span class="streaming-text"></span>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return id;
}

// Atualizar mensagem em streaming
function updateStreamingMessage(id, text, done = false) {
    const element = document.getElementById(id);
    if (element) {
        const contentDiv = element.querySelector('.message-content');
        const formattedContent = formatMessage(text);
        contentDiv.innerHTML = `<span class="${done ? '' : 'streaming-text'}">${formattedContent}</span>`;
        scrollToBottom();
    }
}

// Remover mensagem de loading
function removeLoadingMessage(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

// Adicionar mensagem de erro
function addErrorMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ai';
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <p style="color: #ef4444;">⚠️ ${message}</p>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Renderizar mensagens
function renderMessages() {
    const conversation = conversations.find(c => c.id === currentConversationId);
    
    if (!conversation || conversation.messages.length === 0) {
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
                    </svg>
                </div>
                <h2>Bem-vindo ao LMCS LLM!</h2>
                <p>Eu sou um modelo Transformer treinado para assistente conversacional. Como posso ajudar?</p>
                <div class="suggestions">
                    <button class="suggestion-btn" onclick="useSuggestion('Oi, tudo bem?')">
                        Oi, tudo bem?
                    </button>
                    <button class="suggestion-btn" onclick="useSuggestion('Quero contratar um plano')">
                        Quero contratar um plano
                    </button>
                    <button class="suggestion-btn" onclick="useSuggestion('Preciso de suporte técnico')">
                        Preciso de suporte técnico
                    </button>
                </div>
            </div>
        `;
        return;
    }
    
    chatMessages.innerHTML = '';
    
    conversation.messages.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.role}`;
        
        const avatar = msg.role === 'user' ? 'Você' : 'AI';
        const content = formatMessage(msg.content);
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">${content}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
    });
    
    scrollToBottom();
}

// Formatar mensagem
function formatMessage(content) {
    // Escape HTML
    content = content
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Quebras de linha
    content = content.replace(/\n/g, '<br>');
    
    return content;
}

// Renderizar histórico de conversas
function renderChatHistory() {
    chatHistory.innerHTML = '';
    
    conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = `chat-item ${conv.id === currentConversationId ? 'active' : ''}`;
        item.textContent = conv.title;
        item.onclick = () => switchConversation(conv.id);
        chatHistory.appendChild(item);
    });
}

// Trocar conversa
function switchConversation(id) {
    currentConversationId = id;
    renderMessages();
    renderChatHistory();
    saveConversations();
}

// Scroll para o final
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Salvar conversas no localStorage
function saveConversations() {
    try {
        localStorage.setItem('lmcs-conversations', JSON.stringify(conversations));
    } catch (error) {
        console.error('Error saving conversations:', error);
    }
}

// Carregar conversas do localStorage
function loadConversations() {
    try {
        const stored = localStorage.getItem('lmcs-conversations');
        if (stored) {
            conversations = JSON.parse(stored);
            if (conversations.length > 0) {
                currentConversationId = conversations[0].id;
            }
            renderChatHistory();
            renderMessages();
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
        conversations = [];
    }
}
