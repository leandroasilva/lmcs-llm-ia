import { useState, useRef, useEffect } from "react"
import {
  Send,
  Plus,
  Trash2,
  MessageSquare,
  Settings,
  Bot,
  User,
  X,
  ChevronRight,
  Zap,
  Loader2,
  Wifi,
  WifiOff,
} from "lucide-react"
import { useChat } from "@/hooks/useChat"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

function App() {
  const {
    conversations,
    currentConversation,
    currentConversationId,
    isGenerating,
    serverStatus,
    modelInfo,
    createConversation,
    deleteConversation,
    selectConversation,
    sendMessage,
  } = useChat()

  const [input, setInput] = useState("")
  const [temperature, setTemperature] = useState(0.7)
  const [topK, setTopK] = useState(30)
  const [showSettings, setShowSettings] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [currentConversation?.messages])

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!input.trim() || isGenerating) return
    const text = input.trim()
    setInput("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
    await sendMessage(text, temperature, topK)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleTextareaInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const target = e.currentTarget
    target.style.height = "auto"
    target.style.height = Math.min(target.scrollHeight, 200) + "px"
  }

  const suggestions = [
    "Oi, tudo bem?",
    "Quero contratar um plano",
    "Preciso de suporte técnico",
  ]

  return (
    <div className="flex h-screen w-full bg-background text-foreground overflow-hidden">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex flex-col border-r bg-card transition-all duration-300",
          sidebarOpen ? "w-72" : "w-0 overflow-hidden"
        )}
      >
        <div className="p-4 border-b">
          <Button
            onClick={createConversation}
            className="w-full gap-2"
            variant="default"
          >
            <Plus className="h-4 w-4" />
            Nova Conversa
          </Button>
        </div>

        <ScrollArea className="flex-1 p-2">
          {conversations.length === 0 && (
            <div className="text-center text-muted-foreground text-sm py-8">
              Nenhuma conversa ainda
            </div>
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => selectConversation(conv.id)}
              className={cn(
                "group flex items-center gap-3 rounded-lg px-3 py-2.5 cursor-pointer transition-colors",
                conv.id === currentConversationId
                  ? "bg-primary/10 text-primary"
                  : "hover:bg-accent text-foreground"
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              <span className="flex-1 truncate text-sm">{conv.title}</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 opacity-0 group-hover:opacity-100 shrink-0"
                onClick={(e) => {
                  e.stopPropagation()
                  deleteConversation(conv.id)
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </ScrollArea>

        <div className="p-4 border-t space-y-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {serverStatus === "online" ? (
              <Wifi className="h-3 w-3 text-green-500" />
            ) : (
              <WifiOff className="h-3 w-3 text-destructive" />
            )}
            <span>Servidor {serverStatus === "online" ? "online" : "offline"}</span>
          </div>
          {modelInfo && (
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex justify-between">
                <span>Modelo</span>
                <span className="font-medium text-foreground">{modelInfo.model}</span>
              </div>
              <div className="flex justify-between">
                <span>Vocab</span>
                <span className="font-medium text-foreground">{modelInfo.vocab?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Layers</span>
                <span className="font-medium text-foreground">{modelInfo.layers}</span>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b bg-card">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="shrink-0"
            >
              <ChevronRight
                className={cn("h-4 w-4 transition-transform", sidebarOpen && "rotate-180")}
              />
            </Button>
            <div className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-semibold">LMCS LLM</h1>
            </div>
            {serverStatus === "online" ? (
              <Badge variant="default" className="bg-green-500/20 text-green-600 hover:bg-green-500/20">
                Online
              </Badge>
            ) : (
              <Badge variant="destructive">Offline</Badge>
            )}
          </div>

          <Button
            variant="ghost"
            size="icon"
            onClick={() => setShowSettings(!showSettings)}
            className={cn(showSettings && "bg-accent")}
          >
            <Settings className="h-4 w-4" />
          </Button>
        </header>

        {/* Settings Panel */}
        {showSettings && (
          <Card className="mx-4 mt-4 shrink-0">
            <CardContent className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Configurações</h3>
                <Button variant="ghost" size="icon" onClick={() => setShowSettings(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Slider
                  label="Temperatura"
                  value={temperature}
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  onChange={setTemperature}
                />
                <Slider
                  label="Top-K"
                  value={topK}
                  min={1}
                  max={100}
                  step={1}
                  onChange={setTopK}
                />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Messages Area */}
        <ScrollArea className="flex-1 px-4 py-4">
          {!currentConversation || currentConversation.messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-6 py-12">
              <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10">
                <Zap className="h-8 w-8 text-primary" />
              </div>
              <div className="text-center space-y-2">
                <h2 className="text-2xl font-bold">Bem-vindo ao LMCS LLM!</h2>
                <p className="text-muted-foreground max-w-md">
                  Eu sou um modelo Transformer treinado para atendimento conversacional. Como posso ajudar?
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-lg">
                {suggestions.map((s) => (
                  <Button
                    key={s}
                    variant="outline"
                    className="justify-start text-left h-auto py-3 px-4"
                    onClick={() => {
                      setInput(s)
                      textareaRef.current?.focus()
                    }}
                  >
                    {s}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4 max-w-3xl mx-auto">
              {currentConversation.messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex gap-3",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {msg.role === "assistant" && (
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-1">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  )}
                  <div
                    className={cn(
                      "rounded-2xl px-4 py-2.5 max-w-[80%] text-sm leading-relaxed",
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-br-md"
                        : "bg-muted text-foreground rounded-bl-md"
                    )}
                  >
                    <div className="whitespace-pre-wrap">
                      {msg.content}
                      {msg.isStreaming && (
                        <span className="inline-block w-1.5 h-4 ml-0.5 bg-current animate-pulse align-middle" />
                      )}
                    </div>
                  </div>
                  {msg.role === "user" && (
                    <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center shrink-0 mt-1">
                      <User className="h-4 w-4 text-secondary-foreground" />
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>

        {/* Input Area */}
        <div className="p-4 border-t bg-card">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onInput={handleTextareaInput}
                placeholder="Digite sua mensagem..."
                className="min-h-[52px] max-h-[200px] resize-none py-3"
                disabled={isGenerating}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isGenerating}
                className="shrink-0 h-[52px] w-[52px]"
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2 text-center">
              Pressione Enter para enviar, Shift+Enter para nova linha
            </p>
          </form>
        </div>
      </main>
    </div>
  )
}

export default App
