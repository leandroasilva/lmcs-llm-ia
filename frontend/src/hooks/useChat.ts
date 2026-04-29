import { useState, useCallback, useRef, useEffect } from "react"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
  isStreaming?: boolean
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: number
}

export interface ModelInfo {
  status: string
  model: string
  vocab: number
  d_model: number
  heads: number
  layers: number
  epochs: number
}

interface AskRequest {
  question: string
  temperature: number
  top_k: number
}

function generateId() {
  return Math.random().toString(36).substring(2, 9)
}

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try {
      const stored = localStorage.getItem("lmcs-conversations")
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  })
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [serverStatus, setServerStatus] = useState<"online" | "offline">("offline")
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    localStorage.setItem("lmcs-conversations", JSON.stringify(conversations))
  }, [conversations])

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/health", { signal: AbortSignal.timeout(5000) })
      if (res.ok) {
        const data = await res.json()
        setServerStatus("online")
        setModelInfo(data)
      } else {
        setServerStatus("offline")
      }
    } catch {
      setServerStatus("offline")
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [checkHealth])

  const createConversation = useCallback(() => {
    const conv: Conversation = {
      id: generateId(),
      title: "Nova Conversa",
      messages: [],
      createdAt: Date.now(),
    }
    setConversations((prev) => [conv, ...prev])
    setCurrentConversationId(conv.id)
    return conv.id
  }, [])

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id))
    setCurrentConversationId((prev) => (prev === id ? null : prev))
  }, [])

  const selectConversation = useCallback((id: string) => {
    setCurrentConversationId(id)
  }, [])

  const sendMessage = useCallback(
    async (content: string, temperature: number, topK: number) => {
      if (!content.trim() || isGenerating) return

      let convId = currentConversationId
      if (!convId) {
        convId = createConversation()
      }

      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: content.trim(),
        timestamp: Date.now(),
      }

      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [...c.messages, userMsg],
                title: c.messages.length === 0 ? content.trim().slice(0, 30) : c.title,
              }
            : c
        )
      )

      setIsGenerating(true)

      const assistantMsgId = generateId()
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [
                  ...c.messages,
                  { id: assistantMsgId, role: "assistant", content: "", timestamp: Date.now(), isStreaming: true },
                ],
              }
            : c
        )
      )

      try {
        const req: AskRequest = {
          question: content.trim(),
          temperature,
          top_k: topK,
        }

        const response = await fetch("/api/ask/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let fullContent = ""

        if (!reader) throw new Error("No response body")

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() || ""

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.substring(6))
                if (data.done) {
                  setConversations((prev) =>
                    prev.map((c) =>
                      c.id === convId
                        ? {
                            ...c,
                            messages: c.messages.map((m) =>
                              m.id === assistantMsgId
                                ? { ...m, content: fullContent.trim(), isStreaming: false }
                                : m
                            ),
                          }
                        : c
                    )
                  )
                } else if (data.token) {
                  fullContent += data.token
                  setConversations((prev) =>
                    prev.map((c) =>
                      c.id === convId
                        ? {
                            ...c,
                            messages: c.messages.map((m) =>
                              m.id === assistantMsgId ? { ...m, content: fullContent } : m
                            ),
                          }
                        : c
                    )
                  )
                }
              } catch {
                // ignore parse errors
              }
            }
          }
        }
      } catch (error) {
        setConversations((prev) =>
          prev.map((c) =>
            c.id === convId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantMsgId
                      ? { ...m, content: "Erro ao gerar resposta. Verifique se o servidor está rodando.", isStreaming: false }
                      : m
                  ),
                }
              : c
          )
        )
      } finally {
        setIsGenerating(false)
      }
    },
    [currentConversationId, isGenerating, createConversation]
  )

  const currentConversation = conversations.find((c) => c.id === currentConversationId) || null

  return {
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
    checkHealth,
  }
}
