import { useEffect, useRef, useState } from 'react'
import Card from '../components/Card'
import DataDictionary from '../components/DataDictionary'
import DecisionNotes from '../components/DecisionNotes'
import { api } from '../lib/api'

const SUGGESTIONS = [
  'How many applicants took cash loans versus revolving loans?',
  "What's the average income by education type?",
  'What is the default rate for applicants with a refused prior application versus those without?',
  'How many applicants have more than 2 active bureau credits?',
]

function getSessionId() {
  let id = sessionStorage.getItem('chat_session_id')
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem('chat_session_id', id)
  }
  return id
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const sessionId = useRef(getSessionId())
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send(text) {
    const question = text.trim()
    if (!question || loading) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: question }])
    setLoading(true)
    try {
      const res = await api.chat(sessionId.current, question)
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: res.answer, sql: res.sql, rewritten: res.rewritten_question },
      ])
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', text: `Something went wrong: ${e.message}`, error: true }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-(--color-ink)">Ask the Data</h1>
        <p className="mt-1.5 text-sm text-(--color-ink-muted)">
          Ask a plain-English question, it becomes a validated SQL query behind the scenes.
          Follow-up questions keep context from earlier in this conversation.
        </p>
      </header>

      <DataDictionary />

      <Card className="flex flex-col gap-4">
        <div className="flex max-h-125 min-h-40 flex-col gap-4 overflow-y-auto">
          {messages.length === 0 && (
            <div className="flex flex-col gap-2">
              <div className="text-xs font-medium text-(--color-ink-faint)">Try asking</div>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="w-fit rounded-full border border-(--color-border) px-3.5 py-1.5 text-left text-xs text-(--color-ink-muted) hover:border-(--color-ink-faint) hover:text-(--color-ink)"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                  m.role === 'user'
                    ? 'bg-(--color-ink) text-(--color-surface)'
                    : m.error
                      ? 'bg-(--color-high-bg) text-(--color-high)'
                      : 'bg-(--color-canvas) text-(--color-ink)'
                }`}
              >
                {m.text}
                {m.sql && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs text-(--color-ink-faint)">
                      View generated SQL
                    </summary>
                    <pre className="mt-1.5 overflow-x-auto rounded-lg bg-(--color-surface) p-2.5 text-xs text-(--color-ink-muted)">
                      {m.sql}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          ))}
          {loading && <div className="text-xs text-(--color-ink-faint)">Thinking...</div>}
          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            send(input)
          }}
          className="flex gap-2 border-t border-(--color-border) pt-4"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the applicants..."
            className="flex-1 rounded-lg border border-(--color-border) bg-(--color-canvas) px-3.5 py-2.5 text-sm outline-none focus:border-(--color-ink-faint)"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-(--color-ink) px-4 py-2.5 text-sm font-medium text-(--color-surface) disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </Card>

      <DecisionNotes
        points={[
          {
            title: 'Why a graph, not one prompt',
            detail:
              'A 6-step pipeline (rewrite question, generate SQL, validate, execute, summarize, with a bounded retry loop) rather than a single LLM call. Only 3 tables are used, small enough to fit in one prompt entirely, so no schema-retrieval step is needed.',
          },
          {
            title: 'Why generated SQL is shown, not hidden',
            detail:
              'Independent, code-level validation runs before any query executes: single statement, SELECT-only, a keyword blocklist, and a check that any category value the model filters on actually exists in the data. Showing the SQL makes that verifiable, not just claimed.',
          },
          {
            title: 'What happens when it cannot answer',
            detail:
              'If a question falls outside the 3-table schema, the model is instructed to say so directly rather than guess, and if a query still fails after one self-correction retry, the answer says that honestly too.',
          },
        ]}
      />
    </div>
  )
}
