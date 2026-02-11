import './App.css'

import { useEffect, useMemo, useState } from 'react'

type Audience = 'technical' | 'nontechnical'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  audience?: Audience
}

type SearchResult = {
  idx: number
  chunk: string
  score: number
  source?: string
  doc_id?: string
  title?: string
}

type Metrics = {
  hit_at_k: number
  mrr: number
  top_k: number
}

type MalwareScanResponse = {
  filename: string
  size_bytes: number
  verdict: 'suspicious' | 'likely_clean'
  score: number
  reasons: string[]
}

const API_BASE =
  (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000'

function App() {
  const [activeTab, setActiveTab] = useState<'assistant' | 'file-scan' | 'metrics' | 'explorer'>(
    'assistant',
  )
  const [audience, setAudience] = useState<Audience>('nontechnical')

  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [lastContext, setLastContext] = useState<SearchResult[]>([])

  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [metricsLoading, setMetricsLoading] = useState(false)

  const [explorerQuery, setExplorerQuery] = useState('')
  const [explorerResults, setExplorerResults] = useState<SearchResult[]>([])
  const [explorerLoading, setExplorerLoading] = useState(false)

  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [fileScanResult, setFileScanResult] = useState<MalwareScanResponse | null>(null)
  const [fileScanLoading, setFileScanLoading] = useState(false)

  const accentGradient = useMemo(
    () =>
      'linear-gradient(135deg, #00f5a0 0%, #00d9f5 50%, #6c5ce7 100%)',
    [],
  )

  const fetchMetrics = async () => {
    setMetricsLoading(true)
    try {
      const res = await fetch(`${API_BASE}/metrics/retrieval`)
      if (!res.ok) throw new Error('Failed to load metrics')
      const data = (await res.json()) as Metrics
      setMetrics(data)
    } catch (e) {
      console.error(e)
    } finally {
      setMetricsLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'metrics' && !metrics && !metricsLoading) {
      fetchMetrics()
    }
  }, [activeTab, metrics, metricsLoading])

  const sendMessage = async () => {
    const question = input.trim()
    if (!question || isSending) return

    const newUserMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
      audience,
    }

    setMessages((prev) => [...prev, newUserMessage])
    setInput('')
    setIsSending(true)

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          audience,
          top_k: 5,
        }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail?.detail || 'Assistant error')
      }
      const data = await res.json()

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.answer ?? 'No answer returned.',
        audience,
      }

      setMessages((prev) => [...prev, assistantMessage])
      setLastContext((data.used_chunks || []) as SearchResult[])
    } catch (e: any) {
      console.error(e)
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content:
          'The assistant backend is not reachable or returned an error. Check that the FastAPI server is running and your API key is configured.',
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsSending(false)
    }
  }

  const runExplorerSearch = async () => {
    const q = explorerQuery.trim()
    if (!q) return
    setExplorerLoading(true)
    setExplorerResults([])
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 10 }),
      })
      if (!res.ok) throw new Error('Search failed')
      const data = (await res.json()) as SearchResult[]
      setExplorerResults(data)
    } catch (e) {
      console.error(e)
    } finally {
      setExplorerLoading(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null
    setUploadedFile(file)
    setFileScanResult(null)
  }

  const runFileScan = async () => {
    if (!uploadedFile || fileScanLoading) return
    setFileScanLoading(true)
    setFileScanResult(null)
    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const res = await fetch(`${API_BASE}/scan-file`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        throw new Error('File scan failed')
      }
      const data = (await res.json()) as MalwareScanResponse
      setFileScanResult(data)
    } catch (e) {
      console.error(e)
      setFileScanResult({
        filename: uploadedFile?.name ?? 'unknown',
        size_bytes: uploadedFile?.size ?? 0,
        verdict: 'likely_clean',
        score: 0,
        reasons: [
          'File scan failed. Check that the backend /scan-file endpoint is running and reachable.',
        ],
      })
    } finally {
      setFileScanLoading(false)
    }
  }

  return (
    <div className="app-root">
      <div className="app-bg-glow" />

      <header className="app-header">
        <div className="app-title-block">
          <h1>Cyber Threat Intel Copilot</h1>
          <p>
            RAG over CVEs &amp; MITRE ATT&amp;CK with a single interface for
            analysts and non‑technical users.
          </p>
        </div>
        <div className="app-header-accent" style={{ backgroundImage: accentGradient }}>
          <span>ONLINE</span>
          <span className="dot" />
        </div>
      </header>

      <main className="app-main">
        <aside className="app-sidebar">
          <nav className="app-tabs">
            <button
              className={activeTab === 'assistant' ? 'tab active' : 'tab'}
              onClick={() => setActiveTab('assistant')}
            >
              Assistant
            </button>
            <button
              className={activeTab === 'file-scan' ? 'tab active' : 'tab'}
              onClick={() => setActiveTab('file-scan')}
            >
              File scan
            </button>
            <button
              className={activeTab === 'metrics' ? 'tab active' : 'tab'}
              onClick={() => setActiveTab('metrics')}
            >
              Attack Metrics
            </button>
            <button
              className={activeTab === 'explorer' ? 'tab active' : 'tab'}
              onClick={() => setActiveTab('explorer')}
            >
              Index Explorer
            </button>
          </nav>

          <section className="mode-toggle">
            <h3>Audience mode</h3>
            <div className="mode-toggle-buttons">
              <button
                className={audience === 'nontechnical' ? 'pill active' : 'pill'}
                onClick={() => setAudience('nontechnical')}
              >
                Non‑technical
              </button>
              <button
                className={audience === 'technical' ? 'pill active' : 'pill'}
                onClick={() => setAudience('technical')}
              >
                Security analyst
              </button>
            </div>
            <p className="mode-hint">
              Same RAG engine, different language. Flip this and ask the same
              question to see the difference.
            </p>
          </section>

          <section className="context-panel">
            <h3>Latest retrieved context</h3>
            {!lastContext.length && (
              <p className="muted">
                Ask the assistant to see which CVEs / techniques it grounded on.
              </p>
            )}
            <div className="context-scroll">
              {lastContext.map((c) => (
                <article key={c.idx} className="context-card">
                  <div className="context-meta">
                    {c.doc_id && <span className="badge badge-id">{c.doc_id}</span>}
                    {c.source && (
                      <span className="badge badge-source">{c.source}</span>
                    )}
                  </div>
                  {c.title && <h4>{c.title}</h4>}
                  <p>{c.chunk.slice(0, 260)}…</p>
                  <div className="context-score">
                    Similarity score: {c.score.toFixed(3)}
                  </div>
                </article>
              ))}
            </div>
          </section>
        </aside>

        <section className="app-content">
          {activeTab === 'assistant' && (
            <section className="panel">
              <h2>Assistant console</h2>
              <p className="panel-subtitle">
                Describe what happened, paste logs, or ask “what is this CVE?”
                — the copilot answers from your Kaggle‑built index.
              </p>

              <section className="file-scan-inline file-scan-at-top">
                <h3>File scan (tech &amp; non‑tech)</h3>
                <p className="panel-subtitle">Upload a file to run a heuristic malware check.</p>
                <div className="file-scan-controls">
                  <input
                    type="file"
                    id="assistant-file-input"
                    onChange={handleFileChange}
                    className="file-scan-input-hidden"
                  />
                  <label htmlFor="assistant-file-input" className="file-choose-btn">
                    Choose file
                  </label>
                  <button
                    type="button"
                    className="primary-button file-scan-btn"
                    onClick={runFileScan}
                    disabled={!uploadedFile || fileScanLoading}
                  >
                    {fileScanLoading ? 'Scanning…' : 'Scan file'}
                  </button>
                </div>
                {uploadedFile && !fileScanResult && (
                  <p className="muted file-scan-selected">
                    Selected: {uploadedFile.name} ({uploadedFile.size} bytes)
                  </p>
                )}
                {fileScanResult && (
                  <div
                    className={`file-scan-result ${fileScanResult.verdict === 'suspicious' ? 'verdict-suspicious' : 'verdict-clean'}`}
                  >
                    <div className="file-scan-summary">
                      <span className="badge badge-score">
                        Verdict: {fileScanResult.verdict} ({(fileScanResult.score * 100).toFixed(0)}% suspicious)
                      </span>
                    </div>
                    <p className="muted">
                      File: {fileScanResult.filename} — {fileScanResult.size_bytes} bytes
                    </p>
                    <ul>
                      {fileScanResult.reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>

              <div className="chat-window">
                {messages.length === 0 && (
                  <div className="chat-empty">
                    <p>
                      Try questions like:
                    </p>
                    <ul>
                      <li>
                        “I clicked a suspicious link and now my PC won’t boot.
                        What attacks could this be?”
                      </li>
                      <li>
                        “We see phishing emails against finance. Which ATT&amp;CK
                        techniques should we watch for?”
                      </li>
                    </ul>
                  </div>
                )}

                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={
                      m.role === 'user'
                        ? 'chat-message user'
                        : 'chat-message assistant'
                    }
                  >
                    <div className="chat-bubble">
                      <div className="chat-meta">
                        <span>{m.role === 'user' ? 'You' : 'Assistant'}</span>
                        {m.audience && (
                          <span className="chat-audience">
                            {m.audience === 'technical'
                              ? 'Security analyst view'
                              : 'Non‑technical view'}
                          </span>
                        )}
                      </div>
                      <p>{m.content}</p>
                    </div>
                  </div>
                ))}

                {isSending && (
                  <div className="chat-message assistant">
                    <div className="chat-bubble pending">
                      <p>Thinking over your threat intel index…</p>
                    </div>
                  </div>
                )}
              </div>

              <div className="chat-input-row">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question about incidents, CVEs, or ATT&CK techniques…"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      sendMessage()
                    }
                  }}
                />
                <button
                  className="primary-button"
                  disabled={isSending || !input.trim()}
                  onClick={sendMessage}
                >
                  {isSending ? 'Sending…' : 'Ask'}
                </button>
              </div>
            </section>
          )}

          {activeTab === 'file-scan' && (
            <section className="panel">
              <h2>File scan</h2>
              <p className="panel-subtitle">
                Upload a file to run a rule-based malware heuristic check (POST /scan-file).
                For testing only — not a replacement for a real antivirus.
              </p>
              <div className="file-scan-controls">
                <input
                  type="file"
                  id="tab-file-input"
                  onChange={handleFileChange}
                  className="file-scan-input-hidden"
                />
                <label htmlFor="tab-file-input" className="file-choose-btn">
                  Choose file
                </label>
                <button
                  type="button"
                  className="primary-button file-scan-btn"
                  onClick={runFileScan}
                  disabled={!uploadedFile || fileScanLoading}
                >
                  {fileScanLoading ? 'Scanning…' : 'Scan file'}
                </button>
              </div>
              {uploadedFile && !fileScanResult && (
                <p className="muted file-scan-selected">
                  Selected: {uploadedFile.name} ({uploadedFile.size} bytes)
                </p>
              )}
              {fileScanResult && (
                <div
                  className={`file-scan-result ${fileScanResult.verdict === 'suspicious' ? 'verdict-suspicious' : 'verdict-clean'}`}
                >
                  <div className="file-scan-summary">
                    <span className="badge badge-score">
                      Verdict: {fileScanResult.verdict} ({(fileScanResult.score * 100).toFixed(0)}% suspicious)
                    </span>
                  </div>
                  <p className="muted">
                    File: {fileScanResult.filename} — {fileScanResult.size_bytes} bytes
                  </p>
                  <ul>
                    {fileScanResult.reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}

          {activeTab === 'metrics' && (
            <section className="panel">
              <h2>Retrieval quality metrics</h2>
              <p className="panel-subtitle">
                Hit@k and MRR over a small ATT&amp;CK‑style query set, mirroring
                the notebook’s eval section.
              </p>

              <div className="metrics-grid">
                <div className="metric-card">
                  <h3>Hit@{metrics?.top_k ?? 5}</h3>
                  <p className="metric-value">
                    {metricsLoading || !metrics
                      ? '…'
                      : `${(metrics.hit_at_k * 100).toFixed(1)}%`}
                  </p>
                  <p className="metric-caption">
                    Fraction of queries where at least one relevant CVE ID
                    appears in the top‑k retrieved chunks.
                  </p>
                </div>

                <div className="metric-card">
                  <h3>MRR</h3>
                  <p className="metric-value">
                    {metricsLoading || !metrics
                      ? '…'
                      : metrics.mrr.toFixed(3)}
                  </p>
                  <p className="metric-caption">
                    Mean reciprocal rank of the first relevant document —
                    higher is better.
                  </p>
                </div>
              </div>

              <button
                className="secondary-button"
                onClick={fetchMetrics}
                disabled={metricsLoading}
              >
                {metricsLoading ? 'Recomputing…' : 'Re-run metrics'}
              </button>

              <p className="metrics-note">
                You can extend the evaluation set in the backend to track more
                attack paths, campaigns, or internal playbooks over time.
              </p>
            </section>
          )}

          {activeTab === 'explorer' && (
            <section className="panel">
              <h2>Threat index explorer</h2>
              <p className="panel-subtitle">
                Search directly over the FAISS index to see which chunks and
                sources are available before you hit the LLM.
              </p>

              <div className="explorer-input-row">
                <input
                  type="text"
                  placeholder="Search for CVE IDs, keywords, or technique names…"
                  value={explorerQuery}
                  onChange={(e) => setExplorerQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      runExplorerSearch()
                    }
                  }}
                />
                <button
                  className="secondary-button"
                  onClick={runExplorerSearch}
                  disabled={explorerLoading || !explorerQuery.trim()}
                >
                  {explorerLoading ? 'Searching…' : 'Search'}
                </button>
              </div>

              <div className="explorer-results">
                {explorerResults.map((r) => (
                  <article key={r.idx} className="context-card">
                    <div className="context-meta">
                      {r.doc_id && <span className="badge badge-id">{r.doc_id}</span>}
                      {r.source && (
                        <span className="badge badge-source">{r.source}</span>
                      )}
                      <span className="badge badge-score">
                        {r.score.toFixed(3)}
                      </span>
                    </div>
                    {r.title && <h4>{r.title}</h4>}
                    <p>{r.chunk}</p>
                  </article>
                ))}

                {!explorerLoading && !explorerResults.length && (
                  <p className="muted">
                    No results yet. Try “ransomware”, “remote code execution”, or
                    a specific CVE ID.
                  </p>
                )}
              </div>
            </section>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
