import { useState, type FormEvent } from 'react'
import { login } from './auth'

export default function LoginView({ onLoggedIn }: { onLoggedIn: (sub: string) => void }) {
  const [clientId, setClientId] = useState('')
  const [clientSecret, setClientSecret] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      const sub = await login(clientId.trim(), clientSecret)
      onLoggedIn(sub)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <div className="thread">
        <div className="bubble inv-start">
          <div className="waiting">Sign in to ARGUS.</div>
          <form onSubmit={submit}>
            <label>
              Client ID
              <input
                value={clientId}
                onChange={(e) => setClientId(e.target.value)}
                autoComplete="username"
                disabled={busy}
              />
            </label>
            <label>
              Client secret
              <input
                type="password"
                value={clientSecret}
                onChange={(e) => setClientSecret(e.target.value)}
                autoComplete="current-password"
                disabled={busy}
              />
            </label>
            {error && <div className="failure">{error}</div>}
            <div className="recover">
              <button type="submit" disabled={busy || !clientId.trim() || !clientSecret}>
                {busy ? 'Signing in…' : 'Sign in'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
