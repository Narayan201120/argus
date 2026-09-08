let memoryToken: string | null = null;

export function getToken(): string | null {
  if (memoryToken) return memoryToken;
  try {
    return sessionStorage.getItem('argus_token');
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  memoryToken = token;
  try {
    sessionStorage.setItem('argus_token', token);
  } catch {
    /* storage unavailable, memory copy still works */
  }
}

export function clearToken(): void {
  memoryToken = null;
  try {
    sessionStorage.removeItem('argus_token');
  } catch {
    /* nothing to clear */
  }
}

export async function authFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const token = getToken();
  if (!token) return fetch(input, init);
  const headers = new Headers(init.headers ?? {});
  if (!headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
  return fetch(input, { ...init, headers });
}

let memorySubject: string | null = null;

export function getSubject(): string | null {
  if (memorySubject) return memorySubject;
  try {
    return sessionStorage.getItem('argus_subject');
  } catch {
    return null;
  }
}

export function setSubject(subject: string): void {
  memorySubject = subject;
  try {
    sessionStorage.setItem('argus_subject', subject);
  } catch {
    /* storage unavailable, memory copy still works */
  }
}

export function clearSubject(): void {
  memorySubject = null;
  try {
    sessionStorage.removeItem('argus_subject');
  } catch {
    /* nothing to clear */
  }
}

export function logout(): void {
  clearToken();
  clearSubject();
}

export function isAuthError(err: unknown): boolean {
  return (err as { status?: number } | null | undefined)?.status === 401;
}

export async function fetchMe(): Promise<{ sub: string | null }> {
  const response = await authFetch('/v1/auth/me');
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* keep status-code detail */
    }
    const err = new Error(detail) as Error & { status?: number };
    err.status = response.status;
    throw err;
  }
  return response.json() as Promise<{ sub: string | null }>;
}

export async function login(clientId: string, clientSecret: string): Promise<string> {
  const response = await fetch('/v1/auth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ client_id: clientId, client_secret: clientSecret }),
  });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* keep status-code detail */
    }
    throw new Error(detail);
  }
  const body = (await response.json()) as { access_token: string };
  setToken(body.access_token);
  const me = await fetchMe();
  const sub = me.sub ?? clientId;
  setSubject(sub);
  return sub;
}
