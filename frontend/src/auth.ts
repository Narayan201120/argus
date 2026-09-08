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
