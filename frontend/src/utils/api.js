const API_BASE = import.meta.env.VITE_API_BASE || '/api';

function adminHeaders() {
  const passphrase = localStorage.getItem('admin_passphrase');
  return passphrase ? { 'X-Admin-Passphrase': passphrase } : {};
}

export async function api(path, options = {}) {
  const { body, method = 'GET', ...rest } = options;
  const isFormData = body instanceof FormData;
  const config = {
    method,
    headers: {
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...adminHeaders(),
      ...(rest.headers || {}),
    },
    ...rest,
  };
  if (body !== undefined) config.body = isFormData ? body : JSON.stringify(body);

  const res = await fetch(`${API_BASE}${path}`, config);
  if (res.status === 401 && !path.startsWith('/auth/')) {
    localStorage.removeItem('admin_passphrase');
    localStorage.removeItem('token');
    localStorage.removeItem('admin');
    window.location.href = '/login';
    throw new Error('Session expired');
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.status === 204 ? null : res.json();
}

export function getWsUrl() {
  const base = import.meta.env.VITE_API_BASE || `${window.location.origin}/api`;
  return base.replace(/^http/, 'ws') + '/ws/recognize';
}
