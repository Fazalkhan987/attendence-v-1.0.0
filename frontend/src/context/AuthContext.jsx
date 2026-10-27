import { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../utils/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [admin, setAdmin] = useState(() => {
    const stored = localStorage.getItem('admin');
    return stored ? JSON.parse(stored) : null;
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const passphrase = localStorage.getItem('admin_passphrase');
    if (passphrase) {
      api('/auth/me')
        .then((data) => {
          setAdmin(data);
          localStorage.setItem('admin', JSON.stringify(data));
        })
        .catch(() => {
          localStorage.removeItem('admin_passphrase');
          localStorage.removeItem('token');
          localStorage.removeItem('admin');
          setAdmin(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  async function login(username, password) {
    const data = await api('/auth/login', {
      method: 'POST',
      body: { username, password },
    });
    localStorage.setItem('admin_passphrase', data.access_token);
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('admin', JSON.stringify(data.admin));
    setAdmin(data.admin);
    return data;
  }

  function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('admin_passphrase');
    localStorage.removeItem('admin');
    setAdmin(null);
  }

  return (
    <AuthContext.Provider value={{ admin, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
