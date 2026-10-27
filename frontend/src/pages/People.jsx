import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Search, Plus, Edit2, Trash2, Camera, Loader2,
  UserCheck, UserX, ChevronDown, X,
} from 'lucide-react';
import { api } from '../utils/api';
import { useToast } from '../components/Toast';
import Modal from '../components/Modal';

// ───── Face Capture component ─────
function FaceCapture({ personId, onDone }) {
  const SAMPLE_COUNT = 8;
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [captured, setCaptured] = useState([]);
  const [loading, setLoading] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [autoCapture, setAutoCapture] = useState(true);
  const toast = useToast();

  const startCamera = useCallback(async () => {
    try {
      setCameraError('');
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      setStream(s);
      if (videoRef.current) videoRef.current.srcObject = s;
    } catch {
      setCameraError('Camera access denied or not available.');
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []); // eslint-disable-line

  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
  }, [stream]);

  function capture() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || captured.length >= SAMPLE_COUNT) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.9);
  }

  const register = useCallback(async (samples = captured) => {
    if (samples.length !== SAMPLE_COUNT || loading) return;
    setLoading(true);
    try {
      const formData = new FormData();
      samples.forEach((dataUrl, index) => {
        const [header, encoded] = dataUrl.split(',');
        const binary = atob(encoded);
        const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
        formData.append('images', new Blob([bytes], { type: header.match(/:(.*?);/)[1] }), `sample-${index + 1}.jpg`);
      });
      await api(`/persons/${personId}/face`, {
        method: 'POST',
        body: formData,
      });
      toast.success(`${SAMPLE_COUNT} face samples saved successfully`);
      stream?.getTracks().forEach((t) => t.stop());
      onDone?.();
    } catch (err) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  }, [SAMPLE_COUNT, captured, loading, onDone, personId, stream, toast]);

  function captureNext() {
    const sample = capture();
    if (!sample) return;
    setCaptured((previous) => {
      const next = [...previous, sample];
      if (next.length === SAMPLE_COUNT) register(next);
      return next;
    });
  }

  useEffect(() => {
    if (!autoCapture || loading || cameraError || captured.length >= SAMPLE_COUNT) return undefined;
    const timer = window.setInterval(captureNext, 1100);
    return () => window.clearInterval(timer);
  }, [autoCapture, cameraError, captured.length, loading, SAMPLE_COUNT]);

  return (
    <div className="space-y-4">
      {cameraError ? (
        <div className="text-center py-8 text-red-600 text-sm">{cameraError}</div>
      ) : (
        <div className="space-y-3">
          <div className="relative">
            <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-lg bg-gray-900" />
            {captured.length > 0 && (
              <span className="absolute top-2 right-2 rounded-full bg-emerald-600 px-3 py-1 text-xs font-medium text-white">
                {captured.length} / {SAMPLE_COUNT} captured
              </span>
            )}
          </div>
          <canvas ref={canvasRef} className="hidden" />
          <p className="text-center text-sm text-gray-600">
            {loading ? 'Saving samples…' : autoCapture
              ? 'The camera is automatically capturing samples. Move slightly between captures.'
              : `Capture ${SAMPLE_COUNT} samples. Change your angle or expression between captures.`}
          </p>
          <div className="flex gap-2">
            <button onClick={() => { setCaptured([]); setAutoCapture(true); }} disabled={loading} className="btn-secondary flex-1">
              Start over
            </button>
            <button onClick={() => setAutoCapture((value) => !value)} disabled={loading || captured.length >= SAMPLE_COUNT} className="btn-secondary flex-1">
              {autoCapture ? 'Pause auto capture' : 'Resume auto capture'}
            </button>
            {!autoCapture && (
              <button onClick={captureNext} disabled={loading || captured.length >= SAMPLE_COUNT} className="btn-primary flex-1">
                <Camera size={16} /> Capture
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ───── Add / Edit form ─────
function PersonForm({ initial, onSubmit, onCancel, loading }) {
  const [form, setForm] = useState(
    initial || { person_id: '', full_name: '', department: '', contact_email: '', contact_phone: '', person_type: 'student' }
  );
  const set = (k, v) => setForm((p) => ({ ...p, [k]: v }));

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); onSubmit(form); }}
      className="space-y-4"
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Person ID</label>
          <input
            value={form.person_id}
            onChange={(e) => set('person_id', e.target.value)}
            className="input-field"
            placeholder="STU-001"
            disabled={!!initial}
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Full name</label>
          <input value={form.full_name} onChange={(e) => set('full_name', e.target.value)} className="input-field" required />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Department / Class</label>
          <input value={form.department} onChange={(e) => set('department', e.target.value)} className="input-field" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
          <select value={form.person_type} onChange={(e) => set('person_type', e.target.value)} className="input-field">
            <option value="student">Student</option>
            <option value="employee">Employee</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
          <input type="email" value={form.contact_email} onChange={(e) => set('contact_email', e.target.value)} className="input-field" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
          <input value={form.contact_phone} onChange={(e) => set('contact_phone', e.target.value)} className="input-field" />
        </div>
      </div>
      <div className="flex gap-2 justify-end pt-2">
        <button type="button" onClick={onCancel} className="btn-secondary">Cancel</button>
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? <Loader2 size={16} className="animate-spin" /> : null}
          {initial ? 'Update' : 'Next'}
        </button>
      </div>
    </form>
  );
}

// ───── Main People page ─────
export default function People() {
  const toast = useToast();
  const [persons, setPersons] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [formModal, setFormModal] = useState({ open: false, person: null });
  const [faceModal, setFaceModal] = useState({ open: false, personId: null, name: '' });
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [saving, setSaving] = useState(false);

  const fetchPersons = useCallback(async () => {
    try {
      const data = await api('/persons');
      setPersons(data);
    } catch {} finally { setLoading(false); }
  }, [search, typeFilter]);

  useEffect(() => { fetchPersons(); }, [fetchPersons]);

  async function handleCreate(form) {
    setSaving(true);
    try {
      const metadata = {
        contact_email: form.contact_email || null,
        contact_phone: form.contact_phone || null,
        person_type: form.person_type,
      };
      const body = new FormData();
      body.append('person_id', form.person_id);
      body.append('full_name', form.full_name);
      if (form.department) body.append('department', form.department);
      body.append('metadata', JSON.stringify(metadata));
      await api('/persons', {
        method: 'POST',
        body,
      });
      toast.success('Details saved. Camera enrollment started.');
      setFormModal({ open: false, person: null });
      setFaceModal({ open: true, personId: form.person_id, name: form.full_name });
      fetchPersons();
    } catch (err) { toast.error(err.message); }
    finally { setSaving(false); }
  }

  async function handleUpdate(form) {
    setSaving(true);
    try {
      const { person_id, ...rest } = form;
      const params = new URLSearchParams();
      Object.entries(rest).forEach(([key, value]) => {
        if (key === 'metadata') return;
        if (value) params.set(key, value);
      });
      params.set('metadata', JSON.stringify({
        contact_email: rest.contact_email || null,
        contact_phone: rest.contact_phone || null,
        person_type: rest.person_type || 'student',
      }));
      await api(`/persons/${person_id}?${params.toString()}`, { method: 'PATCH' });
      toast.success('Person updated');
      setFormModal({ open: false, person: null });
      fetchPersons();
    } catch (err) { toast.error(err.message); }
    finally { setSaving(false); }
  }

  async function handleDelete(personId) {
    try {
      await api(`/persons/${personId}`, { method: 'DELETE' });
      toast.success('Person deleted');
      setDeleteConfirm(null);
      fetchPersons();
    } catch (err) { toast.error(err.message); }
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-900">People</h1>
          <p className="text-sm text-gray-500 mt-0.5">Manage students and employees</p>
        </div>
        <button onClick={() => setFormModal({ open: true, person: null })} className="btn-primary">
          <Plus size={16} /> Add person
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name, ID, or department…"
            className="input-field pl-9"
          />
        </div>
        <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} className="input-field w-auto">
          <option value="">All types</option>
          <option value="student">Students</option>
          <option value="employee">Employees</option>
        </select>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 text-left text-xs text-gray-500">
                <th className="px-5 py-3 font-medium">ID</th>
                <th className="px-5 py-3 font-medium">Name</th>
                <th className="px-5 py-3 font-medium">Department</th>
                <th className="px-5 py-3 font-medium">Type</th>
                <th className="px-5 py-3 font-medium">Face</th>
                <th className="px-5 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={6} className="px-5 py-12 text-center">
                  <Loader2 size={24} className="animate-spin mx-auto text-brand-600" />
                </td></tr>
              ) : persons.length === 0 ? (
                <tr><td colSpan={6} className="px-5 py-12 text-center text-gray-400">No people found</td></tr>
              ) : (
                persons.map((p) => (
                  <tr key={p.person_id} className="border-b border-gray-50 hover:bg-gray-50/50">
                    <td className="px-5 py-3 font-mono text-xs text-gray-600">{p.person_id}</td>
                    <td className="px-5 py-3 font-medium text-gray-900">{p.full_name}</td>
                    <td className="px-5 py-3 text-gray-500">{p.department || '—'}</td>
                    <td className="px-5 py-3">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                        p.metadata?.person_type === 'employee' ? 'bg-blue-50 text-blue-700' : 'bg-purple-50 text-purple-700'
                      }`}>{p.metadata?.person_type || 'student'}</span>
                    </td>
                    <td className="px-5 py-3">
                      {p.sample_count > 0 ? (
                        <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700">
                          <UserCheck size={14} /> Registered
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-xs text-gray-400">
                          <UserX size={14} /> Not registered
                        </span>
                      )}
                    </td>
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-1 justify-end">
                        <button
                          onClick={() => setFaceModal({ open: true, personId: p.person_id, name: p.full_name })}
                          className="p-1.5 text-gray-400 hover:text-brand-600 hover:bg-brand-50 rounded-lg"
                          title="Register face"
                        >
                          <Camera size={15} />
                        </button>
                        <button
                          onClick={() => setFormModal({ open: true, person: p })}
                          className="p-1.5 text-gray-400 hover:text-amber-600 hover:bg-amber-50 rounded-lg"
                          title="Edit"
                        >
                          <Edit2 size={15} />
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(p)}
                          className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg"
                          title="Delete"
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add / Edit modal */}
      <Modal
        open={formModal.open}
        onClose={() => setFormModal({ open: false, person: null })}
        title={formModal.person ? 'Edit person' : 'Add new person'}
        wide
      >
        <PersonForm
          initial={formModal.person}
          onSubmit={formModal.person ? handleUpdate : handleCreate}
          onCancel={() => setFormModal({ open: false, person: null })}
          loading={saving}
        />
      </Modal>

      {/* Face registration modal */}
      <Modal
        open={faceModal.open}
        onClose={() => setFaceModal({ open: false, personId: null, name: '' })}
        title={`Register face — ${faceModal.name}`}
      >
        {faceModal.open && (
          <FaceCapture
            personId={faceModal.personId}
            onDone={() => { setFaceModal({ open: false, personId: null, name: '' }); fetchPersons(); }}
          />
        )}
      </Modal>

      {/* Delete confirmation */}
      <Modal
        open={!!deleteConfirm}
        onClose={() => setDeleteConfirm(null)}
        title="Confirm delete"
      >
        {deleteConfirm && (
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              Delete <strong>{deleteConfirm.full_name}</strong> ({deleteConfirm.person_id})? This also removes their attendance records and cannot be undone.
            </p>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setDeleteConfirm(null)} className="btn-secondary">Cancel</button>
              <button onClick={() => handleDelete(deleteConfirm.person_id)} className="btn-danger">Delete</button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
