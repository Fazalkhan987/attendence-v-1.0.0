import { useState, useEffect, useCallback } from 'react';
import { Search, Download, Trash2, Loader2, Calendar } from 'lucide-react';
import { api } from '../utils/api';
import { useToast } from '../components/Toast';
import Modal from '../components/Modal';

export default function AttendanceHistory() {
  const toast = useToast();
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  const fetchRecords = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (search) params.set('person_id', search);
      params.set('limit', '500');
      const data = await api(`/attendance?${params.toString()}`);
      setRecords(data.map((record) => ({
        ...record,
        date: new Date(record.created_at).toLocaleDateString(),
        time_in: new Date(record.created_at).toLocaleTimeString(),
        status: record.event_type,
        confidence: 100,
        full_name: record.person_id,
      })));
    } catch {} finally { setLoading(false); }
  }, [search, dateFrom, dateTo]);

  useEffect(() => { fetchRecords(); }, [fetchRecords]);

  async function handleExport() {
    try {
      const params = new URLSearchParams();
      if (dateFrom) params.set('date_from', dateFrom);
      if (dateTo) params.set('date_to', dateTo);
      const records = await api(`/attendance?limit=500`);
      const csv = ['person_id,event_type,note,created_at',
        ...records.map((r) => [r.person_id, r.event_type, r.note || '', r.created_at]
          .map((value) => `"${String(value).replaceAll('"', '""')}"`).join(','))].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'attendance.csv';
      a.click();
      URL.revokeObjectURL(url);
      toast.success('CSV exported');
    } catch (err) {
      toast.error('Export failed');
    }
  }

  async function handleDelete(id) {
    try {
      toast.error('Attendance deletion is not available in the FastAPI backend.');
    } catch (err) { toast.error(err.message); }
  }

  function setToday() {
    const today = new Date().toISOString().slice(0, 10);
    setDateFrom(today);
    setDateTo(today);
  }

  function setThisWeek() {
    const now = new Date();
    const day = now.getDay();
    const monday = new Date(now);
    monday.setDate(now.getDate() - (day === 0 ? 6 : day - 1));
    setDateFrom(monday.toISOString().slice(0, 10));
    setDateTo(now.toISOString().slice(0, 10));
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Attendance</h1>
          <p className="text-sm text-gray-500 mt-0.5">History and records</p>
        </div>
        <button onClick={handleExport} className="btn-secondary">
          <Download size={16} /> Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="card p-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by name or ID…"
              className="input-field pl-9"
            />
          </div>
          <div className="flex items-center gap-2">
            <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="input-field w-auto text-sm" />
            <span className="text-gray-400 text-sm">to</span>
            <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="input-field w-auto text-sm" />
          </div>
          <div className="flex gap-2">
            <button onClick={setToday} className="btn-secondary text-xs px-3 py-1.5">Today</button>
            <button onClick={setThisWeek} className="btn-secondary text-xs px-3 py-1.5">This week</button>
            <button
              onClick={() => { setSearch(''); setDateFrom(''); setDateTo(''); }}
              className="btn-secondary text-xs px-3 py-1.5"
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 text-left text-xs text-gray-500">
                <th className="px-5 py-3 font-medium">Person ID</th>
                <th className="px-5 py-3 font-medium">Name</th>
                <th className="px-5 py-3 font-medium">Department</th>
                <th className="px-5 py-3 font-medium">Date</th>
                <th className="px-5 py-3 font-medium">Time in</th>
                <th className="px-5 py-3 font-medium">Status</th>
                <th className="px-5 py-3 font-medium">Confidence</th>
                <th className="px-5 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={8} className="px-5 py-12 text-center">
                  <Loader2 size={24} className="animate-spin mx-auto text-brand-600" />
                </td></tr>
              ) : records.length === 0 ? (
                <tr><td colSpan={8} className="px-5 py-12 text-center text-gray-400">No records found</td></tr>
              ) : (
                records.map((r) => (
                  <tr key={r.id} className="border-b border-gray-50 hover:bg-gray-50/50">
                    <td className="px-5 py-3 font-mono text-xs text-gray-600">{r.person_id}</td>
                    <td className="px-5 py-3 font-medium text-gray-900">{r.full_name}</td>
                    <td className="px-5 py-3 text-gray-500">{r.department || '—'}</td>
                    <td className="px-5 py-3 text-gray-500">{r.date}</td>
                    <td className="px-5 py-3 text-gray-500">{r.time_in}</td>
                    <td className="px-5 py-3">
                      <span className="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700">
                        {r.status}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${
                        r.confidence > 70 ? 'bg-emerald-50 text-emerald-700' :
                        r.confidence > 50 ? 'bg-amber-50 text-amber-700' :
                        'bg-red-50 text-red-700'
                      }`}>
                        {r.confidence}%
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button
                        onClick={() => setDeleteConfirm(r)}
                        className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg"
                      >
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        {records.length > 0 && (
          <div className="px-5 py-3 border-t border-gray-100 text-xs text-gray-500">
            Showing {records.length} records
          </div>
        )}
      </div>

      {/* Delete confirmation */}
      <Modal open={!!deleteConfirm} onClose={() => setDeleteConfirm(null)} title="Confirm delete">
        {deleteConfirm && (
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              Delete attendance record for <strong>{deleteConfirm.full_name}</strong> on {deleteConfirm.date}?
            </p>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setDeleteConfirm(null)} className="btn-secondary">Cancel</button>
              <button onClick={() => handleDelete(deleteConfirm.id)} className="btn-danger">Delete</button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
