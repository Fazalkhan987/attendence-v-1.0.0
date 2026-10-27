import { useState, useEffect } from 'react';
import { Users, UserCheck, UserX, ScanFace, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../utils/api';

function StatCard({ icon: Icon, label, value, color }) {
  const colors = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-emerald-50 text-emerald-600',
    red: 'bg-red-50 text-red-600',
    purple: 'bg-purple-50 text-purple-600',
  };
  return (
    <div className="card p-5">
      <div className="flex items-center gap-4">
        <div className={`w-11 h-11 rounded-xl flex items-center justify-center ${colors[color]}`}>
          <Icon size={20} />
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          <p className="text-xs text-gray-500 mt-0.5">{label}</p>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api('/persons'), api('/attendance?limit=100')])
      .then(([persons, attendance]) => {
        const today = new Date().toISOString().slice(0, 10);
        const todayRecords = attendance.filter((record) => record.created_at.startsWith(today));
        setStats({
          total_persons: persons.length,
          today_attendance: new Set(todayRecords.map((record) => record.person_id)).size,
          absent_today: Math.max(0, persons.length - new Set(todayRecords.map((record) => record.person_id)).size),
          registered_faces: persons.filter((person) => person.sample_count > 0).length,
          weekly_trend: [],
          recent_attendance: attendance.slice(0, 10).map((record) => ({
            ...record,
            full_name: record.person_id,
            date: new Date(record.created_at).toLocaleDateString(),
            time_in: new Date(record.created_at).toLocaleTimeString(),
            confidence: 100,
          })),
        });
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={28} className="animate-spin text-brand-600" />
      </div>
    );
  }

  if (!stats) return <p className="text-gray-500">Failed to load dashboard.</p>;

  const chartData = stats.weekly_trend.map((d) => ({
    date: d.date.slice(5), // MM-DD
    count: d.count,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-1">Attendance overview for today</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard icon={Users} label="Total registered" value={stats.total_persons} color="blue" />
        <StatCard icon={UserCheck} label="Present today" value={stats.today_attendance} color="green" />
        <StatCard icon={UserX} label="Absent today" value={stats.absent_today} color="red" />
        <StatCard icon={ScanFace} label="Faces registered" value={stats.registered_faces} color="purple" />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Chart */}
        <div className="card p-5 xl:col-span-2">
          <h2 className="text-sm font-semibold text-gray-900 mb-4">7-day trend</h2>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} barSize={28}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                  cursor={{ fill: 'rgba(52,120,252,0.06)' }}
                />
                <Bar dataKey="count" fill="#3478fc" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent attendance */}
        <div className="card xl:col-span-3 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-100">
            <h2 className="text-sm font-semibold text-gray-900">Recent attendance</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 text-left text-xs text-gray-500">
                  <th className="px-5 py-3 font-medium">Person</th>
                  <th className="px-5 py-3 font-medium">ID</th>
                  <th className="px-5 py-3 font-medium">Date</th>
                  <th className="px-5 py-3 font-medium">Time</th>
                  <th className="px-5 py-3 font-medium">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_attendance.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-5 py-8 text-center text-gray-400">No records yet</td>
                  </tr>
                ) : (
                  stats.recent_attendance.map((r) => (
                    <tr key={r.id} className="border-b border-gray-50 hover:bg-gray-50/50">
                      <td className="px-5 py-3 font-medium text-gray-900">{r.full_name}</td>
                      <td className="px-5 py-3 text-gray-500 font-mono text-xs">{r.person_id}</td>
                      <td className="px-5 py-3 text-gray-500">{r.date}</td>
                      <td className="px-5 py-3 text-gray-500">{r.time_in}</td>
                      <td className="px-5 py-3">
                        <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${
                          r.confidence > 70 ? 'bg-emerald-50 text-emerald-700' :
                          r.confidence > 50 ? 'bg-amber-50 text-amber-700' :
                          'bg-red-50 text-red-700'
                        }`}>
                          {r.confidence}%
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
