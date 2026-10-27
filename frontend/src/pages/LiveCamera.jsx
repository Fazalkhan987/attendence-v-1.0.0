import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera, CameraOff, RefreshCw, MonitorUp, Loader2,
  UserCheck, AlertTriangle, Wifi, WifiOff,
} from 'lucide-react';
import { api, getWsUrl } from '../utils/api';
import { useToast } from '../components/Toast';

export default function LiveCamera() {
  const toast = useToast();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const socketRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const markedRef = useRef(new Set());  // track attendance marked this session
  const recognitionSessionRef = useRef(
    globalThis.crypto?.randomUUID?.() || `camera-${Date.now()}-${Math.random()}`
  );

  const [cameraOn, setCameraOn] = useState(false);
  const [stream, setStream] = useState(null);
  const [wsConnected, setWsConnected] = useState(true);
  const [recognitionError, setRecognitionError] = useState('');
  const [faces, setFaces] = useState([]);
  const [processedFrameUrl, setProcessedFrameUrl] = useState('');
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [cameraError, setCameraError] = useState('');
  const [log, setLog] = useState([]); // attendance log this session
  const [autoMark, setAutoMark] = useState(true);

  // Enumerate cameras
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then((devices) => {
      const videoDevices = devices.filter((d) => d.kind === 'videoinput');
      setCameras(videoDevices);
      if (videoDevices.length > 0 && !selectedCamera) {
        setSelectedCamera(videoDevices[0].deviceId);
      }
    });
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    setCameraError('');
    try {
      const constraints = {
        video: selectedCamera
          ? { deviceId: { exact: selectedCamera }, width: 640, height: 480 }
          : { width: 640, height: 480, facingMode: 'user' },
      };
      const s = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(s);
      setCameraOn(true);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play().catch(() => {});
      }
    } catch (err) {
      setCameraError(
        err.name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow camera access in the browser and try again.'
          : 'Could not access camera. Check that a camera is connected and try again.'
      );
    }
  }, [selectedCamera]);

  useEffect(() => {
    if (videoRef.current && stream && videoRef.current.srcObject !== stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }
  }, [stream]);

  // Stop camera
  const stopCamera = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    setCameraOn(false);
    setFaces([]);
    setRecognitionError('');
    setWsConnected(false);
    setProcessedFrameUrl('');
    requestInFlightRef.current = false;
    if (intervalRef.current) clearInterval(intervalRef.current);
  }, [stream]);

  // Send frames over WebSocket and display the backend-annotated JPEG response.
  useEffect(() => {
    if (!cameraOn) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }

    const socket = new WebSocket(
      `${getWsUrl()}?session_id=${encodeURIComponent(recognitionSessionRef.current)}&passphrase=${encodeURIComponent(localStorage.getItem('admin_passphrase') || '')}`
    );
    socket.binaryType = 'blob';
    socketRef.current = socket;
    socket.onopen = () => {
      setWsConnected(true);
      setRecognitionError('');
    };
    socket.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const payload = JSON.parse(event.data);
        const nextFaces = payload.faces || [];
        setFaces(nextFaces);
        if (autoMark) {
          nextFaces.forEach((face) => {
            if (face.recognized && face.person_id && !markedRef.current.has(face.person_id)) {
              markedRef.current.add(face.person_id);
              api('/attendance', {
                method: 'POST',
                body: { person_id: face.person_id, event_type: 'check-in' },
              }).then((event) => {
                setLog((prev) => [{ ...event, time: new Date().toLocaleTimeString() }, ...prev.slice(0, 49)]);
                toast.success(`${face.full_name || face.person_id} marked present`);
              }).catch((error) => {
                markedRef.current.delete(face.person_id);
                toast.error(error.message);
              });
            }
          });
        }
        return;
      }
      const url = URL.createObjectURL(event.data);
      setProcessedFrameUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return url;
      });
      requestInFlightRef.current = false;
    };
    socket.onerror = () => {
      setWsConnected(false);
      setRecognitionError('Backend video stream is unavailable.');
      requestInFlightRef.current = false;
    };
    socket.onclose = (event) => {
      if (socketRef.current === socket) {
        setWsConnected(false);
        if (event.reason) setRecognitionError(event.reason);
      }
    };

    const recognizeFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2
        || socket.readyState !== WebSocket.OPEN || requestInFlightRef.current
        || socket.bufferedAmount > 2_000_000) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      requestInFlightRef.current = true;
      canvas.toBlob((blob) => {
        if (!blob) {
          requestInFlightRef.current = false;
          return;
        }
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(blob);
        } else {
          requestInFlightRef.current = false;
        }
      }, 'image/jpeg', 0.6);
    };

    intervalRef.current = setInterval(recognizeFrame, 100);

    return () => {
      clearInterval(intervalRef.current);
      if (socketRef.current === socket) {
        socket.close();
        socketRef.current = null;
      }
      requestInFlightRef.current = false;
    };
  }, [cameraOn, autoMark, toast]);

  useEffect(() => () => {
    if (processedFrameUrl) URL.revokeObjectURL(processedFrameUrl);
  }, [processedFrameUrl]);

  function reloadFaces() {
    markedRef.current.clear();
    toast.info('Recognition gallery is refreshed after each enrollment.');
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Live camera</h1>
          <p className="text-sm text-gray-500 mt-0.5">Real-time face recognition and attendance</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full ${
            wsConnected ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'
          }`}>
            {wsConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
            {wsConnected ? 'Recognition ready' : 'Recognition unavailable'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Camera view */}
        <div className="xl:col-span-2 space-y-4">
          <div className="card overflow-hidden">
            <div className="relative bg-gray-900 aspect-video flex items-center justify-center">
              {cameraOn ? (
                <>
                  <video ref={videoRef} autoPlay playsInline muted className="hidden" />
                  {processedFrameUrl ? (
                    <img
                      src={processedFrameUrl}
                      alt="Backend processed camera stream"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-sm text-gray-400">Connecting to recognition stream...</div>
                  )}
                </>
              ) : (
                <div className="text-center text-gray-400 p-8">
                  <Camera size={48} className="mx-auto mb-3 opacity-40" />
                  <p className="text-sm">Camera is off. Click start to begin recognition.</p>
                  {cameraError && <p className="text-red-400 text-xs mt-2">{cameraError}</p>}
                </div>
              )}
              {cameraOn && recognitionError && (
                <div className="absolute bottom-3 left-3 right-3 rounded-lg bg-red-900/80 px-3 py-2 text-xs text-white">
                  Recognition unavailable: {recognitionError}
                </div>
              )}
              <canvas ref={canvasRef} className="hidden" />
            </div>

            {/* Controls */}
            <div className="p-4 border-t border-gray-100 flex flex-wrap items-center gap-3">
              {cameraOn ? (
                <button onClick={stopCamera} className="btn-danger text-sm px-3 py-1.5">
                  <CameraOff size={15} /> Stop
                </button>
              ) : (
                <button onClick={startCamera} className="btn-primary text-sm px-3 py-1.5">
                  <Camera size={15} /> Start camera
                </button>
              )}

              <button onClick={reloadFaces} className="btn-secondary text-sm px-3 py-1.5">
                <RefreshCw size={15} /> Reload faces
              </button>

              {cameras.length > 1 && (
                <select
                  value={selectedCamera}
                  onChange={(e) => setSelectedCamera(e.target.value)}
                  className="input-field w-auto text-sm py-1.5"
                >
                  {cameras.map((c, i) => (
                    <option key={c.deviceId} value={c.deviceId}>
                      {c.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
              )}

              <label className="flex items-center gap-2 text-sm text-gray-600 ml-auto cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={autoMark}
                  onChange={(e) => setAutoMark(e.target.checked)}
                  className="w-4 h-4 text-brand-600 rounded border-gray-300 focus:ring-brand-500"
                />
                Auto-mark attendance
              </label>
            </div>
          </div>

          {/* Detected faces strip */}
          {faces.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {faces.map((f, i) => (
                <div
                  key={i}
                  className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border ${
                    f.recognized
                      ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                      : 'bg-gray-100 border-gray-200 text-gray-600'
                  }`}
                >
                  {f.recognized ? <UserCheck size={15} /> : <AlertTriangle size={15} />}
                  {f.full_name}
                  {f.recognized && (
                    <span className="text-xs opacity-70">{Math.round(f.confidence * 100)}%</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Attendance log sidebar */}
        <div className="card flex flex-col max-h-[600px]">
          <div className="px-5 py-4 border-b border-gray-100">
            <h2 className="text-sm font-semibold text-gray-900">Session log</h2>
            <p className="text-xs text-gray-500 mt-0.5">{log.length} recorded this session</p>
          </div>
          <div className="flex-1 overflow-y-auto">
            {log.length === 0 ? (
              <div className="p-8 text-center text-gray-400 text-sm">No attendance recorded yet</div>
            ) : (
              <div className="divide-y divide-gray-50">
                {log.map((entry, i) => (
                  <div key={i} className="px-5 py-3">
                    <div className="flex items-center gap-2">
                      <UserCheck size={14} className="text-emerald-500 shrink-0" />
                      <span className="text-sm font-medium text-gray-900 truncate">
                        {entry.message?.replace('Attendance marked for ', '')}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 mt-0.5 pl-[22px]">{entry.time}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
