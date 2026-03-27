import React, { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [status, setStatus] = useState("idle");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [messages, setMessages] = useState([]);
  const [robotState, setRobotState] = useState("idle");

  const [settings, setSettings] = useState(null);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);

  // Poll backend status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const statusRes = await fetch(`${API_BASE}/api/session/current`);
        if (statusRes.ok) {
          const statusData = await statusRes.json();
          setStatus(statusData.is_running ? "running" : "idle");
        }
      } catch {
        // ignore polling errors
      }
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 1500);
    return () => clearInterval(id);
  }, []);

  // WebSocket
  useEffect(() => {
    let ws = new WebSocket(`ws://127.0.0.1:8000/ws/conversation`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "turn_complete") {
         setMessages(prev => [...prev, {
            ts: data.timestamp,
            userText: data.transcript,
            agentText: data.response,
             risk: data.risk_rating,
             latency: data.latency?.total_ms
         }]);
      } else if (data.type === "state_change") {
         setRobotState(data.state);
      }
    };
    return () => ws.close();
  }, []);

  // Load settings once at startup
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/settings`);
        if (!res.ok) return;
        const data = await res.json();
        const initial = {
            polar_level: data.polar_level || 0,
            category: data.category || "D",
            subtype: data.subtype || 1,
            modifiers: data.modifiers || [],
            participant_id: "user-123",
            tts_voice: data.tts_voice || "onyx"
        };
        setSettings(initial);
        setSettingsDirty(false);
      } catch {
         setSettings({ polar_level: 0, category: "D", subtype: 1, modifiers: [], participant_id: "test", tts_voice: "onyx" });
      }
    };
    loadSettings();
  }, []);

  const callApi = async (path, body, expectedNextStatus) => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {})
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setError(data.error || `Error ${res.status}`); return; }
      if (expectedNextStatus) setStatus(expectedNextStatus);
    } catch (e) { setError(e.message); } 
      finally { setBusy(false); }
  };

  const handleStart = () => callApi("/api/session/start", {
      participant_id: settings.participant_id,
      polar_level: settings.polar_level,
      category: settings.category,
      subtype: settings.subtype,
      modifiers: settings.modifiers
  }, "running");

  const handleEnd = () => callApi("/api/session/stop", {}, "idle");

  const onSettingsChange = (field, value) => {
    if (!settings) return;
    setSettings(prev => ({ ...prev, [field]: value }));
    setSettingsDirty(true);
  };

  const handleSaveSettings = async () => {
    if (!settings) return;
    setSettingsSaving(true);
    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            polar_level: settings.polar_level,
            category: settings.category,
            subtype: settings.subtype,
            modifiers: settings.modifiers,
            tts_voice: settings.tts_voice
        }),
      });
      setSettingsDirty(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setSettingsSaving(false);
    }
  };

  const isRunning = status === "running";

  return (
    <div style={styles.app}>
      {/* LEFT: controls + settings */}
      <div style={styles.leftPanel}>
        <h1 style={styles.title}>AVCT Control Panel</h1>

        {/* Session status + main buttons */}
        <div style={styles.section}>
          <div style={styles.statusBox}>
            <span style={styles.statusLabel}>System Status:</span>
            <span style={{ ...styles.statusValue, ...statusColorStyle(status) }}>
              {status.toUpperCase()}
            </span>
          </div>
          <div style={styles.buttonRow}>
            <button style={styles.button} onClick={handleStart} disabled={busy || isRunning}>Start Session</button>
            <button style={styles.button} onClick={handleEnd} disabled={busy || !isRunning}>End Session</button>
          </div>
          {error && <p style={styles.error}>Error: {error}</p>}
        </div>

        {/* Settings */}
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>AVCT Matrix</h2>
          {settings && (
            <>
              {/* Polar Slider */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Polar Intensity</h3>
                <label style={styles.label}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Level:</span><strong>{settings.polar_level > 0 ? `+${settings.polar_level}` : settings.polar_level}</strong>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
                    <span style={{ fontSize: "11px", color: "#888" }}>-3 (Support)</span>
                    <input type="range" min="-3" max="3" step="1" style={{ flex: 1 }}
                      value={settings.polar_level}
                      onChange={(e) => onSettingsChange("polar_level", Number(e.target.value))}
                    />
                    <span style={{ fontSize: "11px", color: "#888" }}>+3 (Hostile)</span>
                  </div>
                </label>
              </div>

              {/* Category */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Category</h3>
                <div style={{display: 'flex', gap: '5px', flexWrap: 'wrap'}}>
                    {['B', 'C', 'D', 'E', 'F', 'G'].map(cat => (
                        <button key={cat} 
                          style={{...styles.catButton, backgroundColor: settings.category === cat ? '#2196f3' : '#e0e0e0', color: settings.category === cat ? 'white' : 'black'}}
                          onClick={() => onSettingsChange("category", cat)}
                        >
                            {cat}
                        </button>
                    ))}
                </div>
              </div>

              {/* Subtype */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Subtype</h3>
                <div style={{display: 'flex', gap: '5px'}}>
                    {[1, 2, 3].map(sub => (
                        <button key={sub} 
                          style={{...styles.catButton, backgroundColor: settings.subtype === sub ? '#4caf50' : '#e0e0e0', color: settings.subtype === sub ? 'white' : 'black'}}
                          onClick={() => onSettingsChange("subtype", sub)}
                        >
                            {settings.category}{sub}
                        </button>
                    ))}
                </div>
              </div>

              {/* Modifiers */}
              <div style={styles.subsection}>
                <h3 style={styles.subTitle}>Modifiers (M1-M6)</h3>
                <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px'}}>
                    {['M1', 'M2', 'M3', 'M4', 'M5', 'M6'].map(mod => {
                        const active = settings.modifiers.includes(mod);
                        return (
                            <label key={mod} style={{fontSize: '13px', display: 'flex', alignItems: 'center', gap: '4px'}}>
                                <input type="checkbox" checked={active} onChange={(e) => {
                                    const next = e.target.checked 
                                      ? [...settings.modifiers, mod] 
                                      : settings.modifiers.filter(m => m !== mod);
                                    onSettingsChange("modifiers", next);
                                }}/> {mod}
                            </label>
                        );
                    })}
                </div>
              </div>

              <div style={styles.saveRow}>
                <button style={styles.saveButton} onClick={handleSaveSettings} disabled={!settingsDirty || settingsSaving}>
                  {settingsSaving ? "Applying..." : "Apply Live Settings"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* RIGHT: timer + robot state + chat view */}
      <div style={styles.rightPanel}>
        <div style={styles.rightTopBar}>
          <div style={styles.timerBox}>
            <span style={styles.timerLabel}>Robot State:</span>
            <span style={{ ...styles.timerValue, ...robotStateColorStyle(robotState) }}>{robotState.toUpperCase()}</span>
          </div>
        </div>

        <h2 style={styles.chatTitle}>Live Turn Preview Monitor</h2>
        <div style={styles.chatBox}>
          {messages.length === 0 && <div style={styles.emptyChat}>Waiting for conversation...</div>}
          
          {messages.map((msg, idx) => (
            <div key={idx} style={styles.turnBlock}>
                <div style={styles.userMessage}>
                    <strong>You:</strong> {msg.userText}
                </div>
                <div style={{...styles.robotMessage, borderLeft: `6px solid ${getRiskColor(msg.risk)}`}}>
                    <div style={styles.messageHeader}>
                        <strong>Robot:</strong> 
                        <span style={styles.riskBadge(msg.risk)}>Risk: {msg.risk || "Green"}</span>
                    </div>
                    <div>{msg.agentText}</div>
                    <div style={styles.msgTime}>Latency: {msg.latency}ms | Timestamp: {msg.ts}</div>
                </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function getRiskColor(risk) {
    if(risk === "Red") return "#f44336";
    if(risk === "Amber") return "#ffeb3b";
    return "#4caf50";
}

function statusColorStyle(status) {
  if (status === "running") return { color: "green" };
  return { color: "gray" };
}

function robotStateColorStyle(robotState) {
  switch (robotState) {
    case "listening": return { color: "#1976d2" };
    case "speaking": return { color: "#2e7d32" };
    case "processing": return { color: "#f57c00" };
    default: return { color: "#555" };
  }
}

const styles = {
  app: { display: "flex", height: "100vh", fontFamily: "Inter, Arial, sans-serif" },
  leftPanel: { width: "360px", padding: "20px", borderRight: "1px solid #ddd", overflowY: "auto", backgroundColor: "#fcfcfc" },
  rightPanel: { flex: 1, padding: "20px", display: "flex", flexDirection: "column", backgroundColor: "#f4f6f8" },
  rightTopBar: { display: "flex", justifyContent: "space-between", marginBottom: "10px" },
  title: { marginTop: 0, marginBottom: "16px", fontSize: "22px", color: "#333" },
  section: { marginBottom: "20px" },
  sectionTitle: { marginTop: 0, marginBottom: "10px", fontSize: "16px", color: "#444" },
  subsection: { marginBottom: "12px", padding: "12px", borderRadius: "6px", backgroundColor: "#fff", border: "1px solid #eee", boxShadow: "0 1px 3px rgba(0,0,0,0.05)" },
  subTitle: { marginTop: "0", marginBottom: "8px", fontSize: "14px", color: "#666" },
  statusBox: { display: "flex", alignItems: "center", marginBottom: "10px" },
  statusLabel: { fontWeight: "bold", marginRight: "8px", fontSize: "14px" },
  statusValue: { fontWeight: "bold", fontSize: "14px" },
  buttonRow: { display: "flex", gap: "10px" },
  button: { flex: 1, padding: "10px", fontSize: "14px", cursor: "pointer", borderRadius: "4px", border: "1px solid #ccc", backgroundColor: "#fff" },
  catButton: { flex: 1, padding: "8px 0", cursor: "pointer", border: "none", borderRadius: "4px", fontSize: "14px", fontWeight: "bold" },
  error: { color: "red", fontSize: "13px" },
  chatTitle: { marginTop: "0", marginBottom: "10px", fontSize: "18px", color: "#333" },
  chatBox: { flex: 1, border: "1px solid #ddd", borderRadius: "6px", padding: "15px", overflowY: "auto", backgroundColor: "#fff", boxShadow: "inset 0 1px 4px rgba(0,0,0,0.05)" },
  emptyChat: { color: "#888", fontStyle: "italic", textAlign: "center", marginTop: "20px" },
  turnBlock: { marginBottom: "15px" },
  userMessage: { marginBottom: "4px", padding: "8px 12px", borderRadius: "6px", backgroundColor: "#f0f0f0", color: "#333", fontSize: "14px", width: "fit-content", maxWidth: "80%" },
  robotMessage: { padding: "10px 12px", borderRadius: "6px", backgroundColor: "#e3f2fd", color: "#0d47a1", fontSize: "14px", width: "fit-content", maxWidth: "80%", marginLeft: "auto" },
  messageHeader: { display: "flex", justifyContent: "space-between", marginBottom: "4px", alignItems: "center", gap: "15px" },
  riskBadge: (risk) => ({ fontSize: "11px", fontWeight: "bold", color: "#fff", backgroundColor: getRiskColor(risk), padding: "2px 6px", borderRadius: "10px" }),
  msgTime: { fontSize: "11px", color: "#888", marginTop: "6px" },
  label: { display: "block", marginBottom: "6px", fontSize: "13px" },
  saveRow: { marginTop: "12px" },
  saveButton: { width: "100%", padding: "10px", fontSize: "14px", cursor: "pointer", backgroundColor: "#2196f3", color: "white", border: "none", borderRadius: "4px", fontWeight: "bold" },
  timerBox: { fontSize: "14px", backgroundColor: "#fff", padding: "6px 12px", borderRadius: "20px", border: "1px solid #ddd", boxShadow: "0 1px 2px rgba(0,0,0,0.05)" },
  timerLabel: { marginRight: "6px", color: "#666" }
};
export default App;
