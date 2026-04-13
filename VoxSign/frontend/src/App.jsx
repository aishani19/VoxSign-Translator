import { useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const PREDICT_INTERVAL_MS = 250;

function App() {
  const videoRef = useRef(null);
  const intervalRef = useRef(null);
  const inFlightRef = useRef(false);
  const isPredictingRef = useRef(false);
  const [sessionId, setSessionId] = useState("");
  const [status, setStatus] = useState("Starting...");
  const [sentence, setSentence] = useState("");
  const [grammar, setGrammar] = useState("");
  const [previewSrc, setPreviewSrc] = useState("");
  const [modelLabels, setModelLabels] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [stablePrediction, setStablePrediction] = useState("");
  const [stableConfidence, setStableConfidence] = useState(0);
  const [threshold, setThreshold] = useState(0);
  const [cvMetrics, setCvMetrics] = useState({
    left_hand_detected: false,
    right_hand_detected: false,
    landmarks_count: 0
  });
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    const init = async () => {
      try {
        const response = await fetch(`${API_BASE}/session`, { method: "POST" });
        const payload = await response.json();
        setSessionId(payload.session_id);
        setStatus("Session created.");
        const metaResp = await fetch(`${API_BASE}/meta`);
        if (metaResp.ok) {
          const meta = await metaResp.json();
          setModelLabels(meta.actions || []);
        }
      } catch (error) {
        setStatus(`API unavailable: ${error}`);
      }
    };
    init();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setStatus("Camera started.");
    } catch (error) {
      setStatus(`Camera error: ${error.message}`);
    }
  };

  const captureAndPredict = async () => {
    if (inFlightRef.current) return;
    if (!sessionId) {
      setStatus("Session not ready.");
      return;
    }
    if (!videoRef.current) {
      setStatus("Video element missing.");
      return;
    }
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) {
      setStatus("No camera frame yet. Start camera first.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) {
        setStatus("Could not capture frame.");
        return;
      }
      inFlightRef.current = true;
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("file", blob, "frame.jpg");

      try {
        const response = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          body: formData
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text);
        }
        const payload = await response.json();
        setSentence(payload.sentence || "");
        setGrammar(payload.grammar_result || "");
        setPreviewSrc(`data:image/jpeg;base64,${payload.preview_b64}`);
        setCurrentPrediction(payload.current_prediction || "");
        setConfidence(payload.confidence || 0);
        setStablePrediction(payload.stable_prediction || "");
        setStableConfidence(payload.stable_confidence || 0);
        setThreshold(payload.threshold || 0);
        setCvMetrics(
          payload.cv || {
            left_hand_detected: false,
            right_hand_detected: false,
            landmarks_count: 0
          }
        );
        setStatus("Predicting...");
      } catch (error) {
        setStatus(`Predict failed: ${error.message}`);
      } finally {
        inFlightRef.current = false;
      }
    }, "image/jpeg");
  };

  const stopPredictLoop = () => {
    isPredictingRef.current = false;
    setIsPredicting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setStatus("Prediction paused.");
  };

  const startPredictLoop = () => {
    if (isPredictingRef.current) return;
    isPredictingRef.current = true;
    setIsPredicting(true);
    setStatus("Live prediction running...");
    intervalRef.current = setInterval(() => {
      captureAndPredict();
    }, PREDICT_INTERVAL_MS);
  };

  const applyGrammar = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/apply-grammar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      const payload = await response.json();
      setGrammar(payload.grammar_result || "");
      setStatus("Grammar applied.");
    } catch (error) {
      setStatus(`Grammar failed: ${error.message}`);
    }
  };

  const resetAll = async () => {
    if (!sessionId) return;
    try {
      await fetch(`${API_BASE}/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      setSentence("");
      setGrammar("");
      setPreviewSrc("");
      setCurrentPrediction("");
      setConfidence(0);
      setStablePrediction("");
      setStableConfidence(0);
      setStatus("State reset.");
    } catch (error) {
      setStatus(`Reset failed: ${error.message}`);
    }
  };

  useEffect(() => {
    return () => {
      stopPredictLoop();
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="page">
      <h1>VoxSign Translator (React)</h1>
      <p className="status">{status}</p>
      <p className="status">Model labels: {modelLabels.length ? modelLabels.join(", ") : "loading..."}</p>

      <div className="panels">
        <div className="panel">
          <h3>Live Camera</h3>
          <video ref={videoRef} autoPlay playsInline muted className="video" />
        </div>
        <div className="panel">
          <h3>Prediction Preview</h3>
          {previewSrc ? (
            <img src={previewSrc} alt="Prediction preview" className="preview" />
          ) : (
            <div className="placeholder">No prediction yet</div>
          )}
        </div>
      </div>

      <div className="controls">
        <button onClick={startCamera}>Start Camera</button>
        <button onClick={startPredictLoop} disabled={isPredicting}>
          Start Live Predict
        </button>
        <button onClick={stopPredictLoop} disabled={!isPredicting}>
          Stop Live Predict
        </button>
        <button onClick={captureAndPredict}>Predict Once</button>
        <button onClick={applyGrammar}>Apply Grammar</button>
        <button onClick={resetAll}>Reset</button>
      </div>

      <label>Current Prediction</label>
      <textarea
        value={
          currentPrediction
            ? `${currentPrediction} (confidence ${confidence.toFixed(2)} / threshold ${threshold.toFixed(2)})`
            : "No stable prediction yet"
        }
        readOnly
      />

      <label>Stable Prediction (Voting)</label>
      <textarea
        value={
          stablePrediction
            ? `${stablePrediction} (stable confidence ${stableConfidence.toFixed(2)})`
            : "Waiting for stable sign..."
        }
        readOnly
      />

      <label>Computer Vision Status</label>
      <textarea
        value={`Left hand: ${cvMetrics.left_hand_detected ? "Detected" : "Not detected"} | Right hand: ${
          cvMetrics.right_hand_detected ? "Detected" : "Not detected"
        } | Landmarks: ${cvMetrics.landmarks_count}`}
        readOnly
      />

      <label>Predicted Sentence</label>
      <textarea value={sentence} readOnly />

      <label>Grammar Output</label>
      <textarea value={grammar} readOnly />
    </div>
  );
}

export default App;
