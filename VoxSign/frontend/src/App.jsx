import { useEffect, useRef, useState, useCallback } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const PREDICT_INTERVAL_MS = 250;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const inFlightRef = useRef(false);
  const isPredictingRef = useRef(false);
  const handLandmarkerRef = useRef(null);
  const [sessionId, setSessionId] = useState("");
  const [status, setStatus] = useState("Starting...");
  const [sentence, setSentence] = useState("");
  const [grammar, setGrammar] = useState("");
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
  const [mpReady, setMpReady] = useState(false);

  // Initialize session + MediaPipe HandLandmarker
  useEffect(() => {
    const init = async () => {
      try {
        const response = await fetch(`${API_BASE}/session`, { method: "POST" });
        const payload = await response.json();
        setSessionId(payload.session_id);
        setStatus("Session created. Loading hand detection model...");
        const metaResp = await fetch(`${API_BASE}/meta`);
        if (metaResp.ok) {
          const meta = await metaResp.json();
          setModelLabels(meta.actions || []);
        }
      } catch (error) {
        setStatus(`API unavailable: ${error}`);
      }

      // Load MediaPipe HandLandmarker
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        handLandmarkerRef.current = handLandmarker;
        setMpReady(true);
        setStatus("Hand detection ready. Start camera to begin.");
      } catch (error) {
        setStatus(`MediaPipe load error: ${error.message}`);
      }
    };
    init();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setStatus("Camera started.");
    } catch (error) {
      setStatus(`Camera error: ${error.message}`);
    }
  };

  // Extract 126-dim landmark vector matching Python keypoint_extraction format
  const extractLandmarks = useCallback((results) => {
    const lh = new Float32Array(63);
    const rh = new Float32Array(63);
    let landmarksCount = 0;
    let leftDetected = false;
    let rightDetected = false;

    if (results.landmarks && results.landmarks.length > 0) {
      for (let i = 0; i < results.landmarks.length; i++) {
        const hand = results.landmarks[i];
        const handedness = results.handednesses[i]?.[0]?.categoryName || "Right";
        landmarksCount += hand.length;

        if (handedness === "Left") {
          leftDetected = true;
          for (let j = 0; j < 21; j++) {
            // Left hand: flip x (1.0 - x) to match training data
            lh[j * 3] = 1.0 - hand[j].x;
            lh[j * 3 + 1] = hand[j].y;
            lh[j * 3 + 2] = hand[j].z;
          }
        } else {
          rightDetected = true;
          for (let j = 0; j < 21; j++) {
            // Right hand: keep x as-is
            rh[j * 3] = hand[j].x;
            rh[j * 3 + 1] = hand[j].y;
            rh[j * 3 + 2] = hand[j].z;
          }
        }
      }
    }

    return {
      landmarks: [...lh, ...rh],
      landmarksCount,
      leftDetected,
      rightDetected
    };
  }, []);

  // Draw hand landmarks on canvas overlay
  const drawLandmarks = useCallback((results) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results.landmarks) return;

    const connections = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17]
    ];

    for (let i = 0; i < results.landmarks.length; i++) {
      const hand = results.landmarks[i];
      const handedness = results.handednesses[i]?.[0]?.categoryName || "Right";
      const color = handedness === "Left" ? "#00ff00" : "#ff4444";

      // Draw connections
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (const [a, b] of connections) {
        ctx.beginPath();
        ctx.moveTo(hand[a].x * canvas.width, hand[a].y * canvas.height);
        ctx.lineTo(hand[b].x * canvas.width, hand[b].y * canvas.height);
        ctx.stroke();
      }

      // Draw points
      ctx.fillStyle = color;
      for (const lm of hand) {
        ctx.beginPath();
        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw label
      const minX = Math.min(...hand.map(l => l.x)) * canvas.width;
      const minY = Math.min(...hand.map(l => l.y)) * canvas.height;
      ctx.fillStyle = color;
      ctx.font = "14px Arial";
      ctx.fillText(`${handedness} hand`, minX, Math.max(16, minY - 8));
    }
  }, []);

  const captureAndPredict = useCallback(async () => {
    if (inFlightRef.current) return;
    if (!sessionId || !handLandmarkerRef.current) return;
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    // Run MediaPipe hand detection in browser
    const mpResults = handLandmarkerRef.current.detectForVideo(video, performance.now());
    drawLandmarks(mpResults);

    const { landmarks, landmarksCount, leftDetected, rightDetected } = extractLandmarks(mpResults);

    setCvMetrics({
      left_hand_detected: leftDetected,
      right_hand_detected: rightDetected,
      landmarks_count: landmarksCount
    });

    // Send landmarks to backend
    inFlightRef.current = true;
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          landmarks: landmarks,
          landmarks_count: landmarksCount
        })
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text);
      }
      const payload = await response.json();
      setSentence(payload.sentence || "");
      setGrammar(payload.grammar_result || "");
      setCurrentPrediction(payload.current_prediction || "");
      setConfidence(payload.confidence || 0);
      setStablePrediction(payload.stable_prediction || "");
      setStableConfidence(payload.stable_confidence || 0);
      setThreshold(payload.threshold || 0);
      setStatus("Predicting...");
    } catch (error) {
      setStatus(`Predict failed: ${error.message}`);
    } finally {
      inFlightRef.current = false;
    }
  }, [sessionId, drawLandmarks, extractLandmarks]);

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
      <h1>VoxSign Translator</h1>
      <p className="status">{status}</p>
      <p className="status">
        Model labels: {modelLabels.length ? `${modelLabels.length} signs loaded` : "loading..."}
        {!mpReady && " | Hand detection loading..."}
      </p>

      <div className="video-container">
        <video ref={videoRef} autoPlay playsInline muted className="video" />
        <canvas ref={canvasRef} className="overlay-canvas" />
      </div>

      <div className="controls">
        <button onClick={startCamera}>Start Camera</button>
        <button onClick={startPredictLoop} disabled={isPredicting || !mpReady}>
          Start Live Predict
        </button>
        <button onClick={stopPredictLoop} disabled={!isPredicting}>
          Stop Live Predict
        </button>
        <button onClick={captureAndPredict} disabled={!mpReady}>
          Predict Once
        </button>
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
