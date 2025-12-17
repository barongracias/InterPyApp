/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Upload, Settings, Zap, BarChart3, Sparkles, CheckCircle, Loader, Info } from "lucide-react";
import {
  DatasetStats,
  TrainResponse,
  healthCheck,
  predictModel,
  resetBackend,
  trainModel,
  uploadDataset,
} from "../lib/api";

type Predictions = number[][] | null;
const HelpTip = ({ text }: { text: string }) => (
  <span className="relative inline-flex items-center group align-middle">
    <Info className="w-4 h-4 text-indigo-500" aria-label={text} />
    <span className="absolute left-1/2 top-6 z-20 hidden min-w-[220px] max-w-xs -translate-x-1/2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs font-normal text-gray-700 shadow-lg group-hover:block">
      {text}
    </span>
  </span>
);

const DEFAULT_BACKEND = "http://localhost:8000";
const sanitizeBackend = (raw?: string) => {
  if (!raw) return DEFAULT_BACKEND;
  try {
    const url = new URL(raw);
    const normalised = `${url.origin}${url.pathname.replace(/\/$/, "")}`;
    return normalised || DEFAULT_BACKEND;
  } catch {
    return DEFAULT_BACKEND;
  }
};

export default function Home() {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [storedFilename, setStoredFilename] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [inlineMessage, setInlineMessage] = useState<{ type: "info" | "error" | "success"; text: string } | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const [hiddenSizes, setHiddenSizes] = useState("16,8");
  const [Lambda, setLambda] = useState("0.01");
  const [epochs, setEpochs] = useState("500");
  const [learningRate, setLearningRate] = useState("0.01");
  const [trainValSplit, setTrainValSplit] = useState("0.8");
  const [activation, setActivation] = useState("relu");
  const [weightInit, setWeightInit] = useState("auto");
  const [batchSize, setBatchSize] = useState("64");
  const [gradClip, setGradClip] = useState("5");
  const [seed, setSeed] = useState("42");
  const [lrDecay, setLrDecay] = useState("0.98");
  const [earlyStop, setEarlyStop] = useState("20");
  const [beta1, setBeta1] = useState("0.9");
  const [beta2, setBeta2] = useState("0.999");
  const [epsilon, setEpsilon] = useState("1e-8");

  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [trainLoading, setTrainLoading] = useState(false);
  const [testInput, setTestInput] = useState("0.5,0.5,0.5,0.5,0.5");
  const [testFile, setTestFile] = useState<File | null>(null);
  const [testFileUploading, setTestFileUploading] = useState(false);
  const [testFileReady, setTestFileReady] = useState(false);
  const [testMode, setTestMode] = useState<"values" | "file">("values");
  const [predictions, setPredictions] = useState<Predictions>(null);
  const [healthStatus, setHealthStatus] = useState<"checking" | "ok" | "error">("checking");
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [uploadComplete, setUploadComplete] = useState(false);
  type UploadEntry = { original: string; stored?: string };
  const [uploadHistory, setUploadHistory] = useState<UploadEntry[]>([]);
  const [modelType, setModelType] = useState<"numpy" | "tf">("numpy");
  const [plotKey, setPlotKey] = useState<number>(Date.now());
  const [showNetworkOptions, setShowNetworkOptions] = useState(false);
  const [showTrainingOptions, setShowTrainingOptions] = useState(false);
  const [trainDurationMs, setTrainDurationMs] = useState<number | null>(null);
  const [predictLoading, setPredictLoading] = useState(false);

  const backend = useMemo(() => sanitizeBackend(process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || DEFAULT_BACKEND), []);
  const backendAvailable = healthStatus === "ok";
  const stepLabels: Record<number, string> = {
    1: "Overview",
    2: "Upload",
    3: "Configure",
    4: "Train",
    5: "Results",
    6: "Test",
  };
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

  const filterArtifacts = (artifacts: string[] | undefined, modelType: "numpy" | "tf" | undefined) => {
    if (!artifacts) return [];
    if (modelType === "tf") {
      return artifacts.filter((name) => name.includes("tf"));
    }
    return artifacts.filter((name) => !name.includes("tf"));
  };

  const goToPrevStep = () => {
    setInlineMessage(null);
    setStep((prev) => Math.max(1, prev - 1));
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    if (selectedFile && selectedFile.size > MAX_UPLOAD_BYTES) {
      setInlineMessage({ type: "error", text: "File too large. Limit is 10 MB." });
      e.target.value = "";
      return;
    }
    setFile(selectedFile);
    if (selectedFile) {
      setUploadMessage("");
      setDatasetStats(null);
      setUploadComplete(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setInlineMessage({ type: "error", text: "Please select a .pkl file to upload" });
      return;
    }

    setIsUploading(true);
    setInlineMessage(null);
    try {
      const data = await uploadDataset(backend, file);
      const originalName = data.original_filename || file.name;
      const storedName = data.stored_filename || (data as { path?: string }).path;
      if (!storedName) {
        throw new Error("Upload response missing stored filename; cannot start training.");
      }
      setUploadMessage(`✅ Uploaded: ${originalName}`);
      setDatasetStats(data.stats || null);
      setFileName(originalName);
      setStoredFilename(storedName);
      latestStoredRef.current = storedName;
      setUploadComplete(true);
      setUploadHistory((prev) => [{ original: originalName, stored: storedName }, ...prev].slice(0, 3));
      setInlineMessage({ type: "success", text: "Upload successful, dataset stats ready" });
    } catch (error) {
      setUploadComplete(false);
      setStoredFilename("");
      latestStoredRef.current = "";
      setInlineMessage({ type: "error", text: `Upload failed - ${error instanceof Error ? error.message : "is the backend running?"}` });
    } finally {
      setIsUploading(false);
    }
  };

  const handleTestFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    if (selectedFile && selectedFile.size > MAX_UPLOAD_BYTES) {
      setInlineMessage({ type: "error", text: "File too large. Limit is 10 MB." });
      e.target.value = "";
      return;
    }
    if (selectedFile) {
      setTestFileUploading(true);
      setTestFile(selectedFile);
      setTimeout(() => {
        setTestFileUploading(false);
        setTestFileReady(true);
      }, 500);
    }
  };

  const runHealthCheck = useCallback(async () => {
    setHealthStatus("checking");
    const attempts = 3;
    for (let i = 0; i < attempts; i++) {
      try {
        const data = await healthCheck(backend);
        setHealthStatus(data.status === "ok" ? "ok" : "error");
        return;
      } catch (error) {
        // ignore and retry
      }
      if (i < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, 600));
      } else {
        setHealthStatus("error");
      }
    }
  }, [backend]);

  useEffect(() => {
    runHealthCheck();
  }, [runHealthCheck]);

  useEffect(() => {
    // Keep advanced panels collapsed when arriving on config/train steps
    if (step !== 3) setShowNetworkOptions(false);
    if (step !== 4) setShowTrainingOptions(false);
  }, [step]);

  const isPositiveNumber = (val: string) => {
    const num = Number(val);
    return Number.isFinite(num) && num > 0;
  };

  const hiddenSizesValid = hiddenSizes.split(",").map((s) => s.trim()).filter(Boolean).every((s) => /^\d+$/.test(s) && Number(s) > 0);
  const isPositiveIntOrEmpty = (val: string) => val === "" || (Number.isInteger(Number(val)) && Number(val) > 0);
  const activationValid = ["sigmoid", "tanh", "relu", "leakyrelu"].includes(activation.toLowerCase());
  const weightInitValid = ["auto", "he", "xavier"].includes(weightInit.toLowerCase());
  const lrDecayValid = lrDecay === "" || (Number(lrDecay) > 0 && Number(lrDecay) < 1);
  const earlyStopValid = earlyStop === "" || (Number.isInteger(Number(earlyStop)) && Number(earlyStop) > 0);
  const hyperparamsValid =
    hiddenSizesValid &&
    isPositiveNumber(Lambda) &&
    Number(trainValSplit) > 0 &&
    Number(trainValSplit) < 1 &&
    Number(epochs) > 0 &&
    isPositiveNumber(learningRate) &&
    Number(beta1) > 0 &&
    Number(beta1) < 1 &&
    Number(beta2) > 0 &&
    Number(beta2) < 1 &&
    isPositiveNumber(epsilon) &&
    activationValid &&
    weightInitValid &&
    isPositiveIntOrEmpty(batchSize) &&
    isPositiveNumber(gradClip) &&
    (seed === "" || Number.isInteger(Number(seed))) &&
    lrDecayValid &&
    earlyStopValid;

  const manualInputValid = testInput
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean).length === 5 &&
    testInput
      .split(",")
      .map((s) => Number(s.trim()))
      .every((n) => Number.isFinite(n));

  const formatNumber = (val: number) => Number(val).toFixed(4);
  const formatArray = (vals: number[]) => vals.map((v) => formatNumber(v)).join(", ");
  const flattenPreds = (preds: Predictions) =>
    Array.isArray(preds) ? preds.flat().filter((v): v is number => typeof v === "number" && Number.isFinite(v)) : [];
  const flatPreds = useMemo(() => flattenPreds(predictions), [predictions]);
  const latestStoredRef = useRef<string>("");

  useEffect(() => {
    if (!storedFilename && uploadHistory.length && uploadHistory[0]?.stored) {
      setStoredFilename(uploadHistory[0].stored || "");
      latestStoredRef.current = uploadHistory[0].stored || "";
    }
  }, [uploadHistory, storedFilename]);

  // Clear test state/predictions when switching backend to avoid showing stale results
  useEffect(() => {
    setPredictions(null);
    setTestFile(null);
    setTestFileReady(false);
    setTestFileUploading(false);
    setTestMode("values");
  }, [modelType]);

  const handleReset = async () => {
    try {
      const data = await resetBackend(backend);
      setInlineMessage({ type: "info", text: data.message || "Backend reset" });

      clearJobPolling();
      setJobId(null);
      setJobStatus(null);
      setStep(1);
      setFile(null);
      setFileName("");
      setStoredFilename("");
      setUploadMessage("");
      setIsUploading(false);
      setTrainResult(null);
      setPredictions(null);
      setDatasetStats(null);
      setTestFile(null);
      setTestFileUploading(false);
      setTestFileReady(false);
      setTestMode("values");
      setTestInput("0.5,0.5,0.5,0.5,0.5");
      setHiddenSizes("16,8");
      setLambda("0.01");
      setEpochs("500");
      setLearningRate("0.01");
      setTrainValSplit("0.8");
      setActivation("relu");
      setWeightInit("auto");
      setBatchSize("32");
      setGradClip("5");
      setSeed("42");
      setLrDecay("0.98");
      setEarlyStop("20");
      setBeta1("0.9");
      setBeta2("0.999");
      setEpsilon("1e-8");
      setUploadComplete(false);
      setUploadHistory([]);
      setModelType("numpy");
      setTrainDurationMs(null);
      setShowNetworkOptions(false);
      setShowTrainingOptions(false);
      setPredictLoading(false);
      setTrainLoading(false);
    } catch (error) {
      setInlineMessage({ type: "error", text: "Failed to reset the app. Is the backend running?" });
    }
  };

  const handleTrain = async () => {
    const trainingFile = storedFilename || latestStoredRef.current || uploadHistory[0]?.stored;
    if (!trainingFile) {
      setInlineMessage({ type: "error", text: "Upload a dataset before training." });
      return;
    }
    if (!hyperparamsValid) return;
    setPredictions(null);
    setTestFile(null);
    setTestFileReady(false);
    setTestFileUploading(false);
    setTrainResult(null);
    setTrainLoading(true);
    setTrainDurationMs(null);
    setInlineMessage(null);
    const started = performance.now();
    const formData = new FormData();
    formData.append("pkl_filename", trainingFile);
    formData.append("hidden_sizes", hiddenSizes);
    formData.append("Lambda", Lambda);
    formData.append("epochs", epochs);
    formData.append("learning_rate", learningRate);
    formData.append("train_val_split", trainValSplit);
    formData.append("beta1", beta1);
    formData.append("beta2", beta2);
    formData.append("epsilon", epsilon);
    formData.append("activation", activation);
    formData.append("weight_init", weightInit);
    formData.append("batch_size", batchSize);
    formData.append("grad_clip", gradClip);
    if (seed !== "") formData.append("seed", seed);
    if (lrDecay !== "") formData.append("lr_decay", lrDecay);
    if (earlyStop !== "") formData.append("early_stop_patience", earlyStop);
    formData.append("model_type", modelType);

    try {
      const data = await trainModel(backend, formData);
      const fallbackPlots =
        data.plots && data.plots.length
          ? data.plots
          : data.model_type === "tf"
          ? ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"]
          : ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"];
      setTrainResult({ ...data, plots: fallbackPlots });
      setPlotKey(Date.now());
      setStep(5);
      setTrainDurationMs(data.duration_ms ?? performance.now() - started);
      setInlineMessage({ type: "success", text: "Training completed" });
    } catch (error) {
      setTrainDurationMs(performance.now() - started);
      setInlineMessage({ type: "error", text: error instanceof Error ? error.message : "Training failed" });
    } finally {
      setTrainLoading(false);
    }
  };

  const handlePredict = async () => {
    const formData = new FormData();

    setInlineMessage(null);
    setPredictions(null);

    if (testMode === "file") {
      if (!testFile) {
        setInlineMessage({ type: "error", text: "Please select a .pkl file" });
        return;
      }
      formData.append("input_file", testFile);
    } else {
      if (!manualInputValid) {
        setInlineMessage({ type: "error", text: "Enter exactly 5 numeric values separated by commas." });
        return;
      }
      formData.append("input_values", testInput);
    }
    formData.append("model_type", modelType);

    try {
      setPredictLoading(true);
      const data = await predictModel(backend, formData);
      setPredictions(data.y_pred);
      setInlineMessage({ type: "success", text: "Prediction complete" });
    } catch (error) {
      setPredictions(null);
      setInlineMessage({ type: "error", text: error instanceof Error ? error.message : "Prediction failed - is the backend running?" });
    } finally {
      setPredictLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {step > 1 && (
          <div className="mb-8 flex justify-end">
            <div className="flex items-center gap-3">
              <button
                onClick={goToPrevStep}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-all duration-200 font-medium"
              >
                ← Back
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-all duration-200 font-medium"
              >
                Reset
              </button>
            </div>
          </div>
        )}

        <div className="mb-12">
          {healthStatus === "error" && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-xl">
              Backend unavailable. Check that the server is running at {backend}.
            </div>
          )}
          <div className="flex flex-col items-center justify-center space-y-3 mb-6">
            <div className="flex flex-col sm:flex-row items-center sm:space-x-3 space-y-2 sm:space-y-0">
              <span
                className={`px-3 py-1 text-sm font-semibold rounded-full ${
                  healthStatus === "ok"
                    ? "bg-green-100 text-green-700"
                    : healthStatus === "error"
                    ? "bg-red-100 text-red-700"
                    : "bg-gray-100 text-gray-600"
                }`}
              >
                {healthStatus === "checking"
                  ? "Checking backend..."
                  : healthStatus === "ok"
                  ? "Backend healthy"
                  : "Backend unavailable"}
              </span>
              <a
                href={backend}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-indigo-700 underline"
              >
                {backend}
              </a>
              <button
                onClick={() => {
                  setInlineMessage(null);
                  runHealthCheck();
                }}
                className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-xs hover:bg-indigo-700 transition disabled:opacity-50"
                disabled={healthStatus === "checking"}
              >
                Check
              </button>
            </div>
            <div className="flex items-center justify-center space-x-2">
              {[1, 2, 3, 4, 5, 6].map((s, idx) => (
                <div key={s} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <div
                      className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all duration-300 ${
                        step >= s
                          ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg"
                          : "bg-gray-200 text-gray-500"
                      }`}
                    >
                      {s}
                    </div>
                    <p className="mt-2 text-[11px] font-semibold text-gray-600 text-center min-w-[70px]">
                      {stepLabels[s]}
                    </p>
                  </div>
                  {idx < 5 && (
                    <div
                      className={`w-12 h-1 mx-2 transition-all duration-300 self-center ${
                        step > s ? "bg-indigo-600" : "bg-gray-200"
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {inlineMessage && (
          <div
            className={`mb-6 p-3 rounded-xl border ${
              inlineMessage.type === "error"
                ? "bg-red-50 border-red-200 text-red-700"
                : inlineMessage.type === "success"
                ? "bg-green-50 border-green-200 text-green-700"
                : "bg-blue-50 border-blue-200 text-blue-700"
            }`}
          >
            {inlineMessage.text}
          </div>
        )}

        {step === 1 && (
          <div className="text-center animate-fade-in">
            <div className="bg-white rounded-3xl shadow-2xl p-12 border border-gray-100">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl mb-6">
                <Sparkles className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                5D Interpolator
              </h1>
              <p className="text-gray-600 text-lg mb-6 max-w-3xl mx-auto leading-relaxed">
                Train and test a 5D neural network interpolator. Upload a pickle file with keys <code className="bg-gray-100 px-2 py-1 rounded text-sm">X</code> (N×5) and <code className="bg-gray-100 px-2 py-1 rounded text-sm">y</code> (N×1 or N), optionally configure network architecture and/or training parameters, then inspect metrics, plots, and artifacts.
              </p>
              <div className="text-left text-gray-700 max-w-3xl mx-auto space-y-3 mb-8">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-1" />
                  <div>
                    <p className="font-semibold text-gray-900">Upload your dataset</p>
                    <p className="text-sm text-gray-600">Provide a .pkl containing <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">X</code> (N×5) and <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">y</code> (N×1 or N); we surface row counts and feature ranges so you can verify it loaded.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-1" />
                  <div>
                    <p className="font-semibold text-gray-900">Choose backend & hyperparameters</p>
                    <p className="text-sm text-gray-600">Stick with sensible defaults or switch between NumPy and TensorFlow, adjusting hidden sizes, regularization, and optimizer settings.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-1" />
                  <div>
                    <p className="font-semibold text-gray-900">Train and review results</p>
                    <p className="text-sm text-gray-600">Execute training to see RMSE/R² alongside learning curves, then download artifacts and checkpoints directly from the UI.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-1" />
                  <div>
                    <p className="font-semibold text-gray-900">Validate predictions</p>
                    <p className="text-sm text-gray-600">Test by entering five comma-separated values or uploading a .pkl of evaluation points, then copy the predictions.</p>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setStep(2)}
                className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg"
              >
                Get Started
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="animate-fade-in">
            <div className="bg-white rounded-3xl shadow-2xl p-10 border border-gray-100">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-4">
                  <Upload className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-gray-800">Upload Training Data</h2>
              </div>

              <div className="grid md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                  <div className="border-2 border-dashed border-indigo-300 rounded-2xl p-8 mb-6 bg-indigo-50/50 hover:border-indigo-500 transition-colors duration-200">
                    <input
                      type="file"
                      accept=".pkl"
                      onChange={handleFileSelect}
                      className="w-full text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-gradient-to-r file:from-indigo-600 file:to-purple-600 file:text-white file:font-semibold file:cursor-pointer hover:file:shadow-lg file:transition-all"
                      disabled={isUploading || !backendAvailable}
                    />
                    {file && !isUploading && (
                      <div className="mt-4 flex items-center text-indigo-700 font-medium max-w-full">
                        <CheckCircle className="w-5 h-5 mr-2" />
                        <span className="truncate" title={file.name}>📄 Selected: {file.name}</span>
                      </div>
                    )}
                    {isUploading && (
                      <div className="mt-4 flex items-center text-indigo-600 font-medium">
                        <Loader className="w-5 h-5 mr-2 animate-spin" />
                        <span>Uploading file...</span>
                      </div>
                    )}
                  </div>

                  <button
                    onClick={handleUpload}
                    disabled={isUploading || !file || !backendAvailable}
                    className="w-full px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                  >
                    {isUploading ? "Uploading..." : "Upload Dataset"}
                  </button>

                  {uploadMessage && (
                    <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-xl space-y-3">
                      <p className="text-green-700 font-medium break-words">{uploadMessage}</p>
                    </div>
                  )}

                  {uploadComplete && (
                    <button
                      onClick={() => setStep(3)}
                      className="w-full mt-6 px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg"
                    >
                      Continue to Configure Network
                    </button>
                  )}
                </div>

                <div className="md:col-span-1 md:sticky md:top-6 space-y-4">
                  <div className="p-4 rounded-2xl border border-gray-200 bg-gradient-to-b from-white to-indigo-50 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">Dataset preview</h3>
                    {datasetStats ? (
                    <div className="text-sm text-gray-700 space-y-1">
                      <div className="flex justify-between font-semibold text-gray-900">
                        <span>Rows</span><span>{datasetStats.rows}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-semibold text-gray-900">Features</span><span>{datasetStats.features}</span>
                      </div>
                      <div className="mt-2">
                        <p className="font-semibold text-gray-900">X min</p>
                        <p className="font-mono text-xs text-gray-700">{formatArray(datasetStats.x_min)}</p>
                      </div>
                      <div className="mt-2">
                        <p className="font-semibold text-gray-900">X max</p>
                        <p className="font-mono text-xs text-gray-700">{formatArray(datasetStats.x_max)}</p>
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        <div>
                          <p className="font-semibold text-gray-900">y min</p>
                          <p className="font-mono text-xs text-gray-700">{formatNumber(datasetStats.y_min)}</p>
                        </div>
                        <div>
                          <p className="font-semibold text-gray-900">y max</p>
                          <p className="font-mono text-xs text-gray-700">{formatNumber(datasetStats.y_max)}</p>
                        </div>
                      </div>
                    </div>
                    ) : (
                      <p className="text-sm text-gray-500">Select and upload a .pkl file to preview stats.</p>
                    )}
                  </div>

                  {uploadHistory.length > 0 && (
                    <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                      <h4 className="text-sm font-semibold text-gray-800 mb-2">Recent uploads</h4>
                      <ul className="space-y-1 text-sm text-gray-700">
                        {uploadHistory.map((entry, idx) => (
                          <li key={entry.stored || `${entry.original}-${idx}`} className="flex items-center">
                            <span className="mr-2 text-indigo-500">•</span>
                            <span className="truncate">{entry.original}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="animate-fade-in">
            <div className="bg-white rounded-3xl shadow-2xl p-10 border border-gray-100">
              <div className="flex items-center mb-6 justify-between">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-4">
                    <Settings className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold text-gray-800">Configure Network</h2>
                    <p className="text-sm text-gray-600">Stick with sensible defaults or expand to customise architecture.</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowNetworkOptions((v) => !v)}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg font-semibold transition-all"
                >
                  {showNetworkOptions ? "Hide options" : "Customise"}
                </button>
              </div>

              <div className="flex gap-3 mb-6">
                <button
                  onClick={() => setModelType("numpy")}
                  className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
                    modelType === "numpy"
                      ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  NumPy backend
                </button>
                <button
                  onClick={() => {
                    setModelType("tf");
                    setLearningRate("0.0015");
                    setBatchSize("64");
                    setGradClip("5");
                  }}
                  className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
                    modelType === "tf"
                      ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  TensorFlow backend
                </button>
              </div>
              {modelType === "tf" && (
                <p className="mb-6 text-sm text-indigo-700 bg-indigo-50 border border-indigo-100 rounded-xl px-3 py-2">
                  TensorFlow uses defined activation/init and Adam betas/epsilon. Keep learning rate modest on CPU-only builds.
                </p>
              )}

              {showNetworkOptions ? (
                <div className="space-y-5">
                  <div>
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                      Hidden Layer Sizes
                      <HelpTip text="Comma-separated layer widths (e.g., 16,8). Deeper/wider networks learn more complex patterns but train slower and can overfit small datasets." />
                    </label>
                    <input
                      type="text"
                      value={hiddenSizes}
                      onChange={(e) => setHiddenSizes(e.target.value)}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                      placeholder="e.g., 16,8"
                    />
                    <p className="text-xs text-gray-500 mt-1">Comma-separated positive integers (default: 16,8)</p>
                    {!hiddenSizesValid && <p className="text-xs text-red-600 mt-1">Enter at least one integer layer size.</p>}
                  </div>

                  <div>
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                      Lambda (Regularization)
                      <HelpTip text="L2 weight penalty. Higher values shrink weights more to reduce overfitting; too high can underfit." />
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={Lambda}
                      onChange={(e) => setLambda(e.target.value)}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                      placeholder="e.g., 0.01"
                    />
                    <p className="text-xs text-gray-500 mt-1">Regularization strength (default: 0.01)</p>
                    {!isPositiveNumber(Lambda) && <p className="text-xs text-red-600 mt-1">Must be a positive number.</p>}
                  </div>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div>
                      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                        Activation
                        <HelpTip text="Nonlinearity per hidden layer. ReLU is a good default; LeakyReLU can help avoid dead neurons; tanh/sigmoid bound outputs but can slow training." />
                      </label>
                      <select
                        value={activation}
                        onChange={(e) => setActivation(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                      >
                        <option value="relu">ReLU</option>
                        <option value="leakyrelu">LeakyReLU</option>
                        <option value="tanh">tanh</option>
                        <option value="sigmoid">sigmoid</option>
                      </select>
                      {!activationValid && <p className="text-xs text-red-600 mt-1">Choose a valid activation.</p>}
                    </div>
                    <div>
                      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                        Weight Init
                        <HelpTip text="Starting weights. Auto picks He for ReLU/LeakyReLU and Xavier for tanh/sigmoid; choose manually if you want to experiment." />
                      </label>
                      <select
                        value={weightInit}
                        onChange={(e) => setWeightInit(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                      >
                        <option value="auto">auto</option>
                        <option value="he">he</option>
                        <option value="xavier">xavier</option>
                      </select>
                      {!weightInitValid && <p className="text-xs text-red-600 mt-1">Choose a valid init.</p>}
                    </div>
                    <div>
                      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                        Seed (optional)
                        <HelpTip text="Random seed for reproducible init/shuffling. Leave blank for nondeterministic runs." />
                      </label>
                      <input
                        type="number"
                        value={seed}
                        onChange={(e) => setSeed(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                        placeholder="e.g., 42"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-indigo-50 shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Backend</p>
                    <p className="text-lg font-bold text-gray-900">{modelType === "tf" ? "TensorFlow" : "NumPy"}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Hidden layers</p>
                    <p className="text-lg font-bold text-gray-900">{hiddenSizes}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Activation</p>
                    <p className="text-lg font-bold text-gray-900 capitalize">{activation}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Weight init</p>
                    <p className="text-lg font-bold text-gray-900">{weightInit}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Lambda</p>
                    <p className="text-lg font-bold text-gray-900">{Lambda}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Seed</p>
                    <p className="text-lg font-bold text-gray-900">{seed || "None"}</p>
                  </div>
                </div>
              )}

              <button
                onClick={() => setStep(4)}
                className="w-full mt-8 px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg"
              >
                Continue to Training
              </button>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="animate-fade-in">
            <div className="bg-white rounded-3xl shadow-2xl p-10 border border-gray-100">
              <div className="flex items-center mb-6 justify-between">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-4">
                    <Zap className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold text-gray-800">Training Parameters</h2>
                    <p className="text-sm text-gray-600">Keep defaults or expand to fine-tune.</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowTrainingOptions((v) => !v)}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg font-semibold transition-all"
                >
                  {showTrainingOptions ? "Hide options" : "Customise"}
                </button>
              </div>

              {showTrainingOptions ? (
                <div className="space-y-5">
                  <div>
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                      Epochs
                      <HelpTip text="Full passes over the training data. More epochs can improve fit but increase time and overfitting risk." />
                    </label>
                    <input
                      type="number"
                      min={1}
                      value={epochs}
                      onChange={(e) => setEpochs(e.target.value)}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                      placeholder="500"
                    />
                    <p className="text-xs text-gray-500 mt-1">Number of training iterations (default: 500)</p>
                    {Number(epochs) <= 0 && <p className="text-xs text-red-600 mt-1">Enter a positive integer.</p>}
                  </div>

                  <div>
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                      Learning Rate
                      <HelpTip text="Step size for gradient updates. Higher learns faster but may diverge; lower is stabler but slower." />
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      min={0}
                      value={learningRate}
                      onChange={(e) => setLearningRate(e.target.value)}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                      placeholder="0.01"
                    />
                    <p className="text-xs text-gray-500 mt-1">Step size for gradient descent (default: 0.01)</p>
                    {!isPositiveNumber(learningRate) && <p className="text-xs text-red-600 mt-1">Must be positive.</p>}
                  </div>

                  <div>
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                      Train/Validation Split
                      <HelpTip text="Fraction of data used for training; the rest is held out for validation/early stopping. Smaller training splits can reduce fit quality." />
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      min={0.1}
                      max={0.9}
                      value={trainValSplit}
                      onChange={(e) => setTrainValSplit(e.target.value)}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                      placeholder="0.8"
                    />
                    <p className="text-xs text-gray-500 mt-1">Fraction for training vs validation (default: 0.8)</p>
                    {!(Number(trainValSplit) > 0 && Number(trainValSplit) < 1) && (
                      <p className="text-xs text-red-600 mt-1">Use a value between 0 and 1 (e.g., 0.8).</p>
                    )}
                  </div>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div>
                      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                        Batch Size
                        <HelpTip text="Samples per gradient step. Larger batches are smoother but use more memory; leave blank for full-batch." />
                      </label>
                      <input
                        type="number"
                        min={1}
                        value={batchSize}
                        onChange={(e) => setBatchSize(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                        placeholder="32"
                      />
                      {!isPositiveIntOrEmpty(batchSize) && <p className="text-xs text-red-600 mt-1">Must be a positive integer.</p>}
                    </div>
                    <div>
                      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                        Grad Clip
                        <HelpTip text="Upper bound on gradient norm to prevent exploding gradients. Lower values clip more aggressively." />
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        min={0}
                        value={gradClip}
                        onChange={(e) => setGradClip(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                        placeholder="5.0"
                      />
                      {!isPositiveNumber(gradClip) && <p className="text-xs text-red-600 mt-1">Must be positive.</p>}
                    </div>
                    <div>
                      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                        Learning Rate Decay (optional)
                        <HelpTip text="Multiplier applied to the learning rate each epoch (<1 slows learning over time). Leave blank to keep LR constant." />
                      </label>
                      <input
                        type="number"
                        step="0.01"
                        min={0}
                        max={1}
                        value={lrDecay}
                        onChange={(e) => setLrDecay(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                        placeholder="0.98"
                      />
                      {!lrDecayValid && <p className="text-xs text-red-600 mt-1">Use 0-1 (e.g., 0.98) or leave blank.</p>}
                    </div>
                    <div>
                      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                        Early Stop Patience
                        <HelpTip text="Stop training after this many epochs without validation improvement. Lower values stop earlier to limit overfitting." />
                      </label>
                      <input
                        type="number"
                        min={1}
                        value={earlyStop}
                        onChange={(e) => setEarlyStop(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                        placeholder="20"
                      />
                      {!earlyStopValid && <p className="text-xs text-red-600 mt-1">Must be a positive integer or blank.</p>}
                    </div>
                  </div>

                  <div className="pt-4 border-t border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Adam Optimizer Parameters (Optional)</h3>
                    
                    <div className="space-y-5">
                      <div>
                        <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                          Beta1
                          <HelpTip text="Adam momentum for the first moment (mean). Higher smooths updates but responds slower to changes." />
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          min={0}
                          max={1}
                          value={beta1}
                          onChange={(e) => setBeta1(e.target.value)}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                          placeholder="0.9"
                        />
                        <p className="text-xs text-gray-500 mt-1">Exponential decay rate for first moment (default: 0.9)</p>
                        {!(Number(beta1) > 0 && Number(beta1) < 1) && <p className="text-xs text-red-600 mt-1">Keep Beta1 between 0 and 1.</p>}
                      </div>

                      <div>
                        <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                          Beta2
                          <HelpTip text="Adam momentum for the second moment (variance). High values stabilize but can slow adaptation; keep below 1." />
                        </label>
                        <input
                          type="number"
                          step="0.001"
                          min={0}
                          max={1}
                          value={beta2}
                          onChange={(e) => setBeta2(e.target.value)}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                          placeholder="0.999"
                        />
                        <p className="text-xs text-gray-500 mt-1">Exponential decay rate for second moment (default: 0.999)</p>
                        {!(Number(beta2) > 0 && Number(beta2) < 1) && <p className="text-xs text-red-600 mt-1">Keep Beta2 between 0 and 1.</p>}
                      </div>

                      <div>
                        <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                          Epsilon
                          <HelpTip text="Small constant to avoid divide-by-zero in Adam; usually leave at the default unless debugging numerical issues." />
                        </label>
                        <input
                          type="text"
                          value={epsilon}
                          onChange={(e) => setEpsilon(e.target.value)}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                          placeholder="1e-8"
                        />
                        <p className="text-xs text-gray-500 mt-1">Small constant for numerical stability (default: 1e-8)</p>
                        {!isPositiveNumber(epsilon) && <p className="text-xs text-red-600 mt-1">Must be positive.</p>}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-indigo-50 shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Epochs</p>
                    <p className="text-lg font-bold text-gray-900">{epochs}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Learning rate</p>
                    <p className="text-lg font-bold text-gray-900">{learningRate}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Train/Val split</p>
                    <p className="text-lg font-bold text-gray-900">{trainValSplit}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Batch size</p>
                    <p className="text-lg font-bold text-gray-900">{batchSize || "Full batch"}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Grad clip</p>
                    <p className="text-lg font-bold text-gray-900">{gradClip}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">LR decay</p>
                    <p className="text-lg font-bold text-gray-900">{lrDecay || "None"}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Early stop</p>
                    <p className="text-lg font-bold text-gray-900">{earlyStop || "None"}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Adam β1 / β2</p>
                    <p className="text-lg font-bold text-gray-900">{beta1} / {beta2}</p>
                  </div>
                  <div className="p-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
                    <p className="text-sm font-semibold text-gray-700">Epsilon</p>
                    <p className="text-lg font-bold text-gray-900">{epsilon}</p>
                  </div>
                </div>
              )}

              <button
                onClick={handleTrain}
                disabled={trainLoading || !backendAvailable || !uploadComplete || !hyperparamsValid}
                className="w-full mt-8 px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                {trainLoading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Training Model...
                  </span>
                ) : (
                  "Start Training"
                )}
              </button>

              {!uploadComplete && (
                <p className="mt-3 text-sm text-amber-700 bg-amber-50 border border-amber-200 px-3 py-2 rounded-xl">
                  Upload a dataset first to enable training.
                </p>
              )}

              {trainLoading && (
                <div className="mt-4 bg-indigo-50 border border-indigo-100 rounded-xl p-3">
                  <p className="text-indigo-700 font-medium">
                    {`Training in progress (${epochs} epochs)...`}
                  </p>
                  <div className="mt-2 h-2 rounded-full bg-indigo-100 overflow-hidden">
                    <div className="h-full w-full bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-600 animate-pulse" />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {step === 5 && trainResult && (
          <div className="animate-fade-in space-y-8">
            <div className="bg-white rounded-3xl shadow-2xl p-10 border border-gray-100">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl flex items-center justify-center mr-4">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
                <div className="flex flex-col">
                  <h2 className="text-3xl font-bold text-gray-800">Training Results</h2>
                  <span className="mt-1 inline-flex px-3 py-1 text-xs font-semibold rounded-full bg-indigo-100 text-indigo-700">
                    {trainResult.model_type === "tf" ? "TensorFlow backend" : "NumPy backend"}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-2xl border border-blue-200 space-y-1">
                  <p className="text-sm font-semibold text-blue-700">Train metrics</p>
                  <p className="text-xl font-bold text-blue-900">
                    RMSE: {trainResult.train_loss_end !== undefined ? trainResult.train_loss_end.toFixed(4) : "–"}
                  </p>
                  {trainResult.final_train_r2 !== undefined && (
                    <p className="text-sm font-semibold text-blue-800">R²: {trainResult.final_train_r2.toFixed(4)}</p>
                  )}
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-2xl border border-purple-200 space-y-1">
                  <p className="text-sm font-semibold text-purple-700">Validation metrics</p>
                  <p className="text-xl font-bold text-purple-900">
                    RMSE: {trainResult.val_loss_end !== undefined ? trainResult.val_loss_end.toFixed(4) : "–"}
                  </p>
                  {trainResult.final_val_r2 !== undefined && (
                    <p className="text-sm font-semibold text-purple-800">R²: {trainResult.final_val_r2.toFixed(4)}</p>
                  )}
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-4 mb-8">
                {trainResult.epochs_run !== undefined && (
                  <div className="bg-gradient-to-br from-teal-50 to-teal-100 p-4 rounded-2xl border border-teal-200">
                    <p className="text-sm font-semibold text-teal-700 mb-1">Epochs executed</p>
                    <p className="text-2xl font-bold text-teal-900">{trainResult.epochs_run}</p>
                  </div>
                )}
                {trainResult.best_epoch !== undefined && (
                  <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-4 rounded-2xl border border-indigo-200">
                    <p className="text-sm font-semibold text-indigo-700 mb-1">Best epoch</p>
                    <p className="text-2xl font-bold text-indigo-900">{trainResult.best_epoch}</p>
                  </div>
                )}
                <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-4 rounded-2xl border border-amber-200">
                  <p className="text-sm font-semibold text-amber-700 mb-1">Training time</p>
                  <p className="text-2xl font-bold text-amber-900">
                    {trainDurationMs ? `${(trainDurationMs / 1000).toFixed(2)}s` : "–"}
                  </p>
                  <p className="text-xs text-amber-700">Measured client-side</p>
                </div>
              </div>

              {trainResult.plots && trainResult.plots.length > 0 && (
                <div className="space-y-6">
                  {trainResult.plots.map((plot: string) => (
                    <div key={`${plot}-${trainResult.model_type}`} className="rounded-2xl overflow-hidden shadow-lg border border-gray-200">
                      <img
                        src={`${backend}/plots/${plot}?k=${plotKey}&model_type=${trainResult.model_type}`}
                        alt={plot}
                        className="w-full"
                      />
                    </div>
                  ))}
                </div>
              )}

              {filterArtifacts(trainResult.artifacts, trainResult.model_type as "numpy" | "tf").length > 0 && (
                <div className="mt-8">
                  <p className="text-sm font-semibold text-gray-800 mb-2">Download Model & Weights</p>
                  <div className="grid md:grid-cols-3 gap-3">
                    {filterArtifacts(trainResult.artifacts, trainResult.model_type as "numpy" | "tf").map((artifact) => (
                      <a
                        key={artifact}
                        href={`${backend}/artifacts/${artifact}`}
                        className="px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-800 hover:bg-gray-100 transition"
                      >
                        {artifact}
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="bg-white rounded-3xl shadow-2xl p-10 border border-gray-100">
              <div className="flex items-center justify-between">
                <div className="text-lg text-gray-700">Ready to validate the trained model?</div>
                <button
                  onClick={() => setStep(6)}
                  className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-lg hover:scale-105 transition-all duration-200 font-semibold"
                >
                  Go to Testing
                </button>
              </div>
            </div>
          </div>
        )}

        {step === 6 && trainResult && (
          <div className="animate-fade-in space-y-8">
            <div className="bg-white rounded-3xl shadow-2xl p-10 border border-gray-100">
              <h3 className="text-2xl font-bold text-gray-800 mb-4">Test Your Model</h3>
              <p className="text-gray-600 mb-6">Choose to upload a pickle file or enter values manually</p>

              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => {
                    setTestMode("values");
                    setPredictions(null);
                    setTestFile(null);
                    setTestFileReady(false);
                  }}
                  className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
                    testMode === "values"
                      ? "bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  Manual Input
                </button>
                <button
                  onClick={() => {
                    setTestMode("file");
                    setPredictions(null);
                  }}
                  className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
                    testMode === "file"
                      ? "bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  Upload File
                </button>
              </div>

              <div className="space-y-4">
                {testMode === "values" ? (
                  <>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Enter 5 comma-separated values
                    </label>
                    <input
                      type="text"
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all outline-none font-mono text-gray-900"
                      placeholder="0.5,0.5,0.5,0.5,0.5"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Provide 5 numbers between 0 and 1 separated by commas, e.g. <span className="font-mono">0.2,0.4,0.6,0.3,0.5</span>.
                    </p>
                    {!manualInputValid && (
                      <p className="text-xs text-red-600 mt-1">Expecting exactly 5 numeric values.</p>
                    )}
                  </>
                ) : (
                  <>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Upload test data (.pkl file)
                    </label>
                    <div className="border-2 border-dashed border-purple-300 rounded-2xl p-6 bg-purple-50/50 hover:border-purple-500 transition-colors duration-200">
                      <input
                        type="file"
                        accept=".pkl"
                        onChange={handleTestFileSelect}
                        className="w-full text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-gradient-to-r file:from-purple-600 file:to-pink-600 file:text-white file:font-semibold file:cursor-pointer hover:file:shadow-lg file:transition-all"
                        disabled={testFileUploading}
                      />
                      {testFileUploading && (
                        <div className="mt-3 flex items-center text-purple-600 font-medium">
                          <Loader className="w-5 h-5 mr-2 animate-spin" />
                          <span>Processing file...</span>
                        </div>
                      )}
                      {testFileReady && testFile && (
                        <div className="mt-3 flex items-center text-purple-700 font-medium">
                          <CheckCircle className="w-5 h-5 mr-2" />
                          <span>📄 Ready: {testFile.name}</span>
                        </div>
                      )}
                    </div>
                  </>
                )}

                <button
                  onClick={handlePredict}
                  disabled={
                    !backendAvailable ||
                    (testMode === "file" && (!testFile || testFileUploading)) ||
                    (testMode === "values" && !manualInputValid) ||
                    predictLoading
                  }
                  className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  {predictLoading ? "Generating..." : "Generate Prediction"}
                </button>

                {predictions && (
                  <div className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-200 rounded-2xl">
                    <p className="text-sm font-semibold text-purple-700 mb-2">
                      Predicted Outputs ({flatPreds.length})
                    </p>
                    <div className="max-h-48 overflow-auto grid grid-cols-2 sm:grid-cols-3 gap-2">
                      {flatPreds.map((val, idx) => (
                        <div key={idx} className="px-3 py-2 bg-white border border-purple-100 rounded-lg text-purple-900 font-mono text-sm">
                          #{idx + 1}: {Number(val).toFixed(4)}
                        </div>
                      ))}
                    </div>
                    {flatPreds.length > 0 && (
                      <button
                        onClick={() => navigator.clipboard.writeText(flatPreds.join(", "))}
                        className="mt-3 px-3 py-2 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
                      >
                        Copy all values
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
