/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useEffect } from "react";
import { Upload, Settings, Zap, BarChart3, Sparkles, CheckCircle, Loader } from "lucide-react";

type DatasetStats = {
  rows: number;
  features: number;
  x_min: number[];
  x_max: number[];
  y_min: number;
  y_max: number;
};

type TrainResult = {
  message: string;
  train_loss_start: number;
  train_loss_end: number;
  val_loss_start: number;
  val_loss_end: number;
  plots: string[];
  best_val_rmse?: number;
  best_train_rmse?: number;
  best_epoch?: number;
  epochs_run?: number;
  baseline_rmse?: number;
  artifacts?: string[];
  final_train_r2?: number;
  final_val_r2?: number;
  model_type?: string;
};

type Predictions = number[][] | null;

export default function Home() {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
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

  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
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
  const [uploadHistory, setUploadHistory] = useState<string[]>([]);
  const [modelType, setModelType] = useState<"numpy" | "tf">("numpy");
  const [plotKey, setPlotKey] = useState<number>(Date.now());

  const backend = "http://localhost:8000";
  const backendAvailable = healthStatus === "ok";

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    if (selectedFile) {
      setUploadMessage("");
      setDatasetStats(null);
      setUploadComplete(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a .pkl file to upload");
      return;
    }

    console.log("Starting upload for file:", file.name);
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("Sending request to:", `${backend}/upload`);
      const res = await fetch(`${backend}/upload`, {
        method: "POST",
        body: formData,
      });

      console.log("Response status:", res.status);
      const data = await res.json();
      console.log("Response data:", data);

      if (res.ok) {
        setUploadMessage(`✅ Uploaded: ${data.path || file.name}`);
        setDatasetStats(data.stats || null);
        setFileName(file.name);
        setUploadComplete(true);
        setUploadHistory((prev) => [file.name, ...prev].slice(0, 3));
        console.log("Upload successful, dataset stats ready");
      } else {
        console.error("Upload failed:", data);
        alert(data.error || "Upload failed");
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert(`Upload failed - ${error instanceof Error ? error.message : 'is the backend running?'}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleTestFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    if (selectedFile) {
      setTestFileUploading(true);
      setTestFile(selectedFile);
      setTimeout(() => {
        setTestFileUploading(false);
        setTestFileReady(true);
      }, 500);
    }
  };

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${backend}/health`);
        if (!res.ok) {
          setHealthStatus("error");
          return;
        }
        const data = await res.json();
        setHealthStatus(data.status === "ok" ? "ok" : "error");
      } catch (error) {
        console.error("Health check error:", error);
        setHealthStatus("error");
      }
    };

    const resetBackend = async () => {
      try {
        const res = await fetch(`${backend}/reset`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          console.error("Reset failed:", data.error);
        } else {
          console.log(data.message);
        }
      } catch (error) {
        console.error("Reset error:", error);
      }
    };

    resetBackend();
    checkHealth();
  }, []);

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

  const handleReset = async () => {
    try {
      const res = await fetch(`${backend}/reset`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        console.error("Reset failed:", data.error);
        alert(data.error || "Failed to reset backend");
      } else {
        console.log(data.message);
      }

      setStep(1);
      setFile(null);
      setFileName("");
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
    } catch (error) {
      console.error("Reset error:", error);
      alert("Failed to reset the app. Is the backend running?");
    }
  };

  const handleTrain = async () => {
    if (!uploadComplete || !fileName) {
      alert("Upload a dataset before training.");
      return;
    }
    if (!hyperparamsValid) return;
    setTrainLoading(true);
    const formData = new FormData();
    formData.append("pkl_filename", fileName);
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

    const res = await fetch(`${backend}/train`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setTrainLoading(false);

    if (res.ok) {
      const fallbackPlots =
        data.plots && data.plots.length
          ? data.plots
          : data.model_type === "tf"
          ? ["rmse_vs_epochs_tf.png", "ytrue_vs_ypred_tf.png"]
          : ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"];
      setTrainResult({ ...data, plots: fallbackPlots });
      setPlotKey(Date.now());
      setStep(5);
    } else {
      alert(data.error || "Training failed");
    }
  };

  const handlePredict = async () => {
    const formData = new FormData();

    if (testMode === "file") {
      if (!testFile) {
        alert("Please select a .pkl file");
        return;
      }
      formData.append("input_file", testFile);
    } else {
      if (!manualInputValid) {
        alert("Enter exactly 5 numeric values separated by commas.");
        return;
      }
      formData.append("input_values", testInput);
    }
    formData.append("model_type", modelType);

    try {
      const res = await fetch(`${backend}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) setPredictions(data.y_pred);
      else alert(data.error || "Prediction failed");
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Prediction failed - is the backend running?");
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {step > 1 && (
          <div className="mb-8 flex justify-end">
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-all duration-200 font-medium"
            >
              ← Start Over
            </button>
          </div>
        )}

        <div className="mb-12">
          <div className="flex justify-end mb-2">
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
          </div>
          {healthStatus === "error" && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-xl">
              Backend unavailable. Check that the server is running at {backend}.
            </div>
          )}
          <div className="flex items-center justify-center space-x-2 mb-4">
            {[1, 2, 3, 4, 5].map((s) => (
              <div key={s} className="flex items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all duration-300 ${
                    step >= s
                      ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg"
                      : "bg-gray-200 text-gray-500"
                  }`}
                >
                  {s}
                </div>
                {s < 5 && (
                  <div
                    className={`w-12 h-1 transition-all duration-300 ${
                      step > s ? "bg-indigo-600" : "bg-gray-200"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {step === 1 && (
          <div className="text-center animate-fade-in">
            <div className="bg-white rounded-3xl shadow-2xl p-12 border border-gray-100">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl mb-6">
                <Sparkles className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                5D Interpolator
              </h1>
              <p className="text-gray-600 text-lg mb-8 max-w-2xl mx-auto leading-relaxed">
                Train and test a powerful neural network interpolator using your dataset. 
                Upload a pickle file containing a dictionary with keys <code className="bg-gray-100 px-2 py-1 rounded text-sm">X</code> (shape (N, 5)) and <code className="bg-gray-100 px-2 py-1 rounded text-sm">y</code> (shape (N,) or (N, 1)); extra metadata is ignored.
              </p>
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
                      <div className="mt-4 flex items-center text-indigo-700 font-medium">
                        <CheckCircle className="w-5 h-5 mr-2" />
                        <span>📄 Selected: {file.name}</span>
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
                      <p className="text-green-700 font-medium">{uploadMessage}</p>
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
                        {uploadHistory.map((name) => (
                          <li key={name} className="flex items-center">
                            <span className="mr-2 text-indigo-500">•</span>
                            <span className="truncate">{name}</span>
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
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-4">
                  <Settings className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-gray-800">Configure Network</h2>
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
                    setHiddenSizes("64,32,16");
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

              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Hidden Layer Sizes
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
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Lambda (Regularization)
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Activation
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Weight Init
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Seed (optional)
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
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-4">
                  <Zap className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-gray-800">Training Parameters</h2>
              </div>

              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Epochs
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
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Learning Rate
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
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Train/Validation Split
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Batch Size
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Grad Clip
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Learning Rate Decay (optional)
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Early Stop Patience
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
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Beta1
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
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Beta2
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
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Epsilon
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
                  <p className="text-indigo-700 font-medium">Training in progress ({epochs} epochs)...</p>
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
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-2xl border border-blue-200">
                  <p className="text-sm font-semibold text-blue-700 mb-1">Train RMSE</p>
                  <p className="text-3xl font-bold text-blue-900">
                    {trainResult.train_loss_end.toFixed(4)}
                  </p>
                  {trainResult.best_train_rmse !== undefined && trainResult.best_epoch && (
                    <p className="text-xs text-blue-700 mt-1">
                      Best: {trainResult.best_train_rmse.toFixed(4)} (epoch {trainResult.best_epoch})
                    </p>
                  )}
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-2xl border border-purple-200">
                  <p className="text-sm font-semibold text-purple-700 mb-1">Validation RMSE</p>
                  <p className="text-3xl font-bold text-purple-900">
                    {trainResult.val_loss_end.toFixed(4)}
                  </p>
                  {trainResult.best_val_rmse !== undefined && trainResult.best_epoch && (
                    <p className="text-xs text-purple-700 mt-1">
                      Best: {trainResult.best_val_rmse.toFixed(4)} (epoch {trainResult.best_epoch})
                    </p>
                  )}
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-4 mb-8">
                {trainResult.baseline_rmse !== undefined && (
                  <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-4 rounded-2xl border border-amber-200">
                    <p className="text-sm font-semibold text-amber-700 mb-1">Baseline (mean) RMSE</p>
                    <p className="text-2xl font-bold text-amber-900">
                      {trainResult.baseline_rmse.toFixed(4)}
                    </p>
                  </div>
                )}
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
                {trainResult.final_train_r2 !== undefined && (
                  <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-2xl border border-green-200">
                    <p className="text-sm font-semibold text-green-700 mb-1">Train R²</p>
                    <p className="text-2xl font-bold text-green-900">
                      {trainResult.final_train_r2.toFixed(4)}
                    </p>
                  </div>
                )}
                {trainResult.final_val_r2 !== undefined && (
                  <div className="bg-gradient-to-br from-rose-50 to-rose-100 p-4 rounded-2xl border border-rose-200">
                    <p className="text-sm font-semibold text-rose-700 mb-1">Val R²</p>
                    <p className="text-2xl font-bold text-rose-900">
                      {trainResult.final_val_r2.toFixed(4)}
                    </p>
                  </div>
                )}
              </div>

              {trainResult.plots && trainResult.plots.length > 0 && (
                <div className="space-y-6">
                  {trainResult.plots.map((plot: string) => (
                    <div key={`${plot}-${trainResult.model_type}`} className="rounded-2xl overflow-hidden shadow-lg border border-gray-200">
                      <img
                        src={`${backend}/plots/${plot}?k=${plotKey}`}
                        alt={plot}
                        className="w-full"
                      />
                    </div>
                  ))}
                </div>
              )}

              {trainResult.artifacts && (
                <div className="mt-8">
                  <p className="text-sm font-semibold text-gray-800 mb-2">Artifacts</p>
                  <div className="grid md:grid-cols-3 gap-3">
                    {trainResult.artifacts.map((artifact) => (
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
              <h3 className="text-2xl font-bold text-gray-800 mb-4">🔮 Test Your Model</h3>
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
                    (testMode === "values" && !manualInputValid)
                  }
                  className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  Generate Prediction
                </button>

                {predictions && (
                  <div className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-200 rounded-2xl">
                    <p className="text-sm font-semibold text-purple-700 mb-2">Predicted Output</p>
                    <div className="flex items-center gap-3">
                      <p className="text-2xl font-bold text-purple-900 font-mono break-all">
                        {Number(predictions[0]?.[0] ?? 0).toFixed(4)}
                      </p>
                      <button
                        onClick={() => {
                          const val = predictions[0]?.[0];
                          if (val !== undefined) navigator.clipboard.writeText(String(val));
                        }}
                        className="px-3 py-1 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
                      >
                        Copy
                      </button>
                    </div>
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
