"use client";

import { useState, useEffect } from "react";
import { Upload, Settings, Zap, BarChart3, Sparkles } from "lucide-react";

export default function Home() {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");

  const [hiddenSizes, setHiddenSizes] = useState("16,8");
  const [Lambda, setLambda] = useState("0.01");
  const [epochs, setEpochs] = useState("500");
  const [learningRate, setLearningRate] = useState("0.01");
  const [trainValSplit, setTrainValSplit] = useState("0.8");

  const [trainResult, setTrainResult] = useState<any>(null);
  const [trainLoading, setTrainLoading] = useState(false);
  const [testInput, setTestInput] = useState("0.5,0.5,0.5,0.5,0.5");
  const [testFile, setTestFile] = useState<File | null>(null);
  const [testMode, setTestMode] = useState<"values" | "file">("values");
  const [predictions, setPredictions] = useState<any>(null);

  const backend = "http://localhost:8000";

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a .pkl file to upload");
      return;
    }

    console.log("Starting upload for file:", file.name);
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
        setFileName(file.name);
        console.log("Upload successful, proceeding to step 3");
        setTimeout(() => {
          console.log("Moving to step 3");
          setStep(3);
        }, 1000);
      } else {
        console.error("Upload failed:", data);
        alert(data.error || "Upload failed");
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert(`Upload failed - ${error instanceof Error ? error.message : 'is the backend running?'}`);
    }
  };

  useEffect(() => {
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
  }, []); // empty dependency array → runs only once on mount

  const handleReset = async () => {
    try {
      // Call backend to clear uploads and outputs
      const res = await fetch(`${backend}/reset`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        console.error("Reset failed:", data.error);
        alert(data.error || "Failed to reset backend");
      } else {
        console.log(data.message);
      }

      // Clear frontend state
      setStep(1);
      setFile(null);
      setFileName("");
      setUploadMessage("");
      setTrainResult(null);
      setPredictions(null);
      setTestFile(null);
      setTestMode("values");
      setTestInput("0.5,0.5,0.5,0.5,0.5");
      setHiddenSizes("16,8");
      setLambda("0.01");
      setEpochs("500");
      setLearningRate("0.01");
      setTrainValSplit("0.8");

    } catch (error) {
      console.error("Reset error:", error);
      alert("Failed to reset the app. Is the backend running?");
    }
  };


  const handleTrain = async () => {
    setTrainLoading(true);
    const formData = new FormData();
    formData.append("pkl_filename", fileName);
    formData.append("hidden_sizes", hiddenSizes);
    formData.append("Lambda", Lambda);
    formData.append("epochs", epochs);
    formData.append("learning_rate", learningRate);
    formData.append("train_val_split", trainValSplit);

    const res = await fetch(`${backend}/train`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setTrainLoading(false);

    if (res.ok) {
      setTrainResult(data);
      setStep(5);
    } else {
      alert(data.error || "Training failed");
    }
  };

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append("hidden_sizes", hiddenSizes);
    formData.append("Lambda", Lambda);

    if (testMode === "file") {
      if (!testFile) {
        alert("Please select a .pkl file");
        return;
      }
      formData.append("input_file", testFile);
    } else {
      formData.append("input_values", testInput);
    }

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
        {/* Header with Reset Button */}
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

        {/* Progress Indicator */}
        <div className="mb-12">
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
                Upload a pickle file containing <code className="bg-gray-100 px-2 py-1 rounded text-sm">(X, y)</code> where{" "}
                <code className="bg-gray-100 px-2 py-1 rounded text-sm">X</code> has shape (N, 5).
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
              
              <div className="border-2 border-dashed border-indigo-300 rounded-2xl p-8 mb-6 bg-indigo-50/50 hover:border-indigo-500 transition-colors duration-200">
                <input
                  type="file"
                  accept=".pkl"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  className="w-full text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-gradient-to-r file:from-indigo-600 file:to-purple-600 file:text-white file:font-semibold file:cursor-pointer hover:file:shadow-lg file:transition-all"
                />
                {file && (
                  <p className="mt-4 text-indigo-700 font-medium">📄 Selected: {file.name}</p>
                )}
              </div>

              <button
                onClick={handleUpload}
                className="w-full px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg"
              >
                Upload Dataset
              </button>

              {uploadMessage && (
                <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-xl">
                  <p className="text-green-700 font-medium">{uploadMessage}</p>
                </div>
              )}
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
                  <p className="text-xs text-gray-500 mt-1">Comma-separated layer sizes</p>
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
                    value={epochs}
                    onChange={(e) => setEpochs(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                    placeholder="500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    step="0.001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                    placeholder="0.01"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Train/Validation Split
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={trainValSplit}
                    onChange={(e) => setTrainValSplit(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all outline-none text-gray-900"
                    placeholder="0.8"
                  />
                  <p className="text-xs text-gray-500 mt-1">Value between 0 and 1</p>
                </div>
              </div>

              <button
                onClick={handleTrain}
                disabled={trainLoading}
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
                <h2 className="text-3xl font-bold text-gray-800">Training Results</h2>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-2xl border border-blue-200">
                  <p className="text-sm font-semibold text-blue-700 mb-1">Train RMSE</p>
                  <p className="text-3xl font-bold text-blue-900">
                    {trainResult.train_loss_end.toFixed(4)}
                  </p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-2xl border border-purple-200">
                  <p className="text-sm font-semibold text-purple-700 mb-1">Validation RMSE</p>
                  <p className="text-3xl font-bold text-purple-900">
                    {trainResult.val_loss_end.toFixed(4)}
                  </p>
                </div>
              </div>

              <div className="space-y-6">
                {trainResult.plots.map((plot: string) => (
                  <div key={plot} className="rounded-2xl overflow-hidden shadow-lg border border-gray-200">
                    <img
                      src={`${backend}/plots/${plot}`}
                      alt={plot}
                      className="w-full"
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-3xl shadow-2xl p-10 border border-gray-100">
              <h3 className="text-2xl font-bold text-gray-800 mb-4">🔮 Test Your Model</h3>
              <p className="text-gray-600 mb-6">Choose to upload a pickle file or enter values manually</p>

              {/* Toggle between modes */}
              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => {
                    setTestMode("values");
                    setPredictions(null);
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
                        onChange={(e) => setTestFile(e.target.files?.[0] || null)}
                        className="w-full text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-gradient-to-r file:from-purple-600 file:to-pink-600 file:text-white file:font-semibold file:cursor-pointer hover:file:shadow-lg file:transition-all"
                      />
                      {testFile && (
                        <p className="mt-3 text-purple-700 font-medium">📄 Selected: {testFile.name}</p>
                      )}
                    </div>
                  </>
                )}

                <button
                  onClick={handlePredict}
                  className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:shadow-xl hover:scale-105 transition-all duration-200 font-semibold text-lg"
                >
                  Generate Prediction
                </button>

                {predictions && (
                  <div className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-200 rounded-2xl">
                    <p className="text-sm font-semibold text-purple-700 mb-2">Predicted Output</p>
                    <p className="text-2xl font-bold text-purple-900 font-mono break-all">
                      {JSON.stringify(predictions)}
                    </p>
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