import { useState, useRef } from "react";
import { useSession } from "../hooks/useSession";

export default function UploadPanel({ onExplainReady, onMetricsReady }) {
  const { uploadModel, uploadData, generateExplanations, fetchMetrics, modelInfo, dataInfo } =
    useSession();

  const [modelFile, setModelFile] = useState(null);
  const [dataFile, setDataFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [instanceIndex, setInstanceIndex] = useState(0);
  const [numLimeSamples, setNumLimeSamples] = useState(5000);

  const [modelStatus, setModelStatus] = useState(null);
  const [dataStatus, setDataStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError] = useState(null);

  const modelInputRef = useRef();
  const dataInputRef = useRef();

  const handleModelUpload = async () => {
    if (!modelFile) return;
    setLoading(true);
    setLoadingStep("Loading model…");
    setError(null);
    try {
      const result = await uploadModel(modelFile);
      setModelStatus(result);
    } catch (e) {
      setError(`Model upload failed: ${e.message}`);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const handleDataUpload = async () => {
    if (!dataFile) return;
    setLoading(true);
    setLoadingStep("Validating dataset…");
    setError(null);
    try {
      const result = await uploadData(dataFile, targetColumn);
      setDataStatus(result);
    } catch (e) {
      setError(`Data upload failed: ${e.message}`);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const handleRunAnalysis = async () => {
    if (!modelInfo || !dataStatus) {
      setError("Please upload both a model and dataset first.");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      setLoadingStep("Computing SHAP values (global interpretability)…");
      const explanations = await generateExplanations({
        instanceIndex: parseInt(instanceIndex) || 0,
        numLimeSamples: parseInt(numLimeSamples) || 5000,
      });
      onExplainReady(explanations);

      if (dataStatus?.has_target) {
        setLoadingStep("Computing performance metrics…");
        try {
          const metrics = await fetchMetrics();
          onMetricsReady(metrics);
        } catch (e) {
          console.warn("Metrics failed:", e.message);
        }
      }
    } catch (e) {
      setError(`Analysis failed: ${e.message}`);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="border-l-4 border-cyan-400 pl-4">
        <h2 className="text-lg font-black text-white tracking-tight">Upload & Configure</h2>
        <p className="text-xs text-slate-500 mt-1">
          Upload your trained scikit-learn model and dataset to begin XAI analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Upload */}
        <UploadCard
          title="01 · Model File"
          subtitle=".pkl or .joblib — scikit-learn / XGBoost"
          accept=".pkl,.joblib"
          file={modelFile}
          onFileChange={setModelFile}
          onUpload={handleModelUpload}
          status={modelStatus}
          loading={loading && loadingStep.includes("model")}
          inputRef={modelInputRef}
          successContent={
            modelStatus && (
              <div className="space-y-1 text-xs">
                <InfoRow label="Type" value={modelStatus.model_type} highlight />
                <InfoRow
                  label="Task"
                  value={modelStatus.is_classifier ? "Classification" : "Regression"}
                />
                <InfoRow
                  label="Features"
                  value={modelStatus.n_features ?? "Unknown"}
                />
                <InfoRow
                  label="Classes"
                  value={modelStatus.n_classes ?? "N/A"}
                />
                <InfoRow
                  label="SHAP backend"
                  value={modelStatus.explainer_backend}
                  mono
                />
              </div>
            )
          }
        />

        {/* Data Upload */}
        <UploadCard
          title="02 · Dataset"
          subtitle=".csv — numeric features, optional target column"
          accept=".csv"
          file={dataFile}
          onFileChange={setDataFile}
          onUpload={handleDataUpload}
          status={dataStatus}
          loading={loading && loadingStep.includes("dataset")}
          inputRef={dataInputRef}
          extraFields={
            <div className="mt-3">
              <label className="block text-xs text-slate-500 uppercase tracking-wider mb-1">
                Target Column (optional, for metrics)
              </label>
              <input
                type="text"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                placeholder="e.g. target, label, y"
                className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs px-3 py-2 rounded focus:outline-none focus:border-cyan-400 placeholder-slate-600"
              />
            </div>
          }
          successContent={
            dataStatus && (
              <div className="space-y-1 text-xs">
                <InfoRow
                  label="Shape"
                  value={`${dataStatus.n_rows} rows × ${dataStatus.n_features} features`}
                  highlight
                />
                <InfoRow
                  label="Target"
                  value={dataStatus.has_target ? "Detected" : "Not provided"}
                />
                {dataStatus.compatibility_issues?.length > 0 && (
                  <div className="mt-2 p-2 bg-amber-400/10 border border-amber-400/30 rounded text-amber-400">
                    ⚠ {dataStatus.compatibility_issues[0]}
                  </div>
                )}
              </div>
            )
          }
        />
      </div>

      {/* Analysis Configuration */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
          03 · Analysis Configuration
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-slate-500 uppercase tracking-wider mb-1">
              Instance Index for Local Explanations
            </label>
            <input
              type="number"
              min="0"
              value={instanceIndex}
              onChange={(e) => setInstanceIndex(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-sm px-3 py-2 rounded focus:outline-none focus:border-cyan-400"
            />
            <p className="text-xs text-slate-600 mt-1">
              Row index in dataset to use for SHAP force plot & LIME explanation
            </p>
          </div>
          <div>
            <label className="block text-xs text-slate-500 uppercase tracking-wider mb-1">
              LIME Perturbation Samples
            </label>
            <select
              value={numLimeSamples}
              onChange={(e) => setNumLimeSamples(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-sm px-3 py-2 rounded focus:outline-none focus:border-cyan-400"
            >
              <option value={1000}>1,000 (fast, less stable)</option>
              <option value={3000}>3,000 (balanced)</option>
              <option value={5000}>5,000 (standard, recommended)</option>
              <option value={10000}>10,000 (stable, slow)</option>
            </select>
            <p className="text-xs text-slate-600 mt-1">
              Higher samples → more stable LIME explanations, slower computation
            </p>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-950/50 border border-red-800 text-red-400 rounded-lg p-4 text-sm">
          <span className="font-bold">Error: </span>{error}
        </div>
      )}

      {/* Run Button */}
      <button
        onClick={handleRunAnalysis}
        disabled={loading || !modelInfo || !dataStatus}
        className="w-full py-4 bg-cyan-400 text-slate-950 font-black text-sm tracking-widest uppercase rounded-lg
                   hover:bg-cyan-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed
                   flex items-center justify-center gap-3"
      >
        {loading ? (
          <>
            <Spinner />
            <span>{loadingStep || "Processing…"}</span>
          </>
        ) : (
          "◈ Run XAI Analysis — Generate Explanations"
        )}
      </button>

      {/* Academic Note */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 text-xs text-slate-600 space-y-1">
        <p className="text-slate-500 font-bold">Academic Note — Computational Complexity</p>
        <p>SHAP TreeExplainer: O(TLD²) — polynomial for tree ensembles. Exact Shapley values.</p>
        <p>LIME: O(N × model_inference) — N perturbations. Model-agnostic, always local.</p>
        <p>KernelExplainer (non-tree models): O(2^M × N) — exponential; approximated via sampling.</p>
      </div>
    </div>
  );
}

// ---- Sub-components ----

function UploadCard({
  title, subtitle, accept, file, onFileChange, onUpload, status,
  loading, inputRef, extraFields, successContent,
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-col gap-4">
      <div>
        <h3 className="text-sm font-bold text-white tracking-tight">{title}</h3>
        <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
      </div>

      <div
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          file
            ? "border-cyan-400/50 bg-cyan-400/5"
            : "border-slate-700 hover:border-slate-500"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => onFileChange(e.target.files[0])}
        />
        {file ? (
          <div>
            <p className="text-cyan-400 font-bold text-sm">{file.name}</p>
            <p className="text-slate-600 text-xs mt-1">
              {(file.size / 1024).toFixed(1)} KB
            </p>
          </div>
        ) : (
          <div>
            <p className="text-slate-600 text-2xl mb-1">⬆</p>
            <p className="text-slate-500 text-xs">Click to select file</p>
          </div>
        )}
      </div>

      {extraFields}

      <button
        onClick={onUpload}
        disabled={!file || loading}
        className="w-full py-2.5 bg-slate-800 text-slate-200 text-xs font-bold tracking-widest uppercase rounded
                   hover:bg-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed
                   border border-slate-700 hover:border-slate-500"
      >
        {loading ? "Processing…" : "Upload & Validate"}
      </button>

      {status && (
        <div className="border-t border-slate-800 pt-3">
          <p className="text-xs text-green-400 font-bold mb-2">✓ Loaded Successfully</p>
          {successContent}
        </div>
      )}
    </div>
  );
}

function InfoRow({ label, value, highlight, mono }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-slate-600">{label}</span>
      <span
        className={`${highlight ? "text-cyan-400 font-bold" : "text-slate-400"} ${
          mono ? "font-mono text-xs" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function Spinner() {
  return (
    <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
  );
}
