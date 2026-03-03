import { useState } from "react";
import Plot from "react-plotly.js";
import { useSession } from "../hooks/useSession";

export default function MetricsPanel({ data, onFetchMetrics }) {
  const { fetchMetrics, dataInfo } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFetch = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchMetrics();
      onFetchMetrics(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!data) {
    return (
      <div className="space-y-4">
        <div className="border-l-4 border-cyan-400 pl-4">
          <h2 className="text-lg font-black text-white">Performance Metrics</h2>
          <p className="text-xs text-slate-500 mt-1">
            Requires a labeled dataset (target column specified)
          </p>
        </div>
        {error && (
          <div className="bg-red-950/50 border border-red-800 text-red-400 rounded p-3 text-sm">
            {error}
          </div>
        )}
        <button
          onClick={handleFetch}
          disabled={loading || !dataInfo?.has_target}
          className="px-6 py-3 bg-cyan-400 text-slate-950 font-black text-xs tracking-widest uppercase rounded
                     hover:bg-cyan-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Computing…" : "◎ Compute Metrics"}
        </button>
        {!dataInfo?.has_target && (
          <p className="text-xs text-slate-600">
            Re-upload your dataset with a target_column parameter to enable metrics.
          </p>
        )}
      </div>
    );
  }

  const isClassifier = data.task_type === "classification";

  return (
    <div className="space-y-6">
      <div className="border-l-4 border-cyan-400 pl-4">
        <h2 className="text-lg font-black text-white">Performance Metrics</h2>
        <p className="text-xs text-slate-500 mt-1">
          Task: {data.task_type} · n_samples: {data.n_samples}
        </p>
      </div>

      {isClassifier ? (
        <ClassificationMetrics data={data} />
      ) : (
        <RegressionMetrics data={data} />
      )}

      {/* Notes Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <p className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-widest">
          Metric Interpretations
        </p>
        <div className="space-y-2">
          {Object.entries(data.notes || {}).map(([key, note]) => (
            <div key={key} className="flex gap-3 text-xs">
              <span className="text-cyan-400 font-mono font-bold min-w-24 uppercase">
                {key.replace(/_/g, " ")}
              </span>
              <span className="text-slate-500">{note}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ClassificationMetrics({ data }) {
  const metrics = [
    { label: "Accuracy", value: data.accuracy, note: "All classes" },
    { label: "Precision (W)", value: data.precision_weighted, note: "Weighted" },
    { label: "Recall (W)", value: data.recall_weighted, note: "Weighted" },
    { label: "F1 Score (W)", value: data.f1_weighted, note: "Weighted", accent: true },
    { label: "F1 Score (M)", value: data.f1_macro, note: "Macro" },
    { label: "ROC-AUC", value: data.roc_auc, note: data.roc_auc ? "OvR" : "N/A" },
  ];

  return (
    <div className="space-y-6">
      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        {metrics.map((m) => (
          <KPICard key={m.label} {...m} />
        ))}
      </div>

      {/* Confusion Matrix */}
      {data.confusion_matrix && (
        <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800">
            <p className="text-xs font-bold text-white">Confusion Matrix</p>
            <p className="text-xs text-slate-600">
              Rows = actual class | Cols = predicted class
            </p>
          </div>
          <ConfusionMatrixPlot matrix={data.confusion_matrix} />
        </div>
      )}

      {/* Dual F1 Comparison */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-800">
          <p className="text-xs font-bold text-white">Precision · Recall · F1 Comparison</p>
        </div>
        <PRF1BarChart data={data} />
      </div>
    </div>
  );
}

function RegressionMetrics({ data }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <KPICard label="R² Score" value={data.r2_score} accent />
      <KPICard label="MAE" value={data.mae} />
      <KPICard label="RMSE" value={data.rmse} />
      <KPICard label="MSE" value={data.mse} />
    </div>
  );
}

function KPICard({ label, value, note, accent }) {
  const display =
    value === null || value === undefined
      ? "—"
      : (value * 100).toFixed(1) + "%";

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-center">
      <p className="text-xs text-slate-600 mb-1 truncate">{label}</p>
      <p
        className={`text-xl font-black font-mono ${
          accent ? "text-cyan-400" : "text-white"
        }`}
      >
        {display}
      </p>
      {note && <p className="text-xs text-slate-700 mt-0.5">{note}</p>}
    </div>
  );
}

function ConfusionMatrixPlot({ matrix }) {
  const n = matrix.length;
  const labels = matrix.map((_, i) => `Class ${i}`);

  const plotData = [
    {
      type: "heatmap",
      z: matrix,
      x: labels,
      y: labels,
      colorscale: [
        [0, "#0f172a"],
        [0.5, "#164e63"],
        [1, "#22d3ee"],
      ],
      showscale: true,
      text: matrix.map((row) => row.map((v) => v.toString())),
      texttemplate: "%{text}",
      hovertemplate:
        "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#94a3b8", family: "JetBrains Mono, monospace", size: 11 },
    margin: { l: 80, r: 40, t: 20, b: 80 },
    xaxis: { title: "Predicted Class", gridcolor: "#1e293b" },
    yaxis: { title: "Actual Class", gridcolor: "#1e293b", autorange: "reversed" },
    height: Math.min(500, Math.max(300, n * 60 + 120)),
  };

  return (
    <Plot
      data={plotData}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

function PRF1BarChart({ data }) {
  const plotData = [
    {
      type: "bar",
      name: "Precision (W)",
      x: ["Precision", "Recall", "F1 (Weighted)", "F1 (Macro)"],
      y: [
        data.precision_weighted,
        data.recall_weighted,
        data.f1_weighted,
        data.f1_macro,
      ],
      marker: { color: ["#22d3ee", "#818cf8", "#34d399", "#fb923c"] },
      text: [
        data.precision_weighted,
        data.recall_weighted,
        data.f1_weighted,
        data.f1_macro,
      ].map((v) => (v !== null ? `${(v * 100).toFixed(1)}%` : "N/A")),
      textposition: "outside",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(15,23,42,0.5)",
    font: { color: "#94a3b8", family: "JetBrains Mono, monospace", size: 11 },
    margin: { l: 60, r: 40, t: 20, b: 60 },
    yaxis: {
      range: [0, 1.1],
      title: "Score",
      gridcolor: "#1e293b",
      tickformat: ".0%",
    },
    xaxis: { gridcolor: "#1e293b" },
    height: 300,
    showlegend: false,
  };

  return (
    <Plot
      data={plotData}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
