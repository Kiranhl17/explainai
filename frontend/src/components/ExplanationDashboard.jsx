import { useState } from "react";
import Plot from "react-plotly.js";

export default function ExplanationDashboard({ data, onTabChange }) {
  const [activeSection, setActiveSection] = useState("shap-global");

  if (!data) {
    return (
      <div className="text-center py-24 text-slate-600">
        <p className="text-4xl mb-4">◈</p>
        <p className="text-sm font-bold uppercase tracking-widest">No explanation data</p>
        <p className="text-xs mt-2">Upload a model and dataset, then run analysis</p>
        <button
          onClick={() => onTabChange("upload")}
          className="mt-6 px-6 py-2 bg-slate-800 text-slate-300 text-xs font-bold rounded border border-slate-700 hover:border-cyan-400 hover:text-cyan-400 transition-colors"
        >
          Go to Upload
        </button>
      </div>
    );
  }

  const sections = [
    { id: "shap-global", label: "SHAP Global" },
    { id: "shap-local", label: "SHAP Local Force" },
    { id: "lime", label: "LIME Local" },
    { id: "feature-importance", label: "Feature Importance" },
  ];

  return (
    <div className="space-y-6">
      <div className="border-l-4 border-cyan-400 pl-4">
        <h2 className="text-lg font-black text-white tracking-tight">
          Explainability Dashboard
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          SHAP (global + local) · LIME (local) · Feature Importance
          &nbsp;·&nbsp; Instance #{data.instance_index}
        </p>
      </div>

      {/* Section Tabs */}
      <div className="flex flex-wrap gap-2">
        {sections.map((s) => (
          <button
            key={s.id}
            onClick={() => setActiveSection(s.id)}
            className={`px-4 py-2 text-xs font-bold uppercase tracking-widest rounded border transition-colors ${
              activeSection === s.id
                ? "bg-cyan-400/10 border-cyan-400 text-cyan-400"
                : "bg-slate-900 border-slate-700 text-slate-500 hover:text-slate-300 hover:border-slate-500"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* SHAP Global Section */}
      {activeSection === "shap-global" && (
        <div className="space-y-6">
          <SectionHeader
            title="SHAP Global Interpretability"
            subtitle={`Explainer: ${data.shap_explainer_type}`}
            theory={
              "SHAP decomposes f(x) = φ₀ + Σφᵢ using Shapley axioms. " +
              "The beeswarm plot shows the distribution of SHAP values across ALL instances — " +
              "dot colour encodes feature value (blue=low, red=high). " +
              "The bar chart shows mean |SHAP|, the preferred global importance metric."
            }
          />
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <PlotCard
              title="SHAP Summary (Beeswarm) Plot"
              subtitle="Global — all instances aggregated"
              image={data.shap_summary_plot}
            />
            <PlotCard
              title="Mean |SHAP| Feature Importance"
              subtitle="Global — averaged over dataset"
              image={data.shap_bar_plot}
            />
          </div>
        </div>
      )}

      {/* SHAP Local Force Plot */}
      {activeSection === "shap-local" && (
        <div className="space-y-6">
          <SectionHeader
            title="SHAP Local Interpretability — Force Plot"
            subtitle={`Instance #${data.instance_index} · Base value: ${data.shap_force_data?.base_value?.toFixed(4)}`}
            theory={
              "The force plot shows how each feature PUSHES the prediction away from " +
              "the baseline φ₀ = E[f(X)] for this SPECIFIC instance. " +
              "Red bars increase the prediction; blue bars decrease it. " +
              "Σφᵢ + φ₀ = f(x) — the decomposition is exact."
            }
          />
          <SHAPForcePlot data={data.shap_force_data} />
        </div>
      )}

      {/* LIME Local */}
      {activeSection === "lime" && (
        <div className="space-y-6">
          <SectionHeader
            title="LIME Local Explanation"
            subtitle={`Instance #${data.instance_index} · ${data.lime_explanation?.explanation?.length || 0} features shown`}
            theory={
              "LIME perturbs instance #" +
              data.instance_index +
              " and trains a weighted linear surrogate in its neighbourhood. " +
              "Positive weights push toward the predicted class; negative push away. " +
              "This explanation is LOCAL — applying it to other instances is not valid."
            }
          />
          <LIMEPanel data={data.lime_explanation} plotImage={data.lime_plot} />
        </div>
      )}

      {/* Feature Importance */}
      {activeSection === "feature-importance" && (
        <div className="space-y-6">
          <SectionHeader
            title="Native Feature Importances"
            subtitle={`Method: ${data.feature_importance_method}`}
            theory={
              "Tree-based MDI (Mean Decrease Impurity) importances are fast but biased " +
              "toward high-cardinality continuous features. Mean |SHAP| (shown in Global tab) " +
              "is theoretically superior. Both are provided for comparison."
            }
          />
          <NativeImportancePlot data={data.feature_importances} />
        </div>
      )}
    </div>
  );
}

// ---- Sub-components ----

function SectionHeader({ title, subtitle, theory }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
      <h3 className="text-sm font-black text-white mb-1">{title}</h3>
      <p className="text-xs text-cyan-400 font-mono mb-3">{subtitle}</p>
      <p className="text-xs text-slate-500 leading-relaxed border-l-2 border-slate-700 pl-3">
        <span className="text-slate-400 font-bold">Theory: </span>
        {theory}
      </p>
    </div>
  );
}

function PlotCard({ title, subtitle, image }) {
  if (!image) return null;
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-800">
        <p className="text-xs font-bold text-white">{title}</p>
        <p className="text-xs text-slate-600">{subtitle}</p>
      </div>
      <div className="p-3">
        <img
          src={`data:image/png;base64,${image}`}
          alt={title}
          className="w-full rounded"
          style={{ imageRendering: "crisp-edges" }}
        />
      </div>
    </div>
  );
}

function SHAPForcePlot({ data }) {
  if (!data) return null;

  const { contributions, base_value, prediction, interpretation } = data;

  const positive = contributions.filter((c) => c.shap_value > 0);
  const negative = contributions.filter((c) => c.shap_value < 0);

  // Build Plotly waterfall chart data
  const labels = contributions.map(
    (c) => `${c.feature}=${typeof c.value === "number" ? c.value.toFixed(3) : c.value}`
  );
  const values = contributions.map((c) => c.shap_value);
  const colors = contributions.map((c) =>
    c.shap_value > 0 ? "#ef4444" : "#3b82f6"
  );

  const plotData = [
    {
      type: "bar",
      orientation: "h",
      x: values,
      y: labels,
      marker: { color: colors },
      hovertemplate:
        "<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(15,23,42,0.5)",
    font: { color: "#94a3b8", family: "JetBrains Mono, monospace", size: 11 },
    margin: { l: 200, r: 40, t: 20, b: 50 },
    xaxis: {
      title: "SHAP Value (φᵢ)",
      gridcolor: "#1e293b",
      zerolinecolor: "#334155",
    },
    yaxis: { gridcolor: "#1e293b", automargin: true },
    height: Math.max(350, contributions.length * 35),
    shapes: [
      {
        type: "line",
        x0: 0, x1: 0,
        y0: -0.5, y1: contributions.length - 0.5,
        line: { color: "#475569", width: 1, dash: "dot" },
      },
    ],
  };

  return (
    <div className="space-y-4">
      {/* Summary Banner */}
      <div className="grid grid-cols-3 gap-3">
        <MetricChip label="Base Value (E[f(X)])" value={base_value?.toFixed(4)} />
        <MetricChip label={`Prediction f(x₍${data.instance_index}₎)`} value={prediction?.toFixed(4)} accent />
        <MetricChip
          label="Net SHAP Contribution"
          value={(prediction - base_value).toFixed(4)}
        />
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-800">
          <p className="text-xs font-bold text-white">
            Feature Contributions — Instance #{data.instance_index}
          </p>
          <p className="text-xs text-slate-600">
            Red = increases prediction | Blue = decreases prediction
          </p>
        </div>
        <Plot
          data={plotData}
          layout={layout}
          config={{ displayModeBar: true, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </div>

      <p className="text-xs text-slate-600 bg-slate-900 border border-slate-800 rounded p-3 leading-relaxed">
        {interpretation}
      </p>
    </div>
  );
}

function LIMEPanel({ data, plotImage }) {
  if (!data) return null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <MetricChip
          label="Local Prediction (surrogate)"
          value={data.local_prediction?.toFixed(4) ?? "N/A"}
          accent
        />
        <MetricChip
          label="Surrogate Intercept"
          value={data.intercept?.toFixed(4) ?? "N/A"}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* LIME Plot */}
        {plotImage && (
          <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-800">
              <p className="text-xs font-bold text-white">
                LIME Feature Weights — Surrogate Coefficients
              </p>
            </div>
            <div className="p-3">
              <img
                src={`data:image/png;base64,${plotImage}`}
                alt="LIME explanation"
                className="w-full rounded"
              />
            </div>
          </div>
        )}

        {/* Feature Weight Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800">
            <p className="text-xs font-bold text-white">LIME Rule Table</p>
          </div>
          <div className="overflow-auto max-h-96">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left px-4 py-2 text-slate-500 font-bold">Condition</th>
                  <th className="text-right px-4 py-2 text-slate-500 font-bold">Weight</th>
                  <th className="text-right px-4 py-2 text-slate-500 font-bold">Effect</th>
                </tr>
              </thead>
              <tbody>
                {data.explanation?.map((item, i) => (
                  <tr
                    key={i}
                    className="border-b border-slate-800/50 hover:bg-slate-800/30"
                  >
                    <td className="px-4 py-2 text-slate-400 font-mono">
                      {item.feature_condition}
                    </td>
                    <td
                      className={`px-4 py-2 text-right font-mono font-bold ${
                        item.weight > 0 ? "text-red-400" : "text-blue-400"
                      }`}
                    >
                      {item.weight.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-bold ${
                          item.direction === "positive"
                            ? "bg-red-900/40 text-red-400"
                            : "bg-blue-900/40 text-blue-400"
                        }`}
                      >
                        {item.direction === "positive" ? "↑" : "↓"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-600 bg-slate-900 border border-slate-800 rounded p-3 leading-relaxed">
        {data.interpretation}
      </p>
    </div>
  );
}

function NativeImportancePlot({ data }) {
  if (!data || data.length === 0) return (
    <p className="text-slate-600 text-sm">No feature importances available for this model type.</p>
  );

  const top20 = data.slice(0, 20);
  const plotData = [
    {
      type: "bar",
      orientation: "h",
      x: [...top20.map((d) => d.importance)].reverse(),
      y: [...top20.map((d) => d.feature)].reverse(),
      marker: {
        color: top20.map((_, i) =>
          `rgba(34,211,238,${1 - (i / top20.length) * 0.6})`
        ).reverse(),
      },
      hovertemplate: "<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(15,23,42,0.5)",
    font: { color: "#94a3b8", family: "JetBrains Mono, monospace", size: 11 },
    margin: { l: 180, r: 40, t: 20, b: 50 },
    xaxis: {
      title: "Importance Score (MDI / |coeff|)",
      gridcolor: "#1e293b",
    },
    yaxis: { gridcolor: "#1e293b", automargin: true },
    height: Math.max(300, top20.length * 30),
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-800">
        <p className="text-xs font-bold text-white">Top Features — Native Importances</p>
        <p className="text-xs text-slate-600">
          Mean Decrease Impurity (MDI). Compare with SHAP Mean |φ| for rigorous ranking.
        </p>
      </div>
      <Plot
        data={plotData}
        layout={layout}
        config={{ displayModeBar: true, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

function MetricChip({ label, value, accent }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-center">
      <p className="text-xs text-slate-600 mb-1">{label}</p>
      <p
        className={`text-lg font-black font-mono ${
          accent ? "text-cyan-400" : "text-white"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
