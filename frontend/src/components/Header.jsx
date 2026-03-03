import { useSession } from "../hooks/useSession";

export default function Header() {
  const { sessionId, modelInfo, dataInfo } = useSession();

  return (
    <header className="bg-slate-900 border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 py-5">
        <div className="flex items-start justify-between flex-wrap gap-4">
          {/* Wordmark */}
          <div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-cyan-400 rounded-sm flex items-center justify-center">
                <span className="text-slate-950 font-black text-sm">XAI</span>
              </div>
              <div>
                <h1 className="text-xl font-black tracking-tight text-white">
                  ExplainAI
                </h1>
                <p className="text-xs text-slate-500 tracking-widest uppercase">
                  Model Transparency Visualizer
                </p>
              </div>
            </div>
            <p className="mt-2 text-xs text-slate-600 max-w-md">
              SHAP · LIME · Feature Importance · Interpretable Machine Learning
            </p>
          </div>

          {/* Session Status Indicators */}
          <div className="flex flex-col gap-1.5 text-xs font-mono">
            <StatusBadge
              label="Model"
              value={modelInfo?.model_type || "—"}
              active={!!modelInfo}
            />
            <StatusBadge
              label="Dataset"
              value={dataInfo ? `${dataInfo.n_rows}×${dataInfo.n_features}` : "—"}
              active={!!dataInfo}
            />
            {sessionId && (
              <div className="text-slate-700 text-right">
                session: {sessionId.slice(0, 8)}…
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

function StatusBadge({ label, value, active }) {
  return (
    <div className="flex items-center gap-2 justify-end">
      <span className="text-slate-600 uppercase tracking-widest">{label}</span>
      <span
        className={`px-2 py-0.5 rounded text-xs font-bold ${
          active
            ? "bg-cyan-400/10 text-cyan-400 border border-cyan-400/30"
            : "bg-slate-800 text-slate-600 border border-slate-700"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
