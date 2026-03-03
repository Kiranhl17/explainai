import { useState } from "react";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import ExplanationDashboard from "./components/ExplanationDashboard";
import MetricsPanel from "./components/MetricsPanel";
import Footer from "./components/Footer";
import { SessionProvider } from "./hooks/useSession";
import "./index.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [explanationData, setExplanationData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);

  return (
    <SessionProvider>
      <div className="min-h-screen bg-slate-950 text-slate-100 font-mono">
        <Header />

        {/* Navigation Tabs */}
        <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-4 flex gap-1 pt-2">
            {[
              { id: "upload", label: "01 · Upload", icon: "⬆" },
              { id: "explain", label: "02 · Explanations", icon: "◈" },
              { id: "metrics", label: "03 · Metrics", icon: "◎" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-5 py-3 text-xs font-bold tracking-widest uppercase transition-all border-b-2 ${
                  activeTab === tab.id
                    ? "border-cyan-400 text-cyan-400 bg-cyan-400/5"
                    : "border-transparent text-slate-500 hover:text-slate-300 hover:border-slate-600"
                }`}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>
        </nav>

        {/* Content */}
        <main className="max-w-7xl mx-auto px-4 py-8">
          {activeTab === "upload" && (
            <UploadPanel
              onExplainReady={(data) => {
                setExplanationData(data);
                setActiveTab("explain");
              }}
              onMetricsReady={(data) => {
                setMetricsData(data);
              }}
            />
          )}
          {activeTab === "explain" && (
            <ExplanationDashboard
              data={explanationData}
              onTabChange={setActiveTab}
            />
          )}
          {activeTab === "metrics" && (
            <MetricsPanel
              data={metricsData}
              onFetchMetrics={setMetricsData}
            />
          )}
        </main>

        <Footer />
      </div>
    </SessionProvider>
  );
}
