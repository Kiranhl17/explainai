export default function Footer() {
  return (
    <footer className="border-t border-slate-800 bg-slate-900/50 mt-16">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="flex flex-wrap gap-6 justify-between items-start text-xs text-slate-600">
          <div>
            <p className="text-slate-500 font-bold mb-1">ExplainAI — Model Transparency Visualizer</p>
            <p>Kiran H L</p>
          </div>
          <div className="space-y-1">
            <p className="text-slate-500 font-bold">References</p>
            <p>SHAP: Lundberg & Lee, NeurIPS 2017</p>
            <p>TreeSHAP: Lundberg et al., Nature MI 2020</p>
            <p>LIME: Ribeiro et al., KDD 2016</p>
          </div>
          <div className="space-y-1">
            <p className="text-slate-500 font-bold">Stack</p>
            <p>FastAPI · SHAP · LIME · scikit-learn</p>
            <p>React · Plotly.js · Tailwind CSS</p>
          </div>
          <div className="space-y-1">
            <p className="text-slate-500 font-bold">Future Extensions</p>
            <p>Counterfactual Explanations (DiCE)</p>
            <p>Fairness & Bias Detection (AIF360)</p>
            <p>Model Comparison Module</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
