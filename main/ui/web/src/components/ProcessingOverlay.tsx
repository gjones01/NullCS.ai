import { motion } from "framer-motion";

type ProcessingOverlayProps = {
  title: string;
  subtitle: string;
  steps: string[];
  activeStep: number;
  logs: string;
  reducedMotion: boolean;
};

export function ProcessingOverlay({ title, subtitle, steps, activeStep, logs, reducedMotion }: ProcessingOverlayProps) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 backdrop-blur-xl">
      <div className="glass-panel mx-4 w-full max-w-3xl p-6">
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        <p className="mt-1 text-xs uppercase tracking-[0.14em] text-slate-300">{subtitle}</p>

        <div className="mt-5 space-y-2">
          {steps.map((step, idx) => {
            const done = idx < activeStep;
            const current = idx === activeStep;
            return (
              <div key={step} className="flex items-center gap-3">
                <div className={`h-2 w-2 rounded-full ${done || current ? "bg-cyan-200" : "bg-slate-600"}`} />
                <div className={`text-sm ${done || current ? "text-slate-100" : "text-slate-400"}`}>{step}</div>
                <div className="ml-auto text-[11px] uppercase tracking-[0.12em] text-slate-400">{done ? "done" : current ? "active" : "queued"}</div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 flex items-center gap-2">
          {[0, 1, 2, 3].map((i) => (
            <motion.div
              key={i}
              className="h-1.5 w-10 rounded-full bg-cyan-200/70"
              animate={reducedMotion ? { opacity: 0.55 } : { opacity: [0.25, 1, 0.25], scaleX: [0.92, 1, 0.92] }}
              transition={{ duration: 1.1, delay: i * 0.14, repeat: Infinity, ease: "easeInOut" }}
            />
          ))}
        </div>

        <pre className="mt-5 max-h-48 overflow-auto rounded-lg border border-white/10 bg-black/40 p-3 text-[11px] text-slate-300">{logs || "Waiting for logs..."}</pre>
      </div>
    </div>
  );
}
