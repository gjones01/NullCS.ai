import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import {
  type DemoStatus,
  type EvidenceTable,
  type PlayerRow,
  type Reason,
  explainPlayer,
  getEvidenceTable,
  getPlayers,
  getReasons,
  getReportFiles,
  getStatus,
  runDemo,
  uploadDemo,
} from "./api";
import { EvidenceTabs } from "./components/EvidenceTabs";
import { ForensicBackground } from "./components/ForensicBackground";
import { ProcessingOverlay } from "./components/ProcessingOverlay";
import { ReasonsPanel, SeverityLegend } from "./components/ReasonsPanel";

type Stage = "upload" | "processing" | "results" | "report";

const PIPELINE_STEPS = ["Uploading", "Parsing", "Feature Build", "Model", "Explanation"];

function detectLowPowerDefault(): boolean {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const cores = navigator.hardwareConcurrency || 8;
  const mem = nav.deviceMemory || 8;
  return cores <= 4 || mem <= 4;
}

export default function App() {
  const [stage, setStage] = useState<Stage>("upload");
  const [demoId, setDemoId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<DemoStatus>({ demo_id: "", state: "queued", logs_tail: "", error: "" });
  const [players, setPlayers] = useState<PlayerRow[]>([]);
  const [query, setQuery] = useState("");
  const [selectedSteamid, setSelectedSteamid] = useState("");
  const [reasons, setReasons] = useState<Reason[]>([]);
  const [evidenceTables, setEvidenceTables] = useState<Record<string, EvidenceTable>>({});
  const [isExplaining, setIsExplaining] = useState(false);
  const [error, setError] = useState("");
  const [reducedMotion, setReducedMotion] = useState<boolean>(() => {
    const stored = window.localStorage.getItem("clarity_reduced_motion");
    if (stored === "1") return true;
    if (stored === "0") return false;
    return detectLowPowerDefault();
  });
  const [mouseGlow, setMouseGlow] = useState({ x: window.innerWidth / 2, y: window.innerHeight / 2 });

  useEffect(() => {
    window.localStorage.setItem("clarity_reduced_motion", reducedMotion ? "1" : "0");
    document.documentElement.dataset.motion = reducedMotion ? "reduced" : "full";
  }, [reducedMotion]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => setMouseGlow({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  useEffect(() => {
    if (stage !== "processing" || !demoId) return;
    const timer = setInterval(async () => {
      try {
        const s = await getStatus(demoId);
        setStatus(s);
        if (s.state === "done") {
          const p = await getPlayers(demoId);
          setPlayers(p.players || []);
          setStage("results");
        }
        if (s.state === "error") {
          setError(s.error || "Pipeline error");
        }
      } catch (e: unknown) {
        setError(String(e));
      }
    }, 1200);
    return () => clearInterval(timer);
  }, [stage, demoId]);

  const filtered = useMemo(() => {
    return players
      .filter((p) => p.attacker_name.toLowerCase().includes(query.toLowerCase()) || p.steamid.includes(query))
      .sort((a, b) => a.attacker_name.localeCompare(b.attacker_name));
  }, [players, query]);

  const onUploadAndRun = async () => {
    try {
      if (!file) return;
      setError("");
      setReasons([]);
      setEvidenceTables({});
      const u = await uploadDemo(file, demoId || undefined);
      const id = u.demo_id;
      setDemoId(id);
      setStatus({
        demo_id: id,
        state: "queued",
        logs_tail: "",
        error: "",
        stage_index: 0,
        stage: "Uploading",
        steps: PIPELINE_STEPS,
      });
      await runDemo(id);
      setStage("processing");
    } catch (e: unknown) {
      setError(String(e));
    }
  };

  const onAnalyze = async (steamid: string) => {
    try {
      setError("");
      setSelectedSteamid(steamid);
      setIsExplaining(true);
      await explainPlayer(demoId, steamid);
      const files = await getReportFiles(demoId, steamid);
      const r = await getReasons(demoId, steamid);
      const tableEntries = await Promise.all(
        files.evidence_files.map(async (f) => [f, await getEvidenceTable(demoId, steamid, f, 500)] as const)
      );
      const tables = Object.fromEntries(tableEntries);
      setReasons(r.reasons || []);
      setEvidenceTables(tables);
      setStage("report");
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setIsExplaining(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-app text-slate-100">
      <ForensicBackground reducedMotion={reducedMotion} />
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-10"
        style={{
          background: `radial-gradient(480px circle at ${mouseGlow.x}px ${mouseGlow.y}px, rgba(180, 246, 255, 0.1), transparent 70%)`,
        }}
      />

      <div className="relative z-20 mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8">
        <header className="glass-panel flex flex-wrap items-center justify-between gap-3 p-4 md:p-5">
          <div>
            <motion.h1 initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-semibold tracking-[0.24em] text-white md:text-3xl">
              NULLCS
            </motion.h1>
            <p className="mt-1 text-xs tracking-[0.08em] text-slate-300">See the Unseen: AI-Powered Integrity in Every Match</p>
          </div>
          <label className="glass-subpanel flex items-center gap-2 px-3 py-2 text-xs uppercase tracking-[0.14em]">
            <input type="checkbox" checked={reducedMotion} onChange={(e) => setReducedMotion(e.target.checked)} />
            Reduced Motion / Low Power
          </label>
        </header>

        {stage === "upload" && (
          <section className="flex min-h-[66vh] flex-col items-center justify-center gap-5 md:-translate-y-8">
            <div className="glass-panel w-full max-w-3xl p-6">
              <p className="mb-3 text-sm text-slate-200">Upload a .dem file to analyze players and review evidence-backed behavior signals.</p>
              <input
                className="mb-3 w-full rounded-lg border border-white/10 bg-black/40 p-2.5 text-sm"
                placeholder="Optional demo ID (e.g., TEST_...)"
                value={demoId}
                onChange={(e) => setDemoId(e.target.value)}
              />
              <input type="file" accept=".dem" className="mb-4 block w-full text-sm" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              <button className="btn-primary" onClick={onUploadAndRun} disabled={!file}>
                Analyze Demo
              </button>
            </div>

            <article className="glass-panel w-full max-w-3xl p-5">
              <h2 className="mb-2 text-sm uppercase tracking-[0.16em] text-slate-300">About NullCS</h2>
              <p className="text-sm leading-relaxed text-slate-200">
                NullCS is a passion project and proof-of-concept focused on surfacing suspicious gameplay patterns from demo data.
              </p>
              <p className="mt-2 text-sm leading-relaxed text-slate-200">
                Its outputs are research signals, not definitive proof of cheating. Treat results as a starting point for review, not a final verdict.
              </p>
              <p className="mt-2 text-sm leading-relaxed text-slate-200">
                The model is still being trained and will improve as more high-quality data is added.
              </p>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-300">
                <li>Use results to prioritize manual review.</li>
                <li>Combine model signals with demo context before conclusions.</li>
                <li>Avoid harassment, brigading, and witch-hunts.</li>
                <li>Focus on fair play and responsible reporting.</li>
              </ul>
            </article>
          </section>
        )}

        {stage === "results" && (
          <section className="glass-panel p-6">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-300">Demo {demoId}</div>
              <input
                className="rounded-lg border border-white/10 bg-black/40 p-2 text-sm"
                placeholder="Search player or steamid"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <div className="overflow-hidden rounded-xl border border-white/10">
              <table className="w-full text-sm">
                <thead className="bg-white/5 text-left text-xs uppercase tracking-[0.12em] text-slate-300">
                  <tr>
                    <th className="px-3 py-2">Player</th>
                    <th className="px-3 py-2">SteamID</th>
                    <th className="px-3 py-2">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((p) => (
                    <tr key={p.steamid} className="border-t border-white/10">
                      <td className="px-3 py-2 font-medium text-white">{p.attacker_name || "(unknown)"}</td>
                      <td className="px-3 py-2 text-xs text-slate-300">{p.steamid}</td>
                      <td className="px-3 py-2">
                        <button className="btn-primary h-8 px-3 text-xs" onClick={() => onAnalyze(p.steamid)}>
                          Analyze
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {stage === "report" && (
          <section className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(280px,340px)]">
            <div className="flex flex-col gap-4">
              <div className="glass-panel flex items-center justify-between p-4">
                <div className="text-xs uppercase tracking-[0.14em] text-slate-300">Demo {demoId} | SteamID {selectedSteamid}</div>
                <button className="btn-secondary" onClick={() => setStage("results")}>
                  Back to Players
                </button>
              </div>
              <ReasonsPanel reasons={reasons} />
              <EvidenceTabs tables={evidenceTables} />
            </div>
            <SeverityLegend />
          </section>
        )}

        {error && <div className="glass-panel border-red-400/40 bg-red-950/30 p-3 text-sm text-red-100">{error}</div>}
      </div>

      {(stage === "processing" || isExplaining) && (
        <ProcessingOverlay
          reducedMotion={reducedMotion}
          title={isExplaining ? "Generating Explainability" : "Processing Demo"}
          subtitle={isExplaining ? "Explanation" : status.stage || "Uploading"}
          logs={status.logs_tail}
          steps={PIPELINE_STEPS}
          activeStep={isExplaining ? 4 : status.stage_index ?? 0}
        />
      )}
    </div>
  );
}
