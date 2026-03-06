const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export type PlayerRow = {
  steamid: string;
  attacker_name: string;
  proba_cheater_infer: number;
  risk?: number;
  confidence?: number;
  ci_low?: number | null;
  ci_high?: number | null;
  risk_band?: string;
  top_reasons?: Array<{ title: string; severity: string }>;
  features_summary: Record<string, number | null>;
};

export type ScoreTrace = {
  steamid: string;
  attacker_name: string;
  raw_proba: number;
  calibrated_proba?: number | null;
  risk_display_value: number;
  confidence_value: number;
  ci_p05?: number | null;
  ci_p95?: number | null;
  gating_rules?: Record<string, unknown>;
  evidence_counts?: Record<string, unknown>;
  high_tag_flags?: Record<string, boolean>;
  why_risk_low_despite_high_tags?: string | null;
  feature_row?: Record<string, number | null>;
  model_artifact_path?: string;
  model_version?: string;
  model_sha256?: string;
  feature_list_version?: string;
  feature_vector_hash?: string;
};

export type DemoStatus = {
  demo_id: string;
  state: string;
  logs_tail: string;
  error: string;
  stage_index?: number;
  stage?: string;
  steps?: string[];
};

export type Reason = {
  reason: string;
  severity: "low" | "medium" | "high" | "context" | string;
  summary: string;
  why_it_matters: string;
};

export type ReportFiles = {
  demo_id: string;
  steamid: string;
  report_dir: string;
  report_exists?: boolean;
  reasons_exists: boolean;
  top_row_exists: boolean;
  evidence_files: string[];
};

export type EvidenceTable = {
  demo_id: string;
  steamid: string;
  filename: string;
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(url, init);
  } catch (err) {
    throw new Error(
      `Network request failed. Check that backend is running on ${API_BASE} and CORS allows this frontend origin. Original error: ${String(err)}`
    );
  }
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<T>;
}

export async function uploadDemo(file: File, demoId?: string) {
  const fd = new FormData();
  fd.append("file", file);
  if (demoId) fd.append("demo_id", demoId);
  return fetchJson<{ demo_id: string }>(`${API_BASE}/upload-demo`, { method: "POST", body: fd });
}

export async function runDemo(demoId: string) {
  return fetchJson<{ demo_id: string; state: string }>(`${API_BASE}/demo/${demoId}/run`, { method: "POST" });
}

export async function getStatus(demoId: string) {
  return fetchJson<DemoStatus>(`${API_BASE}/demo/${demoId}/status`);
}

export async function getPlayers(demoId: string): Promise<{ demo_id: string; players: PlayerRow[] }> {
  return fetchJson<{ demo_id: string; players: PlayerRow[] }>(`${API_BASE}/demo/${demoId}/players`);
}

export async function getPlayersDebug(demoId: string): Promise<{ demo_id: string; players: PlayerRow[]; debug?: { enabled: boolean; path: string; trace: { players?: ScoreTrace[] } } }> {
  return fetchJson<{ demo_id: string; players: PlayerRow[]; debug?: { enabled: boolean; path: string; trace: { players?: ScoreTrace[] } } }>(
    `${API_BASE}/demo/${demoId}/players?debug=1`
  );
}

export async function getPlayerScoreTrace(demoId: string, steamid: string) {
  return fetchJson<{ demo_id: string; steamid: string; debug_path: string; trace: ScoreTrace }>(
    `${API_BASE}/demo/${demoId}/player/${steamid}/score-trace`
  );
}

export async function explainPlayer(demoId: string, steamid: string) {
  return fetchJson<{ demo_id: string; steamid: string; evidence_files: string[] }>(
    `${API_BASE}/demo/${demoId}/player/${steamid}/explain`,
    { method: "POST" }
  );
}

export async function getReportFiles(demoId: string, steamid: string) {
  return fetchJson<ReportFiles>(`${API_BASE}/demo/${demoId}/player/${steamid}/report/files`);
}

export async function getReasons(demoId: string, steamid: string) {
  return fetchJson<{ demo_id: string; steamid: string; reasons: Reason[] }>(
    `${API_BASE}/demo/${demoId}/player/${steamid}/report/reasons`
  );
}

export async function getEvidenceTable(demoId: string, steamid: string, filename: string, limit = 500) {
  const safeName = encodeURIComponent(filename);
  return fetchJson<EvidenceTable>(`${API_BASE}/demo/${demoId}/player/${steamid}/report/evidence/${safeName}?limit=${limit}`);
}
