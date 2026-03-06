import { useMemo, useState } from "react";
import type { EvidenceTable } from "../api";

const preferredColumns = ["round_num", "kill_tick", "victim_name", "weapon", "rt_ticks", "is_prefire", "is_thrusmoke", "headshot"];

type SortState = { column: string; dir: "asc" | "desc" };

const evidenceLabelMap: Record<string, string> = {
  "evidence_fast_rt.csv": "Fast Reaction Time",
  "evidence_fast_rt_streak.csv": "Fast RT Streaks",
  "evidence_prefire.csv": "Prefires",
  "evidence_prefire_streak.csv": "Prefire Streaks",
  "evidence_thrusmoke.csv": "Through-Smoke Kills",
  "evidence_headshot_streak.csv": "Headshot Streaks",
  "evidence_long_range_fast_rt_4.csv": "Long-Range Fast RT",
};

function safeText(v: unknown): string {
  if (v === null || v === undefined) return "";
  return String(v);
}

function evidenceLabel(name: string): string {
  if (evidenceLabelMap[name]) return evidenceLabelMap[name];
  return name
    .replace(/\.csv$/i, "")
    .replace(/^evidence_/i, "")
    .split("_")
    .map((s) => (s ? s[0].toUpperCase() + s.slice(1) : s))
    .join(" ");
}

export function EvidenceTabs({ tables }: { tables: Record<string, EvidenceTable> }) {
  const names = useMemo(() => Object.keys(tables), [tables]);
  const [active, setActive] = useState<string>("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortState | null>(null);

  const activeName = active || names[0] || "";
  const table = activeName ? tables[activeName] : null;

  const displayColumns = useMemo(() => {
    if (!table) return [] as string[];
    const available = new Set(table.columns);
    const ordered = preferredColumns.filter((c) => available.has(c));
    const rest = table.columns.filter((c) => !ordered.includes(c));
    return [...ordered, ...rest];
  }, [table]);

  const filteredRows = useMemo(() => {
    if (!table) return [] as Record<string, unknown>[];
    const base = table.rows.filter((row) => {
      if (!search.trim()) return true;
      const hay = Object.values(row).map(safeText).join(" ").toLowerCase();
      return hay.includes(search.toLowerCase());
    });
    if (!sort) return base;

    const sorted = [...base];
    sorted.sort((a, b) => {
      const av = safeText(a[sort.column]);
      const bv = safeText(b[sort.column]);
      const an = Number(av);
      const bn = Number(bv);
      const numeric = !Number.isNaN(an) && !Number.isNaN(bn);
      if (numeric) return sort.dir === "asc" ? an - bn : bn - an;
      return sort.dir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    return sorted;
  }, [table, search, sort]);

  const onSort = (column: string) => {
    setSort((prev) => {
      if (!prev || prev.column !== column) return { column, dir: "asc" };
      if (prev.dir === "asc") return { column, dir: "desc" };
      return null;
    });
  };

  const copyRow = async (row: Record<string, unknown>) => {
    const payload = {
      round_num: safeText(row.round_num),
      kill_tick: safeText(row.kill_tick),
      victim: safeText(row.victim_name ?? row.victim_steamid),
      weapon: safeText(row.weapon),
      rt_ticks: safeText(row.rt_ticks),
      flags: ["is_prefire", "is_thrusmoke", "headshot"].map((k) => `${k}:${safeText(row[k])}`).join(" | "),
    };
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
  };

  return (
    <section className="glass-panel p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm uppercase tracking-[0.16em] text-slate-300">Evidence Tables</h2>
        <input
          className="rounded-lg border border-white/10 bg-black/40 p-2 text-xs"
          placeholder="Search evidence"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {names.length === 0 && <div className="rounded-lg border border-white/10 bg-black/30 p-3 text-sm text-slate-300">No evidence files found.</div>}

      {names.length > 0 && (
        <>
          <div className="mb-3 flex flex-wrap gap-2">
            {names.map((name) => (
              <button
                key={name}
                onClick={() => setActive(name)}
                className={`rounded-full border px-3 py-1 text-xs ${activeName === name ? "border-cyan-200/50 bg-cyan-400/20 text-cyan-100" : "border-white/15 bg-black/30 text-slate-300"}`}
              >
                {evidenceLabel(name)}
              </button>
            ))}
          </div>

          {table && (
            <div className="overflow-auto rounded-xl border border-white/10">
              <table className="min-w-full text-xs">
                <thead className="bg-white/5 text-left uppercase tracking-[0.12em] text-slate-300">
                  <tr>
                    {displayColumns.map((col) => (
                      <th key={col} className="cursor-pointer px-2 py-2" onClick={() => onSort(col)}>
                        {col}
                        {sort?.column === col ? (sort.dir === "asc" ? " ^" : " v") : ""}
                      </th>
                    ))}
                    <th className="px-2 py-2">copy</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row, idx) => (
                    <tr key={`${idx}-${safeText(row.kill_tick)}`} className="border-t border-white/10">
                      {displayColumns.map((col) => (
                        <td key={col} className="px-2 py-1.5 text-slate-200">
                          {safeText(row[col])}
                        </td>
                      ))}
                      <td className="px-2 py-1.5">
                        <button className="btn-secondary h-7 px-2 text-[11px]" onClick={() => copyRow(row)}>
                          Copy
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
