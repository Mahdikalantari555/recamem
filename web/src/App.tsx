import * as React from "react";

import { AppShell } from "./components/layout/AppShell";
import type { NavItem } from "./components/ui/sidebar";
import { ArchitecturePage } from "./features/architecture/ArchitecturePage";
import { BenchmarksPage } from "./features/benchmarks/BenchmarksPage";
import { ExperimentsPage } from "./features/experiments/ExperimentsPage";
import { MemoryPage } from "./features/memory/MemoryPage";
import { PlaygroundPage } from "./features/playground/PlaygroundPage";
import { SettingsPage } from "./features/settings/SettingsPage";
import { TrainingPage } from "./features/training/TrainingPage";
import { api } from "./lib/api/client";
import type { Health } from "./lib/schemas";

const NAV: NavItem[] = [
  { id: "playground", label: "Playground", hint: "Prompt, answer, decision inspector", available: true },
  { id: "experiments", label: "Experiments", hint: "Recorded runs and jobs", available: true },
  { id: "memory", label: "Memory Explorer", hint: "Browse stored facts", available: false },
  { id: "benchmarks", label: "Benchmarks", hint: "LoCoMo / LongMemEval / DMR", available: false },
  { id: "training", label: "Training", hint: "Continual learning jobs", available: false },
  { id: "architecture", label: "Architecture Lab", hint: "Inspect the memory layout", available: false },
  { id: "settings", label: "Settings", hint: "Anchor mode and storage", available: false },
];

export default function App() {
  const [active, setActive] = React.useState("playground");
  const [health, setHealth] = React.useState<Health | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = React.useCallback(async () => {
    try {
      setHealth(await api.health());
      setError(null);
    } catch (exc) {
      setHealth(null);
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, []);

  React.useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 10_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  return (
    <AppShell items={NAV} active={active} onSelect={setActive} health={health} error={error}>
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {active === "experiments" ? (
          <ExperimentsPage />
        ) : active === "memory" ? (
          <MemoryPage />
        ) : active === "benchmarks" ? (
          <BenchmarksPage />
        ) : active === "training" ? (
          <TrainingPage />
        ) : active === "architecture" ? (
          <ArchitecturePage />
        ) : active === "settings" ? (
          <SettingsPage />
        ) : (
          <PlaygroundPage onActivity={() => void refresh()} />
        )}
      </div>
    </AppShell>
  );
}
