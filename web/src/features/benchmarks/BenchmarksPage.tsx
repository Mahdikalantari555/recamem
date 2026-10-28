import * as React from "react";

export function BenchmarksPage() {
  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold">Benchmarks</h2>
      <p className="text-sm text-text-secondary">
        LoCoMo / LongMemEval / DMR results appear here once a benchmark run is
        recorded. No fabricated metrics are shown.
      </p>
      <p className="text-xs text-text-muted">Status: unavailable — run `recamem bench` to populate.</p>
    </section>
  );
}
