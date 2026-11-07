import * as React from "react";

import { Button } from "../../components/ui/button";
import { Card, CardBody, CardHeader, CardTitle } from "../../components/ui/card";
import { Input } from "../../components/ui/input";
import { api, ApiRequestError } from "../../lib/api/client";
import type { MemoryRow } from "../../lib/schemas";

/** Search stored facts and teach new ones. Every row comes from `/recall`. */
export function MemoryPage() {
  const [query, setQuery] = React.useState("");
  const [rows, setRows] = React.useState<MemoryRow[] | null>(null);
  const [text, setText] = React.useState("");
  const [stored, setStored] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  async function search(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const payload = await api.recall(query.trim());
      setRows(payload.results);
    } catch (exc) {
      setRows(null);
      setError(exc instanceof ApiRequestError ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function teach(event: React.FormEvent) {
    event.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const payload = await api.teach(text.trim());
      setStored(payload.facts_stored);
      setText("");
    } catch (exc) {
      setError(exc instanceof ApiRequestError ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="flex flex-col gap-4" aria-labelledby="memory-heading">
      <h2 id="memory-heading" className="text-sm font-semibold tracking-wide">
        Memory Explorer
      </h2>
      {error && (
        <Card role="alert" className="border-danger/60">
          <CardBody>{error}</CardBody>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Search stored facts</CardTitle>
        </CardHeader>
        <CardBody>
          <form onSubmit={search} className="flex flex-col gap-2">
            <label htmlFor="memory-query" className="text-sm text-text-secondary">
              Query
            </label>
            <Input
              id="memory-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="What is the calibration constant?"
            />
            <Button type="submit" disabled={busy || !query.trim()}>
              {busy ? "Searching…" : "Search"}
            </Button>
          </form>
          {rows !== null && rows.length === 0 && (
            <p className="mt-3 text-sm text-text-secondary">
              No stored fact matched this query. Nothing is shown rather than a guess.
            </p>
          )}
          {rows !== null && rows.length > 0 && (
            <table className="mt-3 w-full text-left text-sm">
              <thead>
                <tr className="text-text-secondary">
                  <th scope="col" className="py-1">
                    Fact
                  </th>
                  <th scope="col" className="py-1">
                    Source
                  </th>
                  <th scope="col" className="py-1">
                    Score
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={`${index}-${row.text ?? ""}`} className="border-t border-border">
                    <td className="py-1">{row.text ?? "unavailable"}</td>
                    <td className="py-1">{row.source ?? "unattributed"}</td>
                    <td className="py-1">
                      {typeof row.score === "number" ? row.score.toFixed(3) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Teach a fact</CardTitle>
        </CardHeader>
        <CardBody>
          <form onSubmit={teach} className="flex flex-col gap-2">
            <label htmlFor="memory-text" className="text-sm text-text-secondary">
              Text
            </label>
            <Input
              id="memory-text"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="The calibration constant is 0.13."
            />
            <Button type="submit" disabled={busy || !text.trim()}>
              Store
            </Button>
            {stored !== null && (
              <p className="text-xs text-text-secondary">
                The engine reported {stored} stored proposition{stored === 1 ? "" : "s"}.
              </p>
            )}
          </form>
        </CardBody>
      </Card>
    </section>
  );
}
