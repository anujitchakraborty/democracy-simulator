import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";
import { Play, RotateCcw, AlertTriangle, CheckCircle2 } from "lucide-react";

function parseNumberList(value) {
  if (typeof value !== "string") return [];
  return value
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number(s))
    .filter((x) => Number.isFinite(x));
}

function buildRange(start, end, step) {
  const out = [];
  if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(step) || step <= 0) return out;
  if (end < start) return out;
  for (let x = start; x <= end + 1e-9; x += step) {
    out.push(Number(x.toFixed(10)));
  }
  return out;
}

function approxEqual(a, b, tol = 1e-9) {
  return Math.abs(a - b) <= tol;
}

function runSelfTests() {
  const results = [];

  const list1 = parseNumberList("0.1, 0.8");
  results.push({
    name: "parseNumberList parses comma-separated input",
    passed: list1.length === 2 && approxEqual(list1[0], 0.1) && approxEqual(list1[1], 0.8),
  });

  const list2 = parseNumberList("-0.9 -0.7, -0.5 0 0.5");
  results.push({
    name: "parseNumberList parses mixed spaces and commas",
    passed:
      list2.length === 5 &&
      approxEqual(list2[0], -0.9) &&
      approxEqual(list2[1], -0.7) &&
      approxEqual(list2[2], -0.5) &&
      approxEqual(list2[3], 0) &&
      approxEqual(list2[4], 0.5),
  });

  const range = buildRange(2, 6, 2);
  results.push({
    name: "buildRange creates inclusive grid",
    passed: range.length === 3 && range[0] === 2 && range[1] === 4 && range[2] === 6,
  });

  const emptyRange = buildRange(5, 2, 1);
  results.push({
    name: "buildRange rejects descending ranges",
    passed: emptyRange.length === 0,
  });

  return results;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalFromUniforms(u1, u2) {
  const r = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-12)));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-ax * ax);
  return sign * y;
}

function normCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function makeConstantBlockCorrelation(N0, N1, rhoSS, rhoNN, rhoSN) {
  const N = N0 + N1;
  const sigma = Array.from({ length: N }, () => Array(N).fill(rhoSN));
  for (let i = 0; i < N; i++) sigma[i][i] = 1;
  for (let i = 0; i < N0; i++) {
    for (let j = 0; j < N0; j++) sigma[i][j] = i === j ? 1 : rhoSS;
  }
  for (let i = N0; i < N; i++) {
    for (let j = N0; j < N; j++) sigma[i][j] = i === j ? 1 : rhoNN;
  }
  return sigma;
}

function cloneMatrix(A) {
  return A.map((row) => [...row]);
}

function transpose(A) {
  return A[0].map((_, j) => A.map((row) => row[j]));
}

function multiplyMatrices(A, B) {
  const rows = A.length;
  const cols = B[0].length;
  const inner = B.length;
  const out = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = A[i][k];
      for (let j = 0; j < cols; j++) out[i][j] += aik * B[k][j];
    }
  }
  return out;
}

function identityMatrix(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );
}

function symmetrize(A) {
  const n = A.length;
  const out = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i][j] = 0.5 * (A[i][j] + A[j][i]);
    }
  }
  return out;
}

function jacobiEigenDecomposition(A, maxSweeps = 100, tol = 1e-10) {
  const n = A.length;
  let D = cloneMatrix(A);
  let V = identityMatrix(n);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let p = 0;
    let q = 1;
    let maxVal = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val = Math.abs(D[i][j]);
        if (val > maxVal) {
          maxVal = val;
          p = i;
          q = j;
        }
      }
    }

    if (maxVal < tol) break;

    const app = D[p][p];
    const aqq = D[q][q];
    const apq = D[p][q];
    const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const dkp = D[k][p];
        const dkq = D[k][q];
        D[k][p] = c * dkp - s * dkq;
        D[p][k] = D[k][p];
        D[k][q] = s * dkp + c * dkq;
        D[q][k] = D[k][q];
      }
    }

    D[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    D[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    D[p][q] = 0;
    D[q][p] = 0;

    for (let k = 0; k < n; k++) {
      const vkp = V[k][p];
      const vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq;
      V[k][q] = s * vkp + c * vkq;
    }
  }

  const eigenvalues = Array.from({ length: n }, (_, i) => D[i][i]);
  return { eigenvalues, eigenvectors: V };
}

function nearestSPD(A) {
  const B = symmetrize(A);
  const { eigenvalues, eigenvectors } = jacobiEigenDecomposition(B, 200, 1e-12);
  const clipped = eigenvalues.map((x) => Math.max(x, 1e-8));
  const Dplus = Array.from({ length: clipped.length }, (_, i) =>
    Array.from({ length: clipped.length }, (_, j) => (i === j ? clipped[i] : 0))
  );
  const V = eigenvectors;
  const VT = transpose(V);
  let Ahat = multiplyMatrices(multiplyMatrices(V, Dplus), VT);
  Ahat = symmetrize(Ahat);
  return Ahat;
}

function normalizeToCorrelation(A) {
  const n = A.length;
  const out = cloneMatrix(A);
  const d = Array.from({ length: n }, (_, i) => Math.sqrt(Math.max(out[i][i], 1e-12)));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i][j] = out[i][j] / (d[i] * d[j]);
    }
  }
  for (let i = 0; i < n; i++) out[i][i] = 1;
  return symmetrize(out);
}

function shrinkTowardIdentity(A, lambda) {
  const n = A.length;
  const out = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i][j] = (1 - lambda) * A[i][j] + lambda * (i === j ? 1 : 0);
    }
  }
  return out;
}

function choleskyDecomposition(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      if (i === j) {
        const val = A[i][i] - sum;
        if (val <= 1e-12) throw new Error("Matrix not SPD");
        L[i][j] = Math.sqrt(val);
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }
  return L;
}

function makeSPD(A) {
  let B = normalizeToCorrelation(nearestSPD(A));

  for (let attempt = 0; attempt < 25; attempt++) {
    try {
      const L = choleskyDecomposition(B);
      return { matrix: B, L, adjusted: true, jitter: 0, shrinkage: 0 };
    } catch {
      const lambda = Math.min(0.999999, Math.pow(2, attempt) * 1e-6);
      B = normalizeToCorrelation(shrinkTowardIdentity(B, lambda));
      for (let i = 0; i < B.length; i++) B[i][i] += 1e-10;
    }
  }

  const n = A.length;
  const I = identityMatrix(n);
  const L = choleskyDecomposition(I);
  return { matrix: I, L, adjusted: true, jitter: 0, shrinkage: 1 };
}

function sampleMVN(L, rand) {
  const n = L.length;
  const z = Array(n).fill(0);
  for (let i = 0; i < n; i += 2) {
    const [a, b] = normalFromUniforms(rand(), rand());
    z[i] = a;
    if (i + 1 < n) z[i + 1] = b;
  }
  const out = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = 0; k <= i; k++) sum += L[i][k] * z[k];
    out[i] = sum;
  }
  return out;
}

function simulateMajorityProbability({ N0, N1, mu0, mu1, rhoSS, rhoNN, rhoSN, trials, seed }) {
  const rand = mulberry32(seed);

  const sigma0 = makeConstantBlockCorrelation(N0, 0, rhoSS, 0, 0);
  const { L: L0 } = makeSPD(sigma0);
  let correct0 = 0;
  for (let t = 0; t < trials; t++) {
    const z = sampleMVN(L0, rand);
    let votes = 0;
    for (let i = 0; i < N0; i++) {
      if (normCdf(z[i]) < mu0) votes += 1;
    }
    const pred = votes / N0 > 0.5 ? 1 : 0;
    if (pred === 1) correct0 += 1;
  }
  const P0 = correct0 / trials;

  const sigma = makeConstantBlockCorrelation(N0, N1, rhoSS, rhoNN, rhoSN);
  const { L, adjusted, jitter } = makeSPD(sigma);
  let correctN = 0;
  for (let t = 0; t < trials; t++) {
    const z = sampleMVN(L, rand);
    let votes = 0;
    for (let i = 0; i < N0; i++) {
      if (normCdf(z[i]) < mu0) votes += 1;
    }
    for (let i = 0; i < N1; i++) {
      if (normCdf(z[N0 + i]) < mu1) votes += 1;
    }
    const pred = votes / (N0 + N1) > 0.5 ? 1 : 0;
    if (pred === 1) correctN += 1;
  }
  const Pn = correctN / trials;

  return { P0, Pn, gain: Pn - P0, adjusted, jitter };
}

const linePalette = [
  "#111827",
  "#374151",
  "#6b7280",
  "#9ca3af",
  "#4b5563",
  "#1f2937",
  "#7c3aed",
  "#2563eb",
];

function niceTickFormatter(x) {
  return typeof x === "number" ? x.toFixed(2) : x;
}

export default function DemocracyGainSimulatorApp() {
  const [mu0, setMu0] = useState(0.8);
  const [mu1, setMu1] = useState(0.6);
  const [N0, setN0] = useState(5);
  const [N1Start, setN1Start] = useState(2);
  const [N1End, setN1End] = useState(20);
  const [N1Step, setN1Step] = useState(2);
  const [trials, setTrials] = useState(8000);
  const [rhoSS, setRhoSS] = useState(0.9);
  const [rhoNNText, setRhoNNText] = useState("0.1, 0.8");
  const [rhoSNText, setRhoSNText] = useState("-0.9, -0.7, -0.5, -0.3, 0, 0.5");
  const [seed, setSeed] = useState(42);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [tests, setTests] = useState([]);

  useEffect(() => {
    setTests(runSelfTests());
  }, []);

  const parsedInputs = useMemo(() => {
    const rhoNNVals = parseNumberList(rhoNNText);
    const rhoSNVals = parseNumberList(rhoSNText);
    const N1Vals = buildRange(Number(N1Start), Number(N1End), Number(N1Step));
    return { rhoNNVals, rhoSNVals, N1Vals };
  }, [rhoNNText, rhoSNText, N1Start, N1End, N1Step]);

  const runSimulation = async () => {
    setError("");
    setRunning(true);
    setResults(null);

    try {
      const rhoNNVals = parsedInputs.rhoNNVals;
      const rhoSNVals = parsedInputs.rhoSNVals;
      const N1Vals = parsedInputs.N1Vals;

      if (!(mu0 > 0 && mu0 < 1 && mu1 > 0 && mu1 < 1)) {
        throw new Error("Accuracies must lie strictly between 0 and 1.");
      }
      if (!Number.isInteger(N0) || N0 < 1 || N1Vals.length === 0 || !Number.isFinite(trials) || trials < 100) {
        throw new Error("Choose valid sample sizes and at least 100 trials.");
      }
      if (rhoNNVals.length === 0 || rhoSNVals.length === 0) {
        throw new Error("Enter at least one value for rho_NN and rho_SN.");
      }

      await new Promise((resolve) => setTimeout(resolve, 20));

      const panels = [];
      const warnings = [];
      for (let i = 0; i < rhoNNVals.length; i++) {
        const rhoNN = rhoNNVals[i];
        const panelRows = [];
        for (let idx = 0; idx < N1Vals.length; idx++) {
          const row = { N1: N1Vals[idx] };
          for (let r = 0; r < rhoSNVals.length; r++) {
            const rhoSN = rhoSNVals[r];
            const sim = simulateMajorityProbability({
              N0: Number(N0),
              N1: Number(N1Vals[idx]),
              mu0: Number(mu0),
              mu1: Number(mu1),
              rhoSS: Number(rhoSS),
              rhoNN: Number(rhoNN),
              rhoSN: Number(rhoSN),
              trials: Number(trials),
              seed: Number(seed) + 10000 * i + 100 * idx + r,
            });
            row[`rhoSN_${r}`] = sim.gain;
            if (sim.adjusted) {
              warnings.push(
                `Adjusted matrix for rho_NN=${rhoNN}, rho_SN=${rhoSN}, N1=${N1Vals[idx]} (diag jitter ${sim.jitter.toExponential(1)}).`
              );
            }
          }
          panelRows.push(row);
        }
        panels.push({ rhoNN, rows: panelRows });
      }

      setResults({ panels, rhoSNVals, warnings, config: { mu0, mu1, N0, trials, rhoSS } });
    } catch (e) {
      setError(e.message || "Simulation failed.");
    } finally {
      setRunning(false);
    }
  };

  const resetDefaults = () => {
    setMu0(0.8);
    setMu1(0.6);
    setN0(5);
    setN1Start(2);
    setN1End(20);
    setN1Step(2);
    setTrials(8000);
    setRhoSS(0.9);
    setRhoNNText("0.1, 0.8");
    setRhoSNText("-0.9, -0.7, -0.5, -0.3, 0, 0.5");
    setSeed(42);
    setResults(null);
    setError("");
  };

  const testsPassed = tests.length > 0 && tests.every((t) => t.passed);

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">Democracy Gain Simulator</h1>
          <p className="max-w-3xl text-sm text-slate-600">
            Interactive Monte Carlo app for comparing the gain from adding non-experts under correlated evaluations.
            The app replicates the structure of your MATLAB script using a Gaussian copula with block correlation.
          </p>
        </div>

        <Card className="rounded-2xl shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5" />
              Runtime checks
            </CardTitle>
            <CardDescription>Small self-tests for parsing and grid construction.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className={testsPassed ? "text-emerald-700" : "text-amber-700"}>
              {testsPassed ? "All checks passed." : "Some checks failed."}
            </div>
            {tests.map((test) => (
              <div key={test.name} className="flex items-center justify-between rounded-xl border px-3 py-2">
                <span>{test.name}</span>
                <Badge variant={test.passed ? "secondary" : "destructive"}>{test.passed ? "Pass" : "Fail"}</Badge>
              </div>
            ))}
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Card className="rounded-2xl shadow-sm lg:col-span-1">
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
              <CardDescription>Adjust inputs, then run the simulation.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="mu0">Expert accuracy (mu0)</Label>
                  <Input id="mu0" type="number" step="0.01" value={mu0} onChange={(e) => setMu0(Number(e.target.value))} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mu1">Non-expert accuracy (mu1)</Label>
                  <Input id="mu1" type="number" step="0.01" value={mu1} onChange={(e) => setMu1(Number(e.target.value))} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="n0">Experts (N0)</Label>
                  <Input id="n0" type="number" step="1" value={N0} onChange={(e) => setN0(Number(e.target.value))} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="trials">Trials</Label>
                  <Input id="trials" type="number" step="100" value={trials} onChange={(e) => setTrials(Number(e.target.value))} />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="rhoSS">Within-expert correlation (rho_SS)</Label>
                <Input id="rhoSS" type="number" step="0.01" value={rhoSS} onChange={(e) => setRhoSS(Number(e.target.value))} />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="n1start">N1 start</Label>
                  <Input id="n1start" type="number" step="1" value={N1Start} onChange={(e) => setN1Start(Number(e.target.value))} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="n1end">N1 end</Label>
                  <Input id="n1end" type="number" step="1" value={N1End} onChange={(e) => setN1End(Number(e.target.value))} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="n1step">N1 step</Label>
                  <Input id="n1step" type="number" step="1" value={N1Step} onChange={(e) => setN1Step(Number(e.target.value))} />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="rhoNN">List of rho_NN values</Label>
                <Input id="rhoNN" value={rhoNNText} onChange={(e) => setRhoNNText(e.target.value)} />
              </div>

              <div className="space-y-2">
                <Label htmlFor="rhoSN">List of rho_SN values</Label>
                <Input id="rhoSN" value={rhoSNText} onChange={(e) => setRhoSNText(e.target.value)} />
              </div>

              <div className="space-y-2">
                <Label htmlFor="seed">Random seed</Label>
                <Input id="seed" type="number" step="1" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              </div>

              <div className="flex flex-wrap gap-3 pt-2">
                <Button className="rounded-2xl" onClick={runSimulation} disabled={running}>
                  <Play className="mr-2 h-4 w-4" />
                  {running ? "Running..." : "Run simulation"}
                </Button>
                <Button variant="outline" className="rounded-2xl" onClick={resetDefaults}>
                  <RotateCcw className="mr-2 h-4 w-4" />
                  Reset
                </Button>
              </div>

              <div className="flex flex-wrap gap-2 pt-2 text-xs text-slate-500">
                <Badge variant="secondary">Gaussian copula</Badge>
                <Badge variant="secondary">Majority rule</Badge>
                <Badge variant="secondary">Block correlation matrix</Badge>
              </div>
            </CardContent>
          </Card>

          <div className="space-y-6 lg:col-span-2">
            {error && (
              <Alert className="rounded-2xl border-red-300 bg-red-50 text-red-900">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Simulation error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {!results && !error && (
              <Card className="rounded-2xl shadow-sm">
                <CardContent className="flex min-h-[420px] items-center justify-center p-8 text-center text-slate-500">
                  Run the simulation to generate the comparison plots.
                </CardContent>
              </Card>
            )}

            {results && (
              <>
                <Card className="rounded-2xl shadow-sm">
                  <CardHeader>
                    <CardTitle>Summary</CardTitle>
                    <CardDescription>
                      mu0 = {results.config.mu0}, mu1 = {results.config.mu1}, N0 = {results.config.N0}, trials = {results.config.trials}, rho_SS = {results.config.rhoSS}.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-wrap gap-2 text-sm text-slate-600">
                    <Badge variant="outline">rho_NN: {parsedInputs.rhoNNVals.join(", ")}</Badge>
                    <Badge variant="outline">rho_SN: {parsedInputs.rhoSNVals.join(", ")}</Badge>
                    <Badge variant="outline">N1 grid: {parsedInputs.N1Vals.join(", ")}</Badge>
                  </CardContent>
                </Card>

                {results.panels.map((panel, panelIndex) => (
                  <Card key={panel.rhoNN} className="rounded-2xl shadow-sm">
                    <CardHeader>
                      <CardTitle>Gain vs. N1 for rho_NN = {panel.rhoNN}</CardTitle>
                      <CardDescription>
                        Gain is the difference between full-sample majority accuracy and expert-only majority accuracy.
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[360px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={panel.rows} margin={{ top: 10, right: 24, left: 8, bottom: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="N1" label={{ value: "N1", position: "insideBottom", offset: -4 }} />
                            <YAxis tickFormatter={niceTickFormatter} label={{ value: "Gain = Pn - P0", angle: -90, position: "insideLeft" }} />
                            <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)} />
                            <Legend />
                            <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 4" />
                            {results.rhoSNVals.map((rhoSN, r) => (
                              <Line
                                key={`${panelIndex}-${rhoSN}`}
                                type="monotone"
                                dataKey={`rhoSN_${r}`}
                                name={`rho_SN = ${rhoSN}`}
                                stroke={linePalette[r % linePalette.length]}
                                strokeWidth={2}
                                dot={{ r: 2 }}
                                activeDot={{ r: 4 }}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                ))}

                {results.warnings.length > 0 && (
                  <Card className="rounded-2xl border-amber-300 bg-amber-50 shadow-sm">
                    <CardHeader>
                      <CardTitle className="text-amber-900">Matrix adjustments</CardTitle>
                      <CardDescription className="text-amber-800">
                        Some parameter combinations required a small diagonal adjustment to make the correlation matrix numerically SPD.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm text-amber-900">
                      {results.warnings.slice(0, 12).map((w, i) => (
                        <div key={i}>• {w}</div>
                      ))}
                      {results.warnings.length > 12 && <div>• ...and {results.warnings.length - 12} more.</div>}
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
