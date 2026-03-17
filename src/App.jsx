import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Play, RotateCcw, AlertTriangle } from "lucide-react";

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
    for (let j = 0; j < n; j++) out[i][j] = 0.5 * (A[i][j] + A[j][i]);
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
    for (let j = 0; j < n; j++) out[i][j] = out[i][j] / (d[i] * d[j]);
  }
  for (let i = 0; i < n; i++) out[i][i] = 1;
  return symmetrize(out);
}

function shrinkTowardIdentity(A, lambda) {
  const n = A.length;
  const out = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i][j] = (1 - lambda) * A[i][j] + lambda * (i === j ? 1 : 0);
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
    for (let i = 0; i < N0; i++) if (normCdf(z[i]) < mu0) votes += 1;
    const pred = votes / N0 > 0.5 ? 1 : 0;
    if (pred === 1) correct0 += 1;
  }
  const P0 = correct0 / trials;

  const sigma = makeConstantBlockCorrelation(N0, N1, rhoSS, rhoNN, rhoSN);
  const { L, adjusted, shrinkage } = makeSPD(sigma);

  let correctN = 0;
  for (let t = 0; t < trials; t++) {
    const z = sampleMVN(L, rand);
    let votes = 0;
    for (let i = 0; i < N0; i++) if (normCdf(z[i]) < mu0) votes += 1;
    for (let i = 0; i < N1; i++) if (normCdf(z[N0 + i]) < mu1) votes += 1;
    const pred = votes / (N0 + N1) > 0.5 ? 1 : 0;
    if (pred === 1) correctN += 1;
  }
  const Pn = correctN / trials;

  return { P0, Pn, gain: Pn - P0, adjusted, shrinkage };
}

const palette = [
  "#1f77b4",  // blue
  "#d62728",  // red
  "#2ca02c",  // green
  "#9467bd",  // purple
  "#ff7f0e",  // orange
  "#17becf",  // teal
  "#8c564b",  // brown
];
function sectionStyle() {
  return {
    background: "white",
    border: "1px solid #e2e8f0",
    borderRadius: 16,
    padding: 20,
    boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
  };
}

function labelStyle() {
  return {
    display: "block",
    fontSize: 13,
    fontWeight: 600,
    marginBottom: 6,
  };
}

function inputStyle() {
  return {
    width: "100%",
    padding: "10px 12px",
    border: "1px solid #cbd5e1",
    borderRadius: 10,
    fontSize: 14,
    boxSizing: "border-box",
  };
}

function buttonStyle(primary = true) {
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 14px",
    borderRadius: 12,
    border: primary ? "none" : "1px solid #cbd5e1",
    background: primary ? "#0f172a" : "white",
    color: primary ? "white" : "#0f172a",
    cursor: "pointer",
    fontWeight: 600,
  };
}

export default function App() {
  const [mu0, setMu0] = useState(0.8);
  const [mu1, setMu1] = useState(0.6);
  const [N0, setN0] = useState(5);
  const [N1Start, setN1Start] = useState(2);
  const [N1End, setN1End] = useState(20);
  const [N1Step, setN1Step] = useState(2);
  const [trials, setTrials] = useState(10000);
  const [rhoSS, setRhoSS] = useState(0.9);
  const [rhoNNText, setRhoNNText] = useState("0.1, 0.8");
  const [rhoSNText, setRhoSNText] = useState("-0.9, -0.7, -0.5, -0.3, 0, 0.5");
  const [seed, setSeed] = useState(42);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);



  const parsedInputs = useMemo(() => {
    return {
      rhoNNVals: parseNumberList(rhoNNText),
      rhoSNVals: parseNumberList(rhoSNText),
      N1Vals: buildRange(Number(N1Start), Number(N1End), Number(N1Step)),
    };
  }, [rhoNNText, rhoSNText, N1Start, N1End, N1Step]);

  async function runSimulation() {
    setError("");
    setRunning(true);
    setResults(null);

    try {
      const { rhoNNVals, rhoSNVals, N1Vals } = parsedInputs;

      if (!(mu0 > 0 && mu0 < 1 && mu1 > 0 && mu1 < 1)) {
        throw new Error("Accuracies must lie strictly between 0 and 1.");
      }
      if (!Number.isInteger(N0) || N0 < 1 || N1Vals.length === 0 || !Number.isFinite(trials) || trials < 100) {
        throw new Error("Choose valid sample sizes and at least 100 trials.");
      }
      if (rhoNNVals.length === 0 || rhoSNVals.length === 0) {
        throw new Error("Enter at least one value for ρ_NN and ρ_EN.");
      }

      await new Promise((resolve) => setTimeout(resolve, 20));

      const panels = [];
      const warnings = [];

      for (let i = 0; i < rhoNNVals.length; i++) {
        const rhoNN = rhoNNVals[i];
        const rows = [];

        for (let idx = 0; idx < N1Vals.length; idx++) {
          const N1 = N1Vals[idx];
          const row = { N1 };

          for (let r = 0; r < rhoSNVals.length; r++) {
            const rhoSN = rhoSNVals[r];
            const sim = simulateMajorityProbability({
              N0: Number(N0),
              N1: Number(N1),
              mu0: Number(mu0),
              mu1: Number(mu1),
              rhoSS: Number(rhoSS),
              rhoNN: Number(rhoNN),
              rhoSN: Number(rhoSN),
              trials: Number(trials),
              seed: Number(seed) + 10000 * i + 100 * idx + r,
            });

            row[`rhoSN_${r}`] = sim.gain;

            if (sim.adjusted && sim.shrinkage > 0) {
              warnings.push(`rho_NN=${rhoNN}, rho_EN=${rhoSN}, N1=${N1}: repaired infeasible correlation matrix.`);
            }
          }

          rows.push(row);
        }

        panels.push({ rhoNN, rows });
      }

      setResults({ panels, rhoSNVals: parsedInputs.rhoSNVals, warnings });
    } catch (e) {
      setError(e.message || "Simulation failed.");
    } finally {
      setRunning(false);
    }
  }

  function resetDefaults() {
    setMu0(0.8);
    setMu1(0.6);
    setN0(5);
    setN1Start(2);
    setN1End(20);
    setN1Step(2);
    setTrials(10000);
    setRhoSS(0.9);
    setRhoNNText("0.1, 0.8");
    setRhoSNText("-0.9, -0.7, -0.5, -0.3, 0, 0.5");
    setSeed(42);
    setResults(null);
    setError("");
  }


  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", padding: 24, fontFamily: "Arial, sans-serif", color: "#0f172a" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", display: "grid", gap: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 32 }}>Wisdom of the Crowds Simulator</h1>
          <p style={{ color: "#475569", maxWidth: 800 }}>
            Interactive Monte Carlo app for comparing the gain from adding non-experts (N) under correlated evaluations.
          </p>
        </div>

        
        <div style={{ display: "grid", gridTemplateColumns: "minmax(300px, 360px) 1fr", gap: 24 }}>
          <div style={sectionStyle()}>
            <h2 style={{ marginTop: 0 }}>Pick Parameters</h2>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div>
                <label style={labelStyle()}>Expert accuracy (μ<sub>E</sub>)</label>
                <input style={inputStyle()} type="number" step="0.01" value={mu0} onChange={(e) => setMu0(Number(e.target.value))} />
              </div>
              <div>
                <label style={labelStyle()}>Non-expert accuracy (μ<sub>N</sub>)</label>
                <input style={inputStyle()} type="number" step="0.01" value={mu1} onChange={(e) => setMu1(Number(e.target.value))} />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
              <div>
                <label style={labelStyle()}>Experts (E)</label>
                <input style={inputStyle()} type="number" step="1" value={N0} onChange={(e) => setN0(Number(e.target.value))} />
              </div>
              <div>
                <label style={labelStyle()}>Trials</label>
                <input style={inputStyle()} type="number" step="1000" value={trials} onChange={(e) => setTrials(Number(e.target.value))} />
              </div>
            </div>

            <div style={{ marginTop: 14 }}>
              <label style={labelStyle()}>Within-expert correlation (ρ<sub>EE</sub>)</label>
              <input style={inputStyle()} type="number" step="0.01" value={rhoSS} onChange={(e) => setRhoSS(Number(e.target.value))} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginTop: 14 }}>
              <div>
                <label style={labelStyle()}>Non-experts (E) min</label>
                <input style={inputStyle()} type="number" step="1" value={N1Start} onChange={(e) => setN1Start(Number(e.target.value))} />
              </div>
              <div>
                <label style={labelStyle()}>Non-experts (E) max</label>
                <input style={inputStyle()} type="number" step="1" value={N1End} onChange={(e) => setN1End(Number(e.target.value))} />
              </div>
              <div>
                <label style={labelStyle()}>E step-size</label>
                <input style={inputStyle()} type="number" step="1" value={N1Step} onChange={(e) => setN1Step(Number(e.target.value))} />
              </div>
            </div>

            <div style={{ marginTop: 14 }}>
              <label style={labelStyle()}>Within-nonexpert correlation values (ρ<sub>NN</sub>)</label>
              <input style={inputStyle()} value={rhoNNText} onChange={(e) => setRhoNNText(e.target.value)} />
            </div>

            <div style={{ marginTop: 14 }}>
              <label style={labelStyle()}>Expert-nonexpert correlation values (ρ<sub>EN</sub>)</label>
              <input style={inputStyle()} value={rhoSNText} onChange={(e) => setRhoSNText(e.target.value)} />
            </div>

            <div style={{ marginTop: 14 }}>
              <label style={labelStyle()}>Random seed</label>
              <input style={inputStyle()} type="number" step="1" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </div>

            <div style={{ display: "flex", gap: 10, marginTop: 18, flexWrap: "wrap" }}>
              <button style={buttonStyle(true)} onClick={runSimulation} disabled={running}>
                <Play size={16} />
                {running ? "Running..." : "Run simulation"}
              </button>
              <button style={buttonStyle(false)} onClick={resetDefaults}>
                <RotateCcw size={16} />
                Reset
              </button>
            </div>
          </div>

          <div style={{ display: "grid", gap: 24 }}>
            
            <div
              style={{
                ...sectionStyle(),
                background: "#f8fafc",
                borderColor: "#cbd5e1",
                padding: "16px 18px"
              }}
            >
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  marginBottom: 8,
                  color: "#0f172a"
                }}
              >
                Suggested experiment
              </div>
            
              <div style={{ fontSize: 16, lineHeight: 1.8, color: "#475569" }}>
                Can non-experts contribute in the wisdom of crowds? 
                Generally speaking, even if 
                <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
                  {" "}μ<sub>E</sub> > μ<sub>N</sub>
                  </span>, 
                non-experts can contribute by increasing the sample size, and by potentially decorrelating the assessments (say when 
                <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
                  {" "}ρ<sub>EN</sub> &lt; 0 but ρ<sub>EE</sub>, ρ<sub>NN</sub> &gt; 0
                  </span>).  You can experiment with your chosen values of
                <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
                  {" "}μ<sub>E</sub>, μ<sub>N</sub>, E, ρ<sub>EE</sub>, ρ<sub>NN</sub>, ρ<sub>EN</sub>
                </span>.
              </div>
               <p style={{ fontSize: 16, lineHeight: 1.8, color: "#475569" }}>
                      In the plot below, Gain is the increase in the probability of the majority picking the right state when including non-expert opinions (N+E), compared to without. For your chosen parameters, we plot Gain as a function of the number of non-experts included.
                    </p>
                    
            </div>

            


  <details
  style={{
    ...sectionStyle(),
    background: "#ffffff",
    borderColor: "#e2e8f0",
    padding: "16px 18px"
  }}
>
  <summary
    style={{
      fontSize: 16,
      fontWeight: 600,
      cursor: "pointer",
      color: "#0f172a",
      marginBottom: 8
    }}
  >
    Technical details of the simulation
  </summary>

 

  <div style={{ fontSize: 15, lineHeight: 1.7, color: "#475569", marginTop: 10 }}>
    Individual responses are generated by thresholding latent jointly Gaussian variables.
    Expert and non-expert accuracies correspond to Bernoulli probabilities
    <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
      {" "}μ<sub>E</sub> 
    </span>
     and
    <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
      {" "}μ<sub>N</sub>
    </span>.
    Correlation across responses is introduced through a Gaussian copula: a latent
    multivariate normal vector is generated with correlations
    <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
      {" "}ρ<sub>EE</sub>, ρ<sub>NN</sub>, ρ<sub>EN</sub>
    </span>,
    and binary responses are obtained by thresholding these latent variables.
  </div>

    <div style={{ fontSize: 15, lineHeight: 1.7, color: "#475569", marginTop: 10 }}>
  To simulate the latent Gaussian variables, the app first constructs a block
  correlation matrix using the chosen values of
  <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
    {" "}ρ<sub>EE</sub>, ρ<sub>NN</sub>, ρ<sub>EN</sub>
  </span>.
  A Cholesky decomposition of this matrix is then used to generate correlated
  normal draws from independent standard normal shocks. Because some user-chosen
  parameter combinations do not produce a valid positive semidefinite correlation
  matrix, the app applies a numerical repair step before the decomposition:
  it symmetrizes the matrix, projects it to a nearby positive semidefinite
  matrix, normalizes it back to a correlation matrix, and, if needed, shrinks
  it slightly toward the identity matrix. This ensures that the simulation can
  proceed even when the requested latent correlation structure is infeasible.
</div>

  <div style={{ fontSize: 15, lineHeight: 1.7, color: "#475569", marginTop: 10 }}>
    Each point in the graph represents the average gain
    computed over the specified number of Monte Carlo simulations for the chosen parameter
    vector
    <span style={{ fontFamily: "Georgia, Times New Roman, serif", color: "#0f172a" }}>
      {" "}(ρ<sub>EE</sub>, ρ<sub>NN</sub>, ρ<sub>EN</sub>, E)
    </span>.
  </div>
</details>
            
            {error ? (
              <div style={{ ...sectionStyle(), borderColor: "#fca5a5", background: "#fef2f2", color: "#991b1b" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <AlertTriangle size={18} />
                  <strong>Simulation error</strong>
                </div>
                <div>{error}</div>
              </div>
            ) : null}

            {!results && !error ? (
              <div style={{ ...sectionStyle(), minHeight: 420, display: "flex", alignItems: "center", justifyContent: "center", color: "#64748b" }}>
                Run the simulation to generate the comparison plots.
              </div>
            ) : null}

            {results
              ? results.panels.map((panel, panelIndex) => (
                  <div key={panel.rhoNN} style={sectionStyle()}>
                    <h3 style={{ marginTop: 0 }}>Gain is plotted against the number of Non-experts included, for ρ<sub>NN</sub> = {panel.rhoNN}</h3>
                   
                    
                    <div style={{ width: "100%", height: 360 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={panel.rows} margin={{ top: 10, right: 24, left: 8, bottom: 8 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="N1" />
                          <YAxis />
                          <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)} />
                          <Legend />
                          <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 4" />
                          {results.rhoSNVals.map((rhoSN, r) => (
                            <Line
                              key={`${panelIndex}-${rhoSN}`}
                              type="monotone"
                              dataKey={`rhoSN_${r}`}
                              name={`ρ_EN = ${rhoSN}`}
                              stroke={palette[r % palette.length]}
                              strokeWidth={2}
                              dot={{ r: 2 }}
                              activeDot={{ r: 4 }}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ))
              : null}

            {results && results.warnings.length > 0 ? (
              <div style={{ ...sectionStyle(), borderColor: "#fcd34d", background: "#fffbeb" }}>
                <h3 style={{ marginTop: 0 }}>Matrix adjustments</h3>
                <p style={{ color: "#78350f" }}>
                  Some parameter combinations required repairing an infeasible correlation matrix.
                </p>
                <div style={{ display: "grid", gap: 6, fontSize: 14 }}>
                  {results.warnings.slice(0, 12).map((w, i) => (
                    <div key={i}>• {w}</div>
                  ))}
                  {results.warnings.length > 12 ? <div>• ...and {results.warnings.length - 12} more.</div> : null}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
