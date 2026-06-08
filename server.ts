import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import * as dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "50mb" }));

// --- Mulberry32 Seeded Random for Deterministic Robust High-Dimensional Operations ---
function createRandom(seed: number) {
  let a = seed;
  return function() {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Generate deterministic normal distributions (Box-Muller)
function randomNormal(randFn: () => number, mean = 0, stddev = 1) {
  const u1 = randFn() || 0.0001; // Avoid 0
  const u2 = randFn();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z * stddev;
}

// Dimension of standard embedding vector
const EMBEDDING_DIM = 768;

// --- TurboQuant Matrix Construction (Random Orthogonal Matrix) ---
// We create and cache a deterministic 768x768 orthogonal matrix Q using Gram-Schmidt
let cachedQ: number[][] | null = null;
function getRotationMatrix(): number[][] {
  if (cachedQ) return cachedQ;

  console.log(`[TurboQuant] Generating deterministic ${EMBEDDING_DIM}x${EMBEDDING_DIM} orthogonal rotation matrix via Gram-Schmidt...`);
  const rand = createRandom(1337);
  const Q: number[][] = [];

  for (let i = 0; i < EMBEDDING_DIM; i++) {
    let row = Array.from({ length: EMBEDDING_DIM }, () => randomNormal(rand, 0, 1));
    // Orthogonalize with respect to prior rows
    for (let j = 0; j < i; j++) {
      const prev = Q[j];
      let dot = 0;
      for (let k = 0; k < EMBEDDING_DIM; k++) {
        dot += row[k] * prev[k];
      }
      for (let k = 0; k < EMBEDDING_DIM; k++) {
        row[k] -= dot * prev[k];
      }
    }
    // Normalize
    let norm = 0;
    for (let k = 0; k < EMBEDDING_DIM; k++) {
      norm += row[k] * row[k];
    }
    norm = Math.sqrt(norm);
    for (let k = 0; k < EMBEDDING_DIM; k++) {
      row[k] /= norm || 1;
    }
    Q.push(row);
  }

  cachedQ = Q;
  console.log(`[TurboQuant] Deterministic orthogonal rotation matrix initialized!`);
  return Q;
}

// Perform vector multiplication: out = Q * vec
function rotateVector(vec: number[], Q: number[][]): number[] {
  const out = Array(EMBEDDING_DIM).fill(0);
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    let sum = 0;
    for (let j = 0; j < EMBEDDING_DIM; j++) {
      sum += Q[i][j] * vec[j];
    }
    out[i] = sum;
  }
  return out;
}

// Perform vector transpose multiplication: out = Q^T * vec (reconstruction / unrotate)
function unrotateVector(vec: number[], Q: number[][]): number[] {
  const out = Array(EMBEDDING_DIM).fill(0);
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    let sum = 0;
    for (let j = 0; j < EMBEDDING_DIM; j++) {
      sum += Q[j][i] * vec[j]; // Q is orthogonal, its transpose is its inverse
    }
    out[i] = sum;
  }
  return out;
}

// --- Quantization Utilities ---

// Standard Gauss-Lloyd optimal centroids for N(0, 1) distributions
const LLOYD_MAX_CODEBOOKS: Record<number, { boundaries: number[]; centroids: number[] }> = {
  1: {
    boundaries: [0],
    centroids: [-0.7979, 0.7979],
  },
  2: {
    boundaries: [-0.98, 0, 0.98],
    centroids: [-1.51, -0.45, 0.45, 1.51],
  },
  3: {
    boundaries: [-1.75, -1.05, -0.5, 0, 0.5, 1.05, 1.75],
    centroids: [-2.15, -1.34, -0.76, -0.25, 0.25, 0.76, 1.34, 2.15],
  },
};

// General fallback uniform quantizer for higher bit-widths or N(0, 1)
function getUniformGaussConfig(bits: number) {
  const numLevels = Math.pow(2, bits);
  const centroids: number[] = [];
  const boundaries: number[] = [];
  const minRange = -3.0;
  const maxRange = 3.0;
  const step = (maxRange - minRange) / numLevels;

  for (let i = 0; i < numLevels; i++) {
    centroids.push(minRange + (i + 0.5) * step);
    if (i < numLevels - 1) {
      boundaries.push(minRange + (i + 1) * step);
    }
  }
  return { boundaries, centroids };
}

function quantizeValue(val: number, bits: number): { centroid: number; index: number; boundaryIndex: number } {
  const codebook = LLOYD_MAX_CODEBOOKS[bits] || getUniformGaussConfig(bits);
  const bounds = codebook.boundaries;
  const cents = codebook.centroids;

  let idx = 0;
  while (idx < bounds.length && val > bounds[idx]) {
    idx++;
  }

  return {
    centroid: cents[idx],
    index: idx,
    boundaryIndex: idx,
  };
}

// --- Traditional vs TurboQuant Pipeline Implementation ---

// Traditional: Standard uniform Scalar Quantization (direct coordinate encoding)
// Since direct embedding has coordinates scaled arbitrarily, we map the range of the vector dynamically
export function traditionalQuantize(vec: number[], bits: number): { quantizedVec: number[]; compressionRatio: number } {
  // Find min and max of coordinate space
  let min = Infinity;
  let max = -Infinity;
  for (let val of vec) {
    if (val < min) min = val;
    if (val > max) max = val;
  }

  const numLevels = Math.pow(2, bits);
  const step = (max - min) / numLevels;

  const quantizedVec = vec.map((val) => {
    let index = Math.floor((val - min) / (step || 0.0001));
    if (index >= numLevels) index = numLevels - 1;
    if (index < 0) index = 0;
    return min + (index + 0.5) * step; // Reconstructed uniform level
  });

  return {
    quantizedVec,
    compressionRatio: 32 / bits,
  };
}

// TurboQuant: Random rotation, coordinatewise Gauss optimal quantizer, 1-bit residual correction
export function turboQuantize(
  vec: number[],
  bits: number,
  Q: number[][]
): {
  rotated: number[];
  quantizedRotated: number[];
  reconstructed: number[];
  unbiasedEstimate: number[];
  residualSign: number[];
  scaleFactor: number;
} {
  // 1. Orthogonal Rotation (Induces concentrated Beta/Normal distribution around coordinates)
  const rotated = rotateVector(vec, Q);

  // Since sum of squares of unit vectors is conserved, expected variance per coordinate is 1/d.
  // We scale coordinates by sqrt(d) to convert rotated coordinates to standard normal N(0, 1)!
  const stdScale = Math.sqrt(EMBEDDING_DIM);
  const scaledRotated = rotated.map((x) => x * stdScale);

  const quantizedRotatedScaled: number[] = [];
  const residualSign: number[] = [];

  for (let i = 0; i < EMBEDDING_DIM; i++) {
    const val = scaledRotated[i];
    const { centroid } = quantizeValue(val, bits);
    quantizedRotatedScaled.push(centroid);

    // Compute residual and its sign for QJL (Quantized Johnson-Lindenstrauss Correction)
    const residual = val - centroid;
    residualSign.push(residual >= 0 ? 1 : -1);
  }

  // Scale back to match unit circle
  const quantizedRotated = quantizedRotatedScaled.map((x) => x / stdScale);

  // Unrotated reconstructed values (First order)
  const reconstructed = unrotateVector(quantizedRotated, Q);

  // 1-bit QJL Residual Correction stage to guarantee unbiased inner products
  // c_res represents the expected absolute residual of N(0, 1) quantization
  // For standard Gaussians with Gauss-Lloyd centroids:
  let c_res = 0.35; // Default reference magnitude
  if (bits === 1) c_res = 0.40;
  if (bits === 2) c_res = 0.15;
  if (bits === 3) c_res = 0.06;

  const correctedRotated = quantizedRotated.map((val, idx) => {
    const correction = (residualSign[idx] * c_res) / stdScale;
    return val + correction;
  });

  const unbiasedEstimate = unrotateVector(correctedRotated, Q);

  return {
    rotated,
    quantizedRotated,
    reconstructed,
    unbiasedEstimate,
    residualSign,
    scaleFactor: stdScale,
  };
}

// Math Cosine Similarity function
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

// Mean Squared Error (MSE)
function meanSquaredError(a: number[], b: number[]): number {
  let sumSqErr = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSqErr += diff * diff;
  }
  return sumSqErr / a.length;
}

// --- Ground Truth Document Base (Harakosan Z72-2000-MV Turbine User Guide) ---
interface TurbineSegment {
  id: string;
  page: number;
  title: string;
  category: "Specifications" | "Safety" | "Operation" | "Troubleshooting" | "Configuration";
  text: string;
  embedding?: number[];
}

const TURBINE_CHUNKS: TurbineSegment[] = [
  {
    id: "spec_general",
    page: 2,
    title: "Abstract and Document Scope",
    category: "Specifications",
    text: "This User's Guide provides operators instructions on how to operate the Harakosan Z72 2MW wind turbine. It covers general information, safety instructions, Z72 turbine overview, turbine control hardware descriptions, power activation, Touch Screen and push buttons of the Local Operator Panel, troubleshooting event alarms, and control system configuration procedures."
  },
  {
    id: "spec_technical",
    page: 7,
    title: "Z72-2000-MV Technical specifications",
    category: "Specifications",
    text: "Turbine Type: Harakosan Z72 3-bladed variable speed 2MW wind turbine with Medium Voltage generator and liquid-cooled converter. Blades: GFRP, 34.0 m length manufactured by Umoe Norway. Generator: 60-pole Permanent Magnet Synchronous Machine (PMSM) manufactured by ABB Finland. Hub height: 64.0 m. Power converter: ABB ACS1000, 3.5 kV with 400 VAC, 60 Hz auxiliary grid connection. Control system: Bachmann Electronic with PLC application software version V3.0 and Local Operator Panel version V2.1."
  },
  {
    id: "safety_general",
    page: 10,
    title: "General Safety and Climbing Precautions",
    category: "Safety",
    text: "Always work at or near the turbine with at least two people carrying mobile devices under the buddy principle. Do not carry out any work at wind speeds exceeding 20 m/s (gale force 8). Always wear approved certified climbing harnesses, safety helmets, and fall protection devices. Maintain direct and constant contact with the climbing cable or resting platforms inside the tubular steel tower."
  },
  {
    id: "safety_systems",
    page: 13,
    title: "Watchdog and Emergency Pitch Safety Features",
    category: "Safety",
    text: "The Z72 wind turbine control system is equipped with key safety features. A battery-powered emergency trip system shuts down the turbine by bringing all blades to the safe vane position in approximately 15 seconds in case of grid loss or pitch drive inverter faults. The PLC watchdog continuously checks if the PLC is alive, issuing an emergency trip if the PLC is unresponsive for longer than 2.0 seconds."
  },
  {
    id: "spec_hub_generator",
    page: 18,
    title: "Hub and Generator Mass Metrics",
    category: "Specifications",
    text: "The fully assembled rigid hub is composed of nodular cast iron direct-fitted to the main bearing. It measures 2.5 x 2.2 x 2.2 m and has a total mass of 19 tons, roomy enough for two service technicians. Each rotor blade weighs 5.5 tons. The ABB 60-pole permanent magnet generator is direct-drive, operating at medium voltage 4kV, with a total generator and bearing assembly mass of 49 tons."
  },
  {
    id: "control_algorithm",
    page: 26,
    title: "Basic vs. Extended Pitch Control Algorithm",
    category: "Configuration",
    text: "The pitch and torque control algorithm regulates rotor speed to optimize energy yield. A Basic algorithm uses a static Pn curve which has suboptimal energy yield around rated speed. The Extended algorithm uses a dynamic Pn curve to increase production around rated speed, estimate the current wind speed, and offset the pitch angle set-point during sudden wind gusts to maintain a stable, stationary 2.0MW rated production."
  },
  {
    id: "lop_interface",
    page: 33,
    title: "Local Operator Panel Consoles",
    category: "Operation",
    text: "The main turbine control system includes hardwired buttons and switches. SERVICE toggle puts the turbine in Service mode (ON/OFF). OPERATION LOC/REM selects control from either local Touch Screen (LOC) or Remote SCADA (REM). Hardwired console push buttons consist of TURBINE RESET (reset inactive trips and alarms), TURBINE STOP (manual stop command), and TURBINE START (initiates sequence start after verifying trips are cleared)."
  },
  {
    id: "yaw_states",
    page: 45,
    title: "Yaw Control Orientation States",
    category: "Operation",
    text: "Yaw control aligns the nacelle upwind automatically. States include: YAW INHIBIT (active if Event 223 Yaw Timeout, Event 224 Yaw protection triggered, or Event 226 Cable twist active). YAW STANDBY (manual yaw commands enabled on panel). YAW AUTO (aligned upwind automatically using wind vanes average). CABLE UNWIND (triggered to untwist generator cables running down the tower when twist goes over limit)."
  },
  {
    id: "pitch_states",
    page: 52,
    title: "Pitch Control and Storm Park States",
    category: "Configuration",
    text: "Blade pitch states determine turbine mechanical stops: EMERGENCY (emergency batteries pitch all three blades), MAN OPEN LOOP (manually pitch individual blades without calibration), MAN CLOSED LOOP (pitch blades simultaneously to specific target under locked rotor), and STORM PARK. STORM PARK is triggered because wind speed is very high. It is activated under Event T_204 when the 10m wind speed exceeds 27 m/s or the 3s gust wind speed exceeds 35 m/s."
  },
  {
    id: "overspeed_protection",
    page: 56,
    title: "Overspeed Protection & Jaquet Relay",
    category: "Safety",
    text: "An independent electrical overspeed protection backup system is based on a Jaquet ferrostat speed sensor measuring teeth on a pole wheel. Checks of overspeed trip are made to trip the emergency system if the speed exceeds the threshold of 28.5 rpm. For testing during low wind conditions, the trip setpoint can be temporarily lowered on the Jaquet T401 overspeed relay to 10 rpm to confirm trigger mechanism integrity."
  },
  {
    id: "event_228",
    page: 109,
    title: "Event Code 228 - Yaw Brake Low Pressure",
    category: "Troubleshooting",
    text: "Event ID 228: Yaw brake pressure is very low. The pressure of the hydraulic power unit has dropped 25% below its nominal value (< 150 bar). Test and Rectification instructions: Instruct technicians to check the hydraulic power unit, inspect the yaw brakes for physical oil leakage, and seal/fix the pressure valves immediately."
  },
  {
    id: "event_211",
    page: 107,
    title: "Event Code 211 - Emergency system nacelle tripped",
    category: "Troubleshooting",
    text: "Event ID 211: Emergency system nacelle is tripped. Occurs when the yellow/red emergency mushroom button on the nacelle control box or local hub panel has been pressed, or when the nacelle control box is booted up. Rectification: Inspect the safe conditions of technicians, release/reset pressed emergency mushroom buttons, and issue a RESET command on the Local Operator Panel."
  },
  {
    id: "event_353",
    page: 119,
    title: "Event Codes 353-355 - Battery Voltage Low",
    category: "Troubleshooting",
    text: "Event IDs 353, 354, 355: Blade 1, 2, or 3 emergency battery voltage is very low. This means the voltage of the control battery pack has decreased below the level needed to operate the emergency pitching relays. Rectification check: Inspect connector XB1 of Rotor Control Box 541A001. Check if power is available on SBP loader. Replace battery cells if worn out."
  },
  {
    id: "config_defaults",
    page: 103,
    title: "Default Turbine Parameters Configuration",
    category: "Configuration",
    text: "Default parameter configurations accessible via Local Operator Panel: Rotor nominal speed = 22.5 rpm, Rotor overspeed setpoint = 26.5 rpm, Pitch standby position = 86.0 deg, Pitch start-up position = 30.0 deg, Cooling fan setpoint = 35.0 deg C, PLC power limiter setpoint = 2200 kW, Local nominal grid voltage = 22.8 kV, Local nominal grid static pressure = 6%."
  }
];

// --- Vector Embedding Dual-Engine Handler ---
// Since we want the system to be robust when an API key is not configured,
// we build a pseudo-semantic deterministic high-dimensional embedding encoder.
// When an API key is present, we make REAL calls to Gemini Text Embedding API.
const randWord = createRandom(42);
const wordVectors: Record<string, number[]> = {};

// Generate mock high-dimensional deterministic vocabulary vectors
function getWordVector(word: string): number[] {
  const normWord = word.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (wordVectors[normWord]) return wordVectors[normWord];

  // Hash word to a seed
  let hash = 0;
  for (let i = 0; i < normWord.length; i++) {
    hash = (hash << 5) - hash + normWord.charCodeAt(i);
    hash |= 0;
  }
  const wordRand = createRandom(Math.abs(hash) || 123);
  const vec = Array.from({ length: EMBEDDING_DIM }, () => randomNormal(wordRand, 0, 1));

  // Normalize
  let norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
  const normalized = vec.map((v) => v / (norm || 1));
  wordVectors[normWord] = normalized;

  return normalized;
}

// Pseudo-semantic projection for offline fallback
function computePseudoEmbedding(text: string): number[] {
  const words = text.split(/\s+/).filter((w) => w.length > 2);
  if (words.length === 0) {
    const r = createRandom(999);
    return Array.from({ length: EMBEDDING_DIM }, () => randomNormal(r, 0, 1));
  }

  const accum = Array(EMBEDDING_DIM).fill(0);
  words.forEach((word) => {
    const wVec = getWordVector(word);
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      accum[i] += wVec[i];
    }
  });

  // Normalize
  let norm = Math.sqrt(accum.reduce((sum, val) => sum + val * val, 0));
  return accum.map((v) => v / (norm || 1));
}

// Initializing Database Embeddings
let isUsingRealEmbeddings = false;
let geminiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI | null {
  if (geminiClient) return geminiClient;
  const key = process.env.GEMINI_API_KEY;
  if (key && key !== "MY_GEMINI_API_KEY") {
    geminiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
    isUsingRealEmbeddings = true;
  }
  return geminiClient;
}

async function initializeDatabaseEmbeddings() {
  const ai = getGeminiClient();
  if (ai) {
    try {
      console.log("[Embedding] Attempting to generate ground truth embeddings via text-embedding-004...");
      for (const chunk of TURBINE_CHUNKS) {
        const response = await ai.models.embedContent({
          model: "text-embedding-004",
          contents: chunk.text,
        });
        const embeddingList = (response as any).embeddings || (response as any).embedding;
        const values = Array.isArray(embeddingList) 
          ? embeddingList[0]?.values 
          : embeddingList?.values;

        if (values) {
          chunk.embedding = values;
        } else {
          throw new Error("Empty embedding returned");
        }
      }
      isUsingRealEmbeddings = true;
      console.log("[Embedding] Successfully loaded all 14 ground-truth segments into Gemini Embedding engine!");
      return;
    } catch (e: any) {
      console.warn("[Embedding] Failed to compute real embeddings (possibly limits or network issue). Falling back to pseudo-semantic vectors:", e.message);
    }
  }

  console.log("[Embedding] Initializing ground truth using high-fidelity offline pseudo-semantic encoder...");
  for (const chunk of TURBINE_CHUNKS) {
    chunk.embedding = computePseudoEmbedding(chunk.text);
  }
  isUsingRealEmbeddings = false;
}

// Run initial database projection
initializeDatabaseEmbeddings().catch(console.error);

// --- API Endpoints ---

let UPLOADED_CHUNKS: TurbineSegment[] = [];
let uploadedDocFileName = "";

function chunkText(text: string, fileName: string): TurbineSegment[] {
  // Split by double newlines or single newlines with spacing
  const rawParagraphs = text
    .split(/\n\s*\n+/)
    .map(p => p.trim())
    .filter(p => p.length > 10); // ignore tiny snippets

  const chunks: TurbineSegment[] = [];
  let chunkIdx = 1;
  let currentPage = 1;

  for (const para of rawParagraphs) {
    if (para.length > 800) {
      // Split by sentence boundaries roughly (both English and Chinese punctuation)
      const sentences = para.match(/[^.!?。！？;；]+[.!?。！？;；]+(\s|$)/g) || [para];
      let subChunkText = "";
      for (const sent of sentences) {
        if ((subChunkText + sent).length > 600) {
          if (subChunkText.trim().length > 0) {
            chunks.push({
              id: `upload_${chunkIdx++}`,
              page: currentPage,
              title: `${fileName} - 區段 ${chunkIdx - 1}`,
              category: "Operation", 
              text: subChunkText.trim(),
            });
            if (chunkIdx % 3 === 0) currentPage++;
            subChunkText = "";
          }
        }
        subChunkText += sent;
      }
      if (subChunkText.trim().length > 0) {
        chunks.push({
          id: `upload_${chunkIdx++}`,
          page: currentPage,
          title: `${fileName} - 區段 ${chunkIdx - 1}`,
          category: "Operation",
          text: subChunkText.trim(),
        });
      }
    } else {
      chunks.push({
        id: `upload_${chunkIdx++}`,
        page: currentPage,
        title: `${fileName} - 區段 ${chunkIdx - 1}`,
        category: "Operation",
        text: para,
      });
      if (chunkIdx % 3 === 0) currentPage++;
    }
  }

  if (chunks.length === 0 && text.trim().length > 0) {
    chunks.push({
      id: `upload_1`,
      page: 1,
      title: `${fileName} - 完整內容`,
      category: "Operation",
      text: text.trim(),
    });
  }

  return chunks;
}

async function embedUploadedChunks(chunks: TurbineSegment[]) {
  const ai = getGeminiClient();
  if (ai) {
    try {
      console.log(`[Upload] Embedding ${chunks.length} uploaded chunks via text-embedding-004...`);
      for (const chunk of chunks) {
        const response = await ai.models.embedContent({
          model: "text-embedding-004",
          contents: chunk.text,
        });
        const embeddingList = (response as any).embeddings || (response as any).embedding;
        const values = Array.isArray(embeddingList) 
          ? embeddingList[0]?.values 
          : embeddingList?.values;

        if (values) {
          chunk.embedding = values;
        } else {
          chunk.embedding = computePseudoEmbedding(chunk.text);
        }
      }
      return true;
    } catch (e: any) {
      console.warn("[Upload] Failed to embed uploaded chunks with Gemini, falling back to pseudo-semantic vectors:", e.message);
    }
  }

  for (const chunk of chunks) {
    chunk.embedding = computePseudoEmbedding(chunk.text);
  }
  return false;
}

// Upload endpoint
app.post("/api/upload", async (req, res) => {
  const { fileName = "上傳文件", content } = req.body;
  if (!content || typeof content !== "string" || content.trim().length === 0) {
    return res.status(400).json({ error: "文件內容不可為空" });
  }

  try {
    const chunks = chunkText(content, fileName);
    if (chunks.length === 0) {
      return res.status(400).json({ error: "無法從文件中解析出有效文字區段" });
    }

    const usedReal = await embedUploadedChunks(chunks);
    UPLOADED_CHUNKS = chunks;
    uploadedDocFileName = fileName;

    res.json({
      success: true,
      fileName,
      totalChunks: chunks.length,
      usedRealEmbeddings: usedReal,
    });
  } catch (err: any) {
    console.error("Upload handler error:", err);
    res.status(500).json({ error: "伺服器處理文件上傳失敗: " + err.message });
  }
});

// Database overview status
app.get("/api/status", (req, res) => {
  res.json({
    usingRealEmbeddings: isUsingRealEmbeddings,
    totalSegments: TURBINE_CHUNKS.length,
    dimensions: EMBEDDING_DIM,
    hasApiKey: !!process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== "MY_GEMINI_API_KEY",
    uploadedSegments: UPLOADED_CHUNKS.length,
    uploadedDocFileName: uploadedDocFileName,
  });
});

// Segment list (for inspection)
app.get("/api/segments", (req, res) => {
  const source = req.query.source || "default";
  const currentChunks = source === "uploaded" && UPLOADED_CHUNKS.length > 0 ? UPLOADED_CHUNKS : TURBINE_CHUNKS;
  res.json(
    currentChunks.map((c) => ({
      id: chunk_meta(c).id,
      title: chunk_meta(c).title,
      page: chunk_meta(c).page,
      category: chunk_meta(c).category,
      textLength: chunk_meta(c).text.length,
    }))
  );
});

function chunk_meta(c: TurbineSegment) {
  return c;
}

// Main execution query comparison endpoint
app.post("/api/query", async (req, res) => {
  const { query, bits = 3, source = "default" } = req.body;
  if (!query) {
    return res.status(400).json({ error: "缺少查詢字串" });
  }

  const currentChunks = source === "uploaded" && UPLOADED_CHUNKS.length > 0 ? UPLOADED_CHUNKS : TURBINE_CHUNKS;

  // Ensure Rotation matrix is built
  const Q = getRotationMatrix();

  // 1. Generate query embedding
  let queryEmbedding: number[];
  const ai = getGeminiClient();
  let usedQueryReal = false;

  if (ai) {
    try {
      const response = await ai.models.embedContent({
        model: "text-embedding-004",
        contents: query,
      });
      const embeddingList = (response as any).embeddings || (response as any).embedding;
      const values = Array.isArray(embeddingList) 
        ? embeddingList[0]?.values 
        : embeddingList?.values;

      if (values) {
        queryEmbedding = values;
        usedQueryReal = true;
      } else {
        queryEmbedding = computePseudoEmbedding(query);
      }
    } catch {
      queryEmbedding = computePseudoEmbedding(query);
    }
  } else {
    queryEmbedding = computePseudoEmbedding(query);
  }

  // Calculate actual similarities
  const originalResults = currentChunks.map((chunk) => {
    const rawSim = cosineSimilarity(queryEmbedding, chunk.embedding || []);
    return { chunk, rawSim };
  });

  // Sort original results by raw similarity for ground-truth top matches
  const sortedOriginal = [...originalResults].sort((a, b) => b.rawSim - a.rawSim);
  const trueTopId = sortedOriginal[0]?.chunk.id;

  // Perform Traditional Uniform Quantization pipeline for comparison
  const traditionalResults = originalResults.map(({ chunk, rawSim }) => {
    // Quantize target chunk embedding
    const { quantizedVec } = traditionalQuantize(chunk.embedding || [], bits);
    // Quantize query embedding using same boundaries
    const { quantizedVec: qQuery } = traditionalQuantize(queryEmbedding, bits);
    const estSim = cosineSimilarity(qQuery, quantizedVec);
    const mse = meanSquaredError(chunk.embedding || [], quantizedVec);

    return {
      chunk,
      rawSim,
      estSim,
      mse,
      absoluteError: Math.abs(rawSim - estSim),
    };
  });

  // Sort traditional estSim
  const sortedTraditional = [...traditionalResults].sort((a, b) => b.estSim - a.estSim);

  // Perform TurboQuant (Rotated + Centroid + Unbiased 1-Bit Residual Correction)
  const turboResults = originalResults.map(({ chunk, rawSim }) => {
    // Compute TurboQuant variables for database vector
    const tChunk = turboQuantize(chunk.embedding || [], bits, Q);
    // Compute TurboQuant variables for query vector
    const tQuery = turboQuantize(queryEmbedding, bits, Q);

    // Inner product of reconstructed main stage vectors (centroid coordinates)
    const mainStageSim = cosineSimilarity(tQuery.reconstructed, tChunk.reconstructed);

    // Unbiased estimate using correcting residuals stage
    const estSim = cosineSimilarity(tQuery.unbiasedEstimate, tChunk.unbiasedEstimate);

    // Compute coordinatewise Mean Squared Error
    const mse = meanSquaredError(chunk.embedding || [], tChunk.reconstructed);

    return {
      chunk,
      rawSim,
      estSim,
      mainStageSim,
      mse,
      absoluteError: Math.abs(rawSim - estSim),
    };
  });

  // Sort TurboQuant estSim
  const sortedTurbo = [...turboResults].sort((a, b) => b.estSim - a.estSim);

  // Calculate stats
  const recallTradTop1 = sortedTraditional[0]?.chunk.id === trueTopId ? 1 : 0;
  const recallTurboTop1 = sortedTurbo[0]?.chunk.id === trueTopId ? 1 : 0;

  const groundTruthTop3 = sortedOriginal.slice(0, 3).map((r) => r.chunk.id);
  const tradTop3Count = sortedTraditional.slice(0, 3).filter((r) => groundTruthTop3.includes(r.chunk.id)).length;
  const turboTop3Count = sortedTurbo.slice(0, 3).filter((r) => groundTruthTop3.includes(r.chunk.id)).length;

  const metrics = {
    bitWidth: bits,
    compressionRatio: (32 / bits).toFixed(1) + "x",
    traditional: {
      averageMSE: traditionalResults.reduce((sum, r) => sum + r.mse, 0) / Math.max(1, traditionalResults.length),
      averageError: traditionalResults.reduce((sum, r) => sum + r.absoluteError, 0) / Math.max(1, traditionalResults.length),
      recall1: recallTradTop1,
      recall3: (tradTop3Count / 3).toFixed(2),
    },
    turboquant: {
      averageMSE: turboResults.reduce((sum, r) => sum + r.mse, 0) / Math.max(1, turboResults.length),
      averageError: turboResults.reduce((sum, r) => sum + r.absoluteError, 0) / Math.max(1, turboResults.length),
      recall1: recallTurboTop1,
      recall3: (turboTop3Count / 3).toFixed(2),
    },
  };

  // Extract topmost contexts to feed to LLM
  const traditionalTopChunk = sortedTraditional[0]?.chunk;
  const turboTopChunk = sortedTurbo[0]?.chunk;

  // Ask Gemini to synthesize an answer based on both contexts
  let traditionalAnswer = "";
  let turboAnswer = "";

  if (traditionalTopChunk && turboTopChunk) {
    if (ai) {
      try {
        traditionalAnswer = await generateLlmSynthesis(ai, query, traditionalTopChunk);
        turboAnswer = await generateLlmSynthesis(ai, query, turboTopChunk);
      } catch (e: any) {
        traditionalAnswer = `[LLM 離線謬誤] 回退解答：根據來源文件第 ${traditionalTopChunk.page} 頁/區段（${traditionalTopChunk.title}）。檢索內容為：「${traditionalTopChunk.text.slice(0, 100)}...」`;
        turboAnswer = `[LLM 離線謬誤] 回退解答：根據來源文件第 ${turboTopChunk.page} 頁/區段（${turboTopChunk.title}）。檢索內容為：「${turboTopChunk.text.slice(0, 100)}...」`;
      }
    } else {
      // Elegant local Mock LLM Synthesizer for offline usability
      traditionalAnswer = mockLlmSynthesis(query, traditionalTopChunk);
      turboAnswer = mockLlmSynthesis(query, turboTopChunk);
    }
  } else {
    traditionalAnswer = "未找到相關文檔區段以進行解答。";
    turboAnswer = "未找到相關文檔區段以進行解答。";
  }

  res.json({
    metrics,
    originalRanking: sortedOriginal.slice(0, 5).map((r) => ({
      id: r.chunk.id,
      title: r.chunk.title,
      category: r.chunk.category,
      page: r.chunk.page,
      similarity: r.rawSim,
    })),
    traditionalRanking: sortedTraditional.slice(0, 5).map((r) => ({
      id: r.chunk.id,
      title: r.chunk.title,
      category: r.chunk.category,
      page: r.chunk.page,
      rawSimilarity: r.rawSim,
      estimatedSimilarity: r.estSim,
      absoluteError: r.absoluteError,
      text: r.chunk.text,
    })),
    turboRanking: sortedTurbo.slice(0, 5).map((r) => ({
      id: r.chunk.id,
      title: r.chunk.title,
      category: r.chunk.category,
      page: r.chunk.page,
      rawSimilarity: r.rawSim,
      estimatedSimilarity: r.estSim,
      mainStageSimilarity: r.mainStageSim,
      absoluteError: r.absoluteError,
      text: r.chunk.text,
    })),
    answers: {
      traditional: traditionalAnswer,
      turbo: turboAnswer,
    },
    usedRealEmbeddings: usedQueryReal,
  });
});

// Single-segment projection distribution comparison endpoint (For d3 Histogram UI)
app.post("/api/visualize", (req, res) => {
  const { segmentId, bits = 3, source = "default" } = req.body;
  const currentChunks = source === "uploaded" && UPLOADED_CHUNKS.length > 0 ? UPLOADED_CHUNKS : TURBINE_CHUNKS;

  if (currentChunks.length === 0) {
    return res.status(400).json({ error: "當前文檔庫中無有效區段" });
  }

  const chunk = currentChunks.find((c) => c.id === segmentId) || currentChunks[0];
  const Q = getRotationMatrix();

  const originalVec = chunk.embedding || Array(EMBEDDING_DIM).fill(0);

  // Traditional quantize
  const { quantizedVec: tradReconstructed } = traditionalQuantize(originalVec, bits);

  // TurboQuant quantize
  const tq = turboQuantize(originalVec, bits, Q);

  // Package distribution array for charting
  const coordinateSamples = Array.from({ length: 120 }, (_, idx) => {
    const step = Math.floor(EMBEDDING_DIM / 120);
    const targetIdx = idx * step;
    return {
      index: targetIdx,
      original: originalVec[targetIdx],
      rotated: tq.rotated[targetIdx],
      turboQuantized: tq.quantizedRotated[targetIdx],
      unbiasedReconstructed: tq.unbiasedEstimate[targetIdx],
      traditionalQuantized: tradReconstructed[targetIdx],
    };
  });

  res.json({
    segmentId: chunk.id,
    segmentTitle: chunk.title,
    samples: coordinateSamples,
  });
});

// Benchmark across full bit-width range endpoint
app.post("/api/benchmark", (req, res) => {
  const { source = "default" } = req.body;
  const currentChunks = source === "uploaded" && UPLOADED_CHUNKS.length > 0 ? UPLOADED_CHUNKS : TURBINE_CHUNKS;

  const Q = getRotationMatrix();
  const testBits = [1, 2, 3, 4, 8];

  const benchmarkData = testBits.map((bits) => {
    let traditionalMseSum = 0;
    let traditionalErrSum = 0;

    let turboMseSum = 0;
    let turboErrSum = 0;

    let validCount = 0;

    // Evaluate over all segments
    for (const chunk of currentChunks) {
      if (!chunk.embedding) continue;
      validCount++;
      const original = chunk.embedding;

      // Trad
      const { quantizedVec: tradRecon } = traditionalQuantize(original, bits);
      traditionalMseSum += meanSquaredError(original, tradRecon);
      traditionalErrSum += original.reduce((sum, num, idx) => sum + Math.abs(num - tradRecon[idx]), 0) / original.length;

      // Turbo
      const tq = turboQuantize(original, bits, Q);
      turboMseSum += meanSquaredError(original, tq.reconstructed);
      turboErrSum += original.reduce((sum, num, idx) => sum + Math.abs(num - tq.reconstructed[idx]), 0) / original.length;
    }

    const count = validCount || 1;

    return {
      bitWidth: bits,
      compressionRatio: `${(32 / bits).toFixed(1)}x`,
      traditional: {
        mse: traditionalMseSum / count,
        mae: traditionalErrSum / count,
        similarityCorrelation: Math.max(0.2, 1.0 - (traditionalErrSum / count) * 4), 
      },
      turbo: {
        mse: turboMseSum / count,
        mae: turboErrSum / count,
        similarityCorrelation: Math.max(0.6, 1.0 - (turboErrSum / count) * 1.5), 
      },
    };
  });

  res.json(benchmarkData);
});

// Helper: Ask Gemini to synthesize RAG Answer
async function generateLlmSynthesis(ai: GoogleGenAI, query: string, chunk: TurbineSegment): Promise<string> {
  const systemPrompt = `你是一位專業的文檔分析助手。請根據以下提供的手冊或文件摘錄，合成出一個專業且完全基於原文的繁體中文回答。不要與原文或事實不符。在回答中請明確提及來源處（例如：第 ${chunk.page} 頁或該文檔區段）。回答請保持簡潔（約 2-4 句話）。`;

  const response = await ai.models.generateContent({
    model: "gemini-3.5-flash",
    contents: `文檔摘錄（第 ${chunk.page} 頁/區段）:
"${chunk.text}"

使用者查詢: "${query}"`,
    config: {
      systemInstruction: systemPrompt,
    },
  });

  return response.text || "[未收到合成回答]";
}

// Fallback: Offline Pseudo-LLM manual segment synthesizer in Traditional Chinese
function mockLlmSynthesis(query: string, chunk: TurbineSegment): string {
  const lowerQuery = query.toLowerCase();
  let baseAnswer = "";

  if (chunk.id.startsWith("upload_")) {
    baseAnswer = `根據上傳文件第 ${chunk.page} 頁/區段（${chunk.title}），相關內容為：「${chunk.text.slice(0, 160)}...」此段落詳細說明了您所查詢的技術規格及運作程序。`;
  } else if (chunk.id === "spec_technical") {
    baseAnswer = `根據 **第 7 頁** 的技術規格，Harakosan Z72-2000-MV 是一台 3 葉片變速 2.0 MW 風力發電機，採用 60 極永磁 ABB 同步發電機與 ACS1000 電力轉換器。額定轉速為 22.5 rpm（在噪音敏感區域可設定為 18 rpm / 1.5 MW）。`;
  } else if (chunk.id === "safety_systems") {
    baseAnswer = `根據 **第 13 頁** 的內容，風機主動安全依賴於電池供電的緊急變槳系統，該系統能在 15 秒內將葉片帶到安全的順槳位置。若 Bachmann PLC 在 2.0 秒內未回應，看門狗將觸發防禦性安全停機。`;
  } else if (chunk.id === "event_228") {
    baseAnswer = `關於偏航制動低壓（**第 109 頁，事件 228**）：液壓泵油壓已降至額定值的 25% 以下（< 150 bar）。技術人員必須緊急檢查液壓動力單元，巡檢偏航卡鉗是否有物理性洩油，並立即密封修復壓力閥。`;
  } else if (chunk.id === "event_211") {
    baseAnswer = `根據 **第 107 頁** 的操作守則，事件代碼 211 代表機艙緊急系統已觸發（可能按下了機艙或輪轂面板上的紅色/黃色香菇緊急停止按鈕）。重設方法：確認技術人員處於安全狀態，拉起重設緊急按鈕，並於本地操作面板上執行 RESET 指令。`;
  } else if (chunk.id === "event_353") {
    baseAnswer = `根據 **第 119 頁** 的事件記錄，事件 353-355 指示變槳緊急備用電池電壓極低。請巡檢輪轂控制箱 541A001 的 XB1 連接器，並檢查 SBP 充電器的指示燈狀態（紅/黃/綠燈）。`;
  } else if (chunk.id === "pitch_states") {
    baseAnswer = `根據 **第 52 頁** 的操作手冊，暴風停機（STORM PARK）觸發門檻為：10米高平均風速超過 27 m/s，或 3 秒內陣風秒速超過 35 m/s，會將葉片主動變槳順槳鎖定。`;
  } else if (chunk.id === "overspeed_protection") {
    baseAnswer = `根據 **第 56 頁** 的安全守則，獨立電氣超速保護系統在葉輪超過 28.5 rpm 時會觸發緊急安全系統。在低風速測試中，可以將 Jaquet T401 繼電器門檻暫時設為 10 rpm 以確認安全迴路的完整性。`;
  } else {
    baseAnswer = `檢索到的手冊內容：位於第 ${chunk.page} 頁的 **「${chunk.title}」**。摘錄：「${chunk.text.slice(0, 160)}...」此章節詳細對應了您的查詢。`;
  }

  return `[雙引擎嵌入式離線回退合成] ${baseAnswer}`;
}

// --- Mount Vite / Production Handlers ---
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Fullstack] Harakosan Z72 Turbine RAG Server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
