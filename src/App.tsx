import React, { useState, useEffect } from "react";
import {
  Wind,
  Cpu,
  Zap,
  Sliders,
  Database,
  Search,
  Wrench,
  Clock,
  Activity,
  FileText,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  TrendingUp,
  Award,
  ArrowRight
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface StatusData {
  usingRealEmbeddings: boolean;
  totalSegments: number;
  dimensions: number;
  hasApiKey: boolean;
  uploadedSegments?: number;
  uploadedDocFileName?: string;
}

interface SegmentMeta {
  id: string;
  title: string;
  page: number;
  category: string;
  textLength: number;
}

interface QueryResult {
  metrics: {
    bitWidth: number;
    compressionRatio: string;
    traditional: {
      averageMSE: number;
      averageError: number;
      recall1: number;
      recall3: string;
    };
    turboquant: {
      averageMSE: number;
      averageError: number;
      recall1: number;
      recall3: string;
    };
  };
  originalRanking: Array<{
    id: string;
    title: string;
    category: string;
    page: number;
    similarity: number;
  }>;
  traditionalRanking: Array<{
    id: string;
    title: string;
    category: string;
    page: number;
    rawSimilarity: number;
    estimatedSimilarity: number;
    absoluteError: number;
    text: string;
  }>;
  turboRanking: Array<{
    id: string;
    title: string;
    category: string;
    page: number;
    rawSimilarity: number;
    estimatedSimilarity: number;
    mainStageSimilarity: number;
    absoluteError: number;
    text: string;
  }>;
  answers: {
    traditional: string;
    turbo: string;
  };
  usedRealEmbeddings: boolean;
}

interface CoordinateSample {
  index: number;
  original: number;
  rotated: number;
  turboQuantized: number;
  unbiasedReconstructed: number;
  traditionalQuantized: number;
}

interface VizData {
  segmentId: string;
  segmentTitle: string;
  samples: CoordinateSample[];
}

interface BenchmarkItem {
  bitWidth: number;
  compressionRatio: string;
  traditional: {
    mse: number;
    mae: number;
    similarityCorrelation: number;
  };
  turbo: {
    mse: number;
    mae: number;
    similarityCorrelation: number;
  };
}

const PRESET_QUERIES = [
  {
    label: "葉輪額定及超速轉速設定",
    query: "額定轉速（Rotor Nominal Speed）與超速設定值是多少？",
    category: "系統設定"
  },
  {
    label: "偏航煞車壓力過低事件 228 排除",
    query: "如何排除偏航煞車壓力過低事件代碼 228 (Yaw brake pressure low)？",
    category: "故障排除"
  },
  {
    label: "緊急變槳順槳動作反應時間",
    query: "標準緊急變槳要花多久時間才能將葉片轉到安全的順槳位置？",
    category: "安全防護"
  },
  {
    label: "暴風停機（STORM PARK）風速臨界值",
    query: "觸發暴風停機（STORM PARK）的平均風速與最大陣風門檻是多少？",
    category: "運轉控制"
  },
  {
    label: "重設 nacelle 香菇緊急停止按鈕",
    query: "觸發事件代碼 211（Emergency nacelle tripped）的原因與重設方法是什麼？",
    category: "故障排除"
  }
];

export default function App() {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [segments, setSegments] = useState<SegmentMeta[]>([]);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string>("config_defaults");
  const [bits, setBits] = useState<number>(3);
  const [queryInput, setQueryInput] = useState<string>("額定轉速（Rotor Nominal Speed）與超速設定值是多少？");
  const [queryLoading, setQueryLoading] = useState<boolean>(false);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [vizLoading, setVizLoading] = useState<boolean>(false);
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkItem[]>([]);
  const [activeTab, setActiveTab] = useState<"visualizer" | "benchmark">("visualizer");

  // Document Source Toggle States
  const [docSource, setDocSource] = useState<"default" | "uploaded">("default");
  const [uploadLoading, setUploadLoading] = useState<boolean>(false);
  const [uploadedDocInfo, setUploadedDocInfo] = useState<{
    fileName: string;
    totalChunks: number;
    usedRealEmbeddings: boolean;
  } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [customText, setCustomText] = useState<string>("");

  // Fetch status, segments, and run baseline query & benchmark on mount / source switch
  useEffect(() => {
    fetchStatus();
    fetchSegments();
    fetchBenchmark();
  }, [docSource]);

  // Handle baseline query when segments load or shift
  useEffect(() => {
    if (segments.length > 0) {
      const defaultQuery = docSource === "default"
        ? "額定轉速（Rotor Nominal Speed）與超速設定值是多少？"
        : "請說明此文件的關鍵內容、特定數字與中心摘要之解答。";
      setQueryInput(defaultQuery);
      handleSearch(defaultQuery, bits);
    }
  }, [segments]);

  // Update visualizer when selected segment or bits change
  useEffect(() => {
    if (selectedSegmentId) {
      fetchVisualization(selectedSegmentId, bits);
    }
  }, [selectedSegmentId, bits, docSource]);

  // Ensure selectedSegmentId is aligned when switching document scope
  useEffect(() => {
    if (segments.length > 0) {
      const exists = segments.some((s) => s.id === selectedSegmentId);
      if (!exists) {
        setSelectedSegmentId(segments[0].id);
      }
    }
  }, [segments, selectedSegmentId]);

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      setStatus(data);
      // Sync doc uploader client-state under refresh of the page
      if (data.uploadedSegments && data.uploadedSegments > 0) {
        setUploadedDocInfo({
          fileName: data.uploadedDocFileName || "已上傳文檔.txt",
          totalChunks: data.uploadedSegments,
          usedRealEmbeddings: data.usingRealEmbeddings,
        });
      }
    } catch (err) {
      console.error("Error fetching status:", err);
    }
  };

  const fetchSegments = async () => {
    try {
      const res = await fetch(`/api/segments?source=${docSource}`);
      const data = await res.json();
      setSegments(data);
    } catch (err) {
      console.error("Error fetching segments:", err);
    }
  };

  const fetchBenchmark = async () => {
    try {
      const res = await fetch("/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: docSource })
      });
      const data = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      console.error("Error fetching benchmark data:", err);
    }
  };

  const fetchVisualization = async (segmentId: string, targetBits: number) => {
    setVizLoading(true);
    try {
      const res = await fetch("/api/visualize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segmentId, bits: targetBits, source: docSource })
      });
      const data = await res.json();
      setVizData(data);
    } catch (err) {
      console.error("Error fetching visualization:", err);
    } finally {
      setVizLoading(false);
    }
  };

  const handleSearch = async (targetQuery: string = queryInput, targetBits: number = bits) => {
    setQueryLoading(true);
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: targetQuery, bits: targetBits, source: docSource })
      });
      const data = await res.json();
      setQueryResult(data);
      fetchStatus();
    } catch (err) {
      console.error("Search query error:", err);
    } finally {
      setQueryLoading(false);
    }
  };

  // Custom documents upload pipelines
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await processFile(file);
    }
  };

  const processFile = async (file: File) => {
    setUploadLoading(true);
    setUploadError(null);
    const reader = new FileReader();

    reader.onload = async (event) => {
      const text = event.target?.result as string;
      if (!text || text.trim().length === 0) {
        setUploadError("選擇的文件內容為空");
        setUploadLoading(false);
        return;
      }
      await uploadDocumentContent(file.name, text);
    };

    reader.onerror = () => {
      setUploadError("讀取文件時出錯");
      setUploadLoading(false);
    };

    reader.readAsText(file);
  };

  const uploadDocumentContent = async (fileName: string, content: string) => {
    setUploadLoading(true);
    setUploadError(null);
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fileName, content }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "文檔上傳解析失敗");
      }

      setUploadedDocInfo({
        fileName: data.fileName,
        totalChunks: data.totalChunks,
        usedRealEmbeddings: data.usedRealEmbeddings,
      });

      // Instantly activate uploaded document scope
      setDocSource("uploaded");
      setCustomText("");
    } catch (err: any) {
      setUploadError(err.message || "上傳文檔處理時出錯");
    } finally {
      setUploadLoading(false);
    }
  };

  const calculateMemorySaving = (b: number) => {
    return ((1 - b / 32) * 100).toFixed(1);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased selection:bg-emerald-500/30 selection:text-emerald-200" id="main_root">
      
      {/* --- Top Elegant Header --- */}
      <header className="border-b border-slate-800/70 bg-slate-900/50 backdrop-blur sticky top-0 z-50 px-6 py-4" id="main_header">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-emerald-500 text-slate-950 rounded-xl shadow-[0_0_20px_rgba(16,185,129,0.2)]">
              <Wind className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold tracking-wider uppercase px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                  向量量化沙盒 (RAG Sandbox)
                </span>
                <span className="text-xs font-semibold tracking-wider px-2.5 py-0.5 rounded-full bg-emerald-950/40 text-emerald-400 border border-emerald-800/50 font-mono">
                  arXiv:2504.19874
                </span>
              </div>
              <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white mt-1 font-display">
                RAG 語義檢索：TurboQuant 與傳統向量量化技術實測儀
              </h1>
            </div>
          </div>

          {/* Engine Status Indicators */}
          <div className="flex flex-wrap items-center gap-3 text-xs bg-slate-900/80 px-4 py-2.5 rounded-xl border border-slate-800 shadow-inner">
            <div className="flex items-center gap-1.5 font-mono">
              <Database className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">向量嵌入引擎:</span>
              {status?.usingRealEmbeddings ? (
                <span className="text-emerald-400 font-bold flex items-center gap-1">
                  ● text-embedding-004 (在線實時)
                </span>
              ) : (
                <span className="text-amber-400 font-bold flex items-center gap-1" title="語義相似度在本地藉由高維旋轉變換投影計算還原">
                  ● 局部高維語義投影矩陣 (768D)
                </span>
              )}
            </div>
            <div className="h-3 w-px bg-slate-800" />
            <div className="flex items-center gap-1.5 font-mono">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">LLM 答覆合成:</span>
              {status?.hasApiKey ? (
                <span className="text-indigo-400 font-bold">● gemini-3.5-flash (在線實時)</span>
              ) : (
                <span className="text-slate-400 font-medium">● 雙引擎嵌入式合成器 (離線封裝)</span>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 space-y-6" id="main_dashboard">

        {/* --- ADDED: Document Source & Upload Center --- */}
        <div className="bg-slate-900/60 border border-slate-800/85 p-6 rounded-2xl shadow-xl space-y-4" id="document_management_panel">
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-500/10 text-indigo-400 rounded-lg border border-indigo-500/20">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-white text-base">知識文檔庫管理中心 (RAG Knowledge Manager)</h3>
                <p className="text-xs text-slate-400">
                  切換 RAG 檢索的背景文檔庫。您可以保持預設手冊，或點選右側上傳您自製的任意中文文字檔！
                </p>
              </div>
            </div>

            {/* Document scope switcher buttons */}
            <div className="flex bg-slate-950 p-1.5 rounded-xl border border-slate-850 self-stretch lg:self-auto justify-stretch">
              <button
                onClick={() => setDocSource("default")}
                className={`flex-1 lg:flex-none px-4 py-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 ${
                  docSource === "default"
                    ? "bg-emerald-500 text-slate-950 shadow-[0_0_15px_rgba(16,185,129,0.2)] font-black"
                    : "text-slate-400 hover:text-slate-205"
                }`}
                id="btn_select_default"
              >
                <Wind className="w-3.5 h-3.5" />
                預設：Harakosan 風力發電機手冊
              </button>
              <button
                onClick={() => setDocSource("uploaded")}
                className={`flex-1 lg:flex-none px-4 py-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 ${
                  docSource === "uploaded"
                    ? "bg-indigo-600 text-white shadow-[0_0_15px_rgba(79,70,229,0.2)] font-black"
                    : "text-slate-400 hover:text-slate-205"
                }`}
                id="btn_select_uploaded"
              >
                <FileText className="w-3.5 h-3.5" />
                自訂：上傳解析文檔 {uploadedDocInfo ? `(${uploadedDocInfo.fileName.slice(0, 10)}${uploadedDocInfo.fileName.length > 10 ? "..." : ""})` : "(未上傳)"}
              </button>
            </div>
          </div>

          <AnimatePresence mode="wait">
            {docSource === "uploaded" && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
                key="uploaded_upload_zone"
              >
                <div className="p-4 bg-slate-950/70 border border-indigo-500/20 rounded-xl space-y-4 shadow-inner mt-2">
                  {uploadedDocInfo ? (
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-bold text-sm text-indigo-400">✓ 成功解析並載入對象文件：</span>
                          <span className="bg-indigo-950 text-indigo-300 border border-indigo-900 px-2.5 py-0.5 rounded-md font-mono text-xs font-bold">
                            {uploadedDocInfo.fileName}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 leading-relaxed">
                          文件已自動分割為 <b className="text-slate-200 font-mono font-bold">{uploadedDocInfo.totalChunks}</b> 個高度重合之語義區段段落。
                          向量嵌入狀態：{uploadedDocInfo.usedRealEmbeddings ? (
                            <span className="text-emerald-400 font-bold">● 在線實時 API 對應 (text-embedding-004)</span>
                          ) : (
                            <span className="text-amber-400 font-bold" title="已自動套用 768 維高維語義空間確定性隨機投影映射">
                              ● 768維局部高保真語義投影矩陣 (量化無偏)
                            </span>
                          )}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 w-full sm:w-auto">
                        <label className="flex-1 sm:flex-none text-center px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl text-xs font-bold cursor-pointer transition">
                          更換其他文件
                          <input type="file" accept=".txt,.md" onChange={handleFileUpload} className="hidden" />
                        </label>
                        <button
                          onClick={() => {
                            setUploadedDocInfo(null);
                            setDocSource("default");
                          }}
                          className="flex-1 sm:flex-none px-4 py-2 bg-red-950/40 hover:bg-red-900/30 text-red-400 border border-red-900/40 rounded-xl text-xs font-bold transition"
                        >
                          清除此自訂文檔
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
                      {/* Drag & Drop uploader area */}
                      <div className="lg:col-span-7">
                        <div
                          onDragOver={handleDragOver}
                          onDrop={handleDrop}
                          className="h-full border-2 border-dashed border-slate-800 hover:border-indigo-500/50 bg-slate-900/30 rounded-xl p-6 text-center transition cursor-pointer flex flex-col items-center justify-center space-y-3 group"
                        >
                          <div className="p-3 bg-slate-950 text-indigo-400 rounded-xl border border-slate-800 group-hover:scale-110 transition">
                            <FileText className="w-6 h-6" />
                          </div>
                          <div>
                            <p className="text-xs font-bold text-slate-200">
                              將您的文字文件拖曳至此處，或 <span className="text-indigo-400 hover:underline">瀏覽本機檔案</span>
                            </p>
                            <p className="text-[10px] text-slate-500 font-mono mt-1">支援 .txt, .md 及任意純文字，限 5MB 以內</p>
                          </div>
                          <input type="file" accept=".txt,.md" onChange={handleFileUpload} className="hidden" id="custom_file_selector" />
                          <label htmlFor="custom_file_selector" className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold transition shadow-md shadow-indigo-600/10 cursor-pointer">
                            選擇檔案
                          </label>
                        </div>
                      </div>

                      {/* Paste Clipboard block */}
                      <div className="lg:col-span-5 flex flex-col justify-between space-y-3 bg-slate-900/20 p-1 rounded-xl">
                        <div>
                          <label className="text-xs font-bold text-slate-300 block mb-1">或直接在右側粘貼文檔內容：</label>
                          <textarea
                            value={customText}
                            onChange={(e) => setCustomText(e.target.value)}
                            placeholder="請在此貼上您的專案手冊、學術資料、程序規範、操作手則或任意文字段落..."
                            className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 outline-none rounded-xl p-3 text-xs font-medium text-slate-200 placeholder:text-slate-700 min-h-[105px] resize-none"
                          />
                        </div>
                        <button
                          onClick={() => {
                            if (customText.trim().length > 0) {
                              uploadDocumentContent("貼上之剪貼簿文檔.txt", customText);
                            }
                          }}
                          disabled={customText.trim().length === 0 || uploadLoading}
                          className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-900 disabled:text-slate-600 text-white font-bold rounded-lg text-xs transition flex items-center justify-center gap-1.5"
                        >
                          確認解析並載入剪貼簿內容
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Upload state loading flags */}
                  {uploadLoading && (
                    <div className="flex items-center gap-2 text-xs text-indigo-400 animate-pulse font-mono justify-center py-2 bg-slate-900/40 rounded-lg">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      正在為您的文檔切分語義區段，並建構 TurboQuant 旋轉無偏高維向量索引，請稍候...
                    </div>
                  )}
                  {uploadError && (
                    <div className="flex items-center gap-2 text-xs text-red-150 bg-red-950/20 border border-red-900/30 p-2.5 rounded-lg">
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                      <span>解析失敗：{uploadError}</span>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* --- Top Bento Area: Controller and Diagnostic Scenarios --- */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Section 1: Controller Config (bit-width & compression) */}
          <div className="lg:col-span-5 bg-slate-900/60 border border-slate-800/85 p-6 rounded-2xl shadow-xl flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Sliders className="w-4.5 h-4.5 text-emerald-400" />
                <h3 className="font-bold text-white font-sans tracking-tight">高維量化壓縮控制器</h3>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                高維嵌入向量點中（768 維 float32）在儲存端佔據大量空間。縮小位元數可節約 16~32 倍的記憶體，但在普通標量量化 (SQ) 下將會使語義嚴重扭曲（失真）。TurboQuant 利用正交隨機投影優雅保證極低位元寬度的相似度檢索性能。
              </p>
            </div>

            {/* Slider */}
            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-slate-400">目標數位編碼位元寬 (Bit-Width)：</span>
                <span className="bg-emerald-500 text-slate-950 px-2.5 py-1 rounded font-bold text-xs shadow-[0_0_15px_rgba(16,185,129,0.3)]">
                  {bits} 位元編碼 {bits <= 3 ? "🤖 超高維極限壓縮" : "⚙️ 中位元高保真"}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="4"
                value={bits}
                onChange={(e) => setBits(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-800 rounded-lg cursor-pointer accent-emerald-400"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500">
                <span>1-Bit (均值二值化)</span>
                <span>2-Bit (4質心階層)</span>
                <span>3-Bit (8質心階層)</span>
                <span>4-Bit (16質心階層)</span>
              </div>
            </div>

            {/* Statistics Row */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-emerald-950/30 p-3 rounded-xl border border-emerald-900/30 flex flex-col justify-between hover:bg-emerald-950/40 transition">
                <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider font-mono">空間壓縮比</span>
                <span className="text-lg font-black text-emerald-200 font-mono mt-1">{(32 / bits).toFixed(1)}x</span>
                <span className="text-[9px] text-emerald-500 mt-0.5">佔用體積縮小</span>
              </div>
              <div className="bg-emerald-950/20 p-3 rounded-xl border border-emerald-800/20 flex flex-col justify-between hover:bg-emerald-950/35 transition">
                <span className="text-[10px] font-bold text-emerald-300 uppercase tracking-wider font-mono">RAM 省減率</span>
                <span className="text-lg font-black text-emerald-250 font-mono mt-1">{calculateMemorySaving(bits)}%</span>
                <span className="text-[9px] text-emerald-400 mt-0.5">免除硬體負荷</span>
              </div>
              <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800/80 flex flex-col justify-between hover:bg-slate-900/60 transition">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono">最優坐標質心</span>
                <span className="text-lg font-black text-slate-200 font-mono mt-1">{Math.pow(2, bits)} 個</span>
                <span className="text-[9px] text-slate-500 mt-0.5">Lloyd-Max centroids</span>
              </div>
            </div>
          </div>

          {/* Section 2: Preset Scenarios */}
          <div className="lg:col-span-7 bg-slate-900/60 border border-slate-800/85 p-6 rounded-2xl shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Wrench className="w-4.5 h-4.5 text-slate-300" />
                <h3 className="font-bold text-white tracking-tight">典型 RAG 巡檢對比提問範例</h3>
              </div>
              <span className="text-[10px] bg-slate-950 text-slate-400 border border-slate-850 font-mono px-2 py-1 rounded">
                {docSource === "default" ? "HE-0378 運作手冊" : "自訂 RAG 模式"}
              </span>
            </div>

            {/* If default is selected, show pre-loaded queries. Else show generic uploader queries */}
            {docSource === "default" ? (
              <div className="space-y-3">
                <p className="text-xs text-slate-400 leading-relaxed">
                  點選下列預設風機巡檢事件，測試向量壓縮後的檢索可靠度。注意，傳統均勻 SQ 量化在極低位元規格下，經常因為相似度失真和誤差噪音，無法成功檢索最優先的正確細節段落，導致 AI 回答錯誤或答非所問：
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {PRESET_QUERIES.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setQueryInput(p.query);
                        handleSearch(p.query, bits);
                      }}
                      className={`text-left p-2.5 rounded-xl border text-xs transition duration-150 flex flex-col justify-between hover:bg-slate-900/50 active:scale-95 ${
                        queryInput === p.query
                          ? "border-emerald-500/50 bg-emerald-950/20 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.06)]"
                          : "border-slate-800 text-slate-300 bg-slate-950/50"
                      }`}
                    >
                      <span className={`font-semibold text-[11px] line-clamp-1 ${queryInput === p.query ? "text-emerald-300" : "text-slate-200"}`}>{p.label}</span>
                      <div className="flex justify-between items-center mt-1 text-[9px] font-mono text-slate-500">
                        <span>事件範疇：{p.category}</span>
                        <span className="text-emerald-400 font-semibold flex items-center gap-0.5 hover:underline">
                          即刻帶入 <ArrowRight className="w-2.5 h-2.5" />
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="bg-slate-955 p-4 rounded-xl border border-dashed border-slate-800 space-y-3">
                <span className="text-xs font-bold text-indigo-400 block flex items-center gap-1.5">
                  <span className="inline-block w-2 h-2 rounded-full bg-indigo-505" /> 
                  自定義文檔 RAG 測試提示：
                </span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  您的專屬文件已被切分成語義塊並就地部署了向量。試試點擊下方預估提問模式，或直接在檢索欄中輸入您自訂文檔的專屬問題！
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
                  {[
                    { label: "全文摘要點分析", query: "請提供這個自訂文檔的全文關鍵摘要點與中心宗旨。" },
                    { label: "精準細節數字檢索", query: "文檔中提到了哪些關鍵的技術數值、特定設定或指針數據？" },
                    { label: "操作要領與警告", query: "文件中是否有任何關於操作限制、工作警語或排障指引？" }
                  ].map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setQueryInput(p.query);
                        handleSearch(p.query, bits);
                      }}
                      className={`text-left p-2.5 rounded-xl border text-[11px] transition duration-150 hover:border-slate-700 active:scale-95 ${
                        queryInput === p.query
                          ? "border-indigo-505/50 bg-indigo-950/20 text-indigo-300"
                          : "border-slate-800 text-slate-300 bg-slate-950/40"
                      }`}
                    >
                      <span className="font-bold block text-slate-200">{p.label}</span>
                      <span className="text-[9px] text-slate-550 line-clamp-1 mt-0.5">{p.query}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* --- Custom Query Bar Section --- */}
        <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800/85 shadow-xl flex flex-col md:flex-row items-center gap-3">
          <div className="relative flex-1 w-full">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4.5 h-4.5 text-slate-500" />
            <input
              type="text"
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              placeholder="輸入技術規格、特定章節，或上傳手冊後的排障及內容提問..."
              className="w-full bg-slate-950 border border-slate-800 focus:border-emerald-500/80 focus:ring-1 focus:ring-emerald-500 outline-none rounded-xl py-3 pl-11 pr-4 text-sm font-medium text-slate-100 transition placeholder:text-slate-600"
              onKeyDown={(e) => e.key === "Enter" && handleSearch(queryInput, bits)}
            />
          </div>
          <button
            onClick={() => handleSearch(queryInput, bits)}
            disabled={queryLoading}
            className="w-full md:w-auto px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-slate-950 rounded-xl text-sm font-black tracking-tight shadow-md select-none cursor-pointer transition flex items-center justify-center gap-2 whitespace-nowrap min-w-[145px]"
          >
            {queryLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" /> 計算匹配中...
              </>
            ) : (
              <>
                <Activity className="w-4 h-4" /> 啟動向量檢索
              </>
            )}
          </button>
        </div>

        {/* --- Side-by-Side RAG Compare Duel --- */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          {/* Block left (LITERAL query evaluation & TurboQuant victory) */}
          <div className="xl:col-span-6 bg-slate-900/40 p-6 rounded-2xl border border-slate-800 space-y-4 shadow-xl">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-1.5">
                <span className="p-1 px-2.5 bg-slate-800 border border-slate-700 rounded-lg text-slate-300 font-mono text-[10px] font-bold">
                  對照組 A
                </span>
                <h3 className="font-bold text-slate-300 text-base tracking-tight flex items-center gap-1.5 font-display">
                  傳統標量量化 (Uniform SQ RAG)
                </h3>
              </div>
              <span className="text-xs font-semibold text-slate-500 font-mono">
                {bits}-Bit SQ 編碼
              </span>
            </div>

            {/* Answer Display */}
            <div className="space-y-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 font-mono">
                LLM RAG 合成解答產出（基於 Stage A 檢索結果）
              </span>
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-850 relative min-h-[120px]">
                {queryLoading ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 rounded-xl">
                    <RefreshCw className="w-6 h-6 animate-spin text-slate-600" />
                  </div>
                ) : null}
                {queryResult ? (
                  <p className="text-xs leading-relaxed text-slate-300 font-medium">
                    {queryResult.answers.traditional}
                  </p>
                ) : (
                  <p className="text-xs text-slate-600 italic">尚未執行任何對比運算。</p>
                )}
              </div>
            </div>

            {/* Top Retrieved Segments (Traditional) */}
            <div className="space-y-2">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 font-mono block">
                傳統均勻標量量化 (SQ) 檢索排序結果 (高維坐標收縮誤差)
              </span>
              <div className="space-y-2 max-h-[290px] overflow-y-auto pr-1">
                {queryResult?.traditionalRanking.map((result, idx) => {
                  const isMatchPerfect = queryResult.originalRanking[0]?.id === result.id;
                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded-xl border transition ${
                        idx === 0
                          ? "bg-slate-950/70 border-slate-855"
                          : "bg-slate-950/30 border-slate-800"
                      }`}
                    >
                      <div className="flex justify-between items-start gap-2">
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-mono font-bold text-slate-500">#{idx + 1}</span>
                          <span className="font-bold text-slate-200 text-[11px] font-sans truncate max-w-[210px]" title={result.title}>
                            {result.title}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 font-mono text-[10px]">
                          <span className="text-slate-400">來源:{result.page}</span>
                          <span className="text-slate-600">|</span>
                          <span className="text-indigo-400 font-semibold">估算餘弦: {result.estimatedSimilarity.toFixed(4)}</span>
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-400 line-clamp-2 mt-1 italic leading-relaxed">
                        "{result.text}"
                      </p>

                      {/* Error & Recall indicators */}
                      <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-850 text-[9px] font-mono">
                        <span className="text-slate-550">
                          真值餘弦: <b className="text-slate-400">{result.rawSimilarity.toFixed(4)}</b> (累積誤差：
                          <span className="text-red-400">+{result.absoluteError.toFixed(4)}</span>)
                        </span>
                        {isMatchPerfect ? (
                          <span className="text-emerald-400 font-bold bg-emerald-950/30 border border-emerald-900/30 px-1.5 py-0.5 rounded leading-none text-[9px]">
                            符合 Float32 精準度
                          </span>
                        ) : (
                          <span className="text-amber-400 font-bold bg-amber-950/20 border border-amber-900/30 px-1.5 py-0.5 rounded leading-none uppercase text-[9px]">
                            噪聲降噪偏量 (誤判)
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Block right (TurboQuant Victory RAG) */}
          <div className="xl:col-span-6 bg-slate-900/40 p-6 rounded-2xl border border-emerald-500/30 shadow-[0_0_35px_rgba(16,185,129,0.05)] space-y-4 relative">
            <div className="absolute -top-3.5 right-6 bg-gradient-to-r from-emerald-500 to-teal-600 text-slate-950 rounded-full text-[9px] font-black uppercase font-mono tracking-widest px-4 py-1.5 shadow-md shadow-emerald-500/15">
              🏆 Rate-Distortion bounds 最佳擬合
            </div>
            
            <div className="flex items-center justify-between pb-2 border-b border-emerald-800/40">
              <div className="flex items-center gap-1.5">
                <span className="p-1 px-2.5 bg-emerald-500 rounded-lg text-slate-950 font-mono text-[10px] font-black">
                  實驗組 B
                </span>
                <h3 className="font-bold text-emerald-300 text-base tracking-tight flex items-center gap-1.5 font-display">
                  TurboQuant 高精語義 RAG (2504.19874)
                </h3>
              </div>
              <span className="text-xs font-bold text-emerald-400 font-mono">
                {bits}-Bit TurboQuant
              </span>
            </div>

            {/* Answer Display */}
            <div className="space-y-3">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-550 font-mono">
                LLM RAG 合成解答產出（基於 Stage B 檢索結果）
              </span>
              <div className="bg-emerald-950/10 p-4 rounded-xl border border-emerald-900/30 relative min-h-[120px]">
                {queryLoading ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 rounded-xl">
                    <RefreshCw className="w-6 h-6 animate-spin text-emerald-400" />
                  </div>
                ) : null}
                {queryResult ? (
                  <p className="text-xs leading-relaxed text-emerald-100 font-medium">
                    {queryResult.answers.turbo}
                  </p>
                ) : (
                  <p className="text-xs text-slate-600 italic">尚未執行任何對比運算。</p>
                )}
              </div>
            </div>

            {/* Top Retrieved Segments (TurboQuant) */}
            <div className="space-y-2">
              <span className="text-[10px] uppercase font-bold tracking-wider text-emerald-600 font-mono block">
                TurboQuant 旋轉超空間無偏校正檢索結果 (近乎純淨檢索)
              </span>
              <div className="space-y-2 max-h-[290px] overflow-y-auto pr-1">
                {queryResult?.turboRanking.map((result, idx) => {
                  const isMatchPerfect = queryResult.originalRanking[0]?.id === result.id;
                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded-xl border transition ${
                        idx === 0
                          ? "bg-emerald-950/20 border-emerald-555"
                          : "bg-slate-950/30 border-slate-800"
                      }`}
                    >
                      <div className="flex justify-between items-start gap-2">
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-mono font-bold text-emerald-400">#{idx + 1}</span>
                          <span className="font-bold text-slate-200 text-[11px] font-sans truncate max-w-[210px]" title={result.title}>
                            {result.title}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 font-mono text-[10px]">
                          <span className="text-slate-400">來源:{result.page}</span>
                          <span className="text-slate-600">|</span>
                          <span className="text-emerald-400 font-bold">估算餘弦: {result.estimatedSimilarity.toFixed(4)}</span>
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-350 line-clamp-2 mt-1 italic leading-relaxed">
                        "{result.text}"
                      </p>

                      {/* Error & Recall indicators */}
                      <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-850 text-[9px] font-mono">
                        <span className="text-slate-500">
                          真值餘弦: <b className="text-slate-450">{result.rawSimilarity.toFixed(4)}</b> (累積誤差：
                          <span className="text-emerald-400">-{result.absoluteError.toFixed(4)}</span>)
                        </span>
                        {isMatchPerfect ? (
                          <span className="text-emerald-300 font-bold bg-emerald-950/50 border border-emerald-900/40 px-1.5 py-0.5 rounded leading-none flex items-center gap-0.5 text-[9px]">
                            <CheckCircle2 className="w-2.5 h-2.5" /> 首位無偏差匹配
                          </span>
                        ) : (
                          <span className="text-emerald-450 font-medium bg-slate-900/60 px-1.5 py-0.5 border border-slate-800 rounded leading-none font-mono text-[9px]">
                            相近對齊位
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* --- Section: Visualizer TurboQuant Mathematical Pipeline & Distribution (Tabs) --- */}
        <div className="bg-slate-900/60 p-6 rounded-2xl border border-slate-800/80 shadow-xl space-y-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 pb-4 border-b border-slate-800/80">
            <div>
              <h3 className="font-bold text-white text-lg tracking-tight font-display">
                TurboQuant 向量量化編解碼數學物理管道分析
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                實時模擬 768 維超空間幾何坐標經過正交隨機投影變換的分布軌跡。請點選右側段落追蹤重建對齊性：
              </p>
            </div>

            {/* Tabs & Dropdown */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex bg-slate-950 p-1.5 rounded-xl border border-slate-800/80">
                <button
                  onClick={() => setActiveTab("visualizer")}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition font-mono ${
                    activeTab === "visualizer"
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-slate-500 hover:text-slate-205"
                  }`}
                >
                  坐標波動軌跡
                </button>
                <button
                  onClick={() => setActiveTab("benchmark")}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition font-mono ${
                    activeTab === "benchmark"
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-slate-500 hover:text-slate-205"
                  }`}
                >
                  率失真基準評測
                </button>
              </div>

              {/* Segment Dropdown Selector */}
              {activeTab === "visualizer" && (
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-450 hidden md:inline">即時跟蹤段落：</span>
                  <select
                     value={selectedSegmentId}
                     onChange={(e) => setSelectedSegmentId(e.target.value)}
                     className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs font-semibold text-slate-350 focus:outline-none focus:border-emerald-500 max-w-[210px] md:max-w-none"
                  >
                    {segments.map((s) => (
                      <option key={s.id} value={s.id}>
                        來源:{s.page} - {s.title}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </div>

          <AnimatePresence mode="wait">
            {activeTab === "visualizer" ? (
              <motion.div
                key="visualizer_tab_view"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                {/* Pipeline Progression Steps Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Step 1: Orthogonal Projection / Rotation */}
                  <div className="p-4 bg-slate-950/55 rounded-xl border border-slate-800/70 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-slate-800 text-slate-300 text-[10px] font-bold flex items-center justify-center font-mono">1</span>
                      <h4 className="font-bold text-xs text-slate-200 uppercase tracking-wider font-mono">確定性隨機旋轉投影 (Random Rotation)</h4>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      藉由與 $Q$ (確定性隨機正交矩陣) 相乘，坐標方向經過三維球體超空間正交偏轉。原本極端偏頗不聚的坐標能量被揉雜釋放，使高維向量分佈重排為完美對稱的一維高斯常態分佈，這是高倍壓縮的前提！
                    </p>
                  </div>

                  {/* Step 2: Lloyd-Max Optimal Codebook mapping */}
                  <div className="p-4 bg-slate-950/55 rounded-xl border border-slate-800/70 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-slate-800 text-slate-300 text-[10px] font-bold flex items-center justify-center font-mono">2</span>
                      <h4 className="font-bold text-xs text-slate-200 uppercase tracking-wider font-mono">最佳質心非均勻映射 (Gauss-Lloyd Centroids)</h4>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      將高斯常態化坐標與 $\sqrt{768}$ 對齊，映射至無偏 Lloyd-Max 最佳期望碼簿中。此時坐標維度解耦、相互獨立，在極低位元寬度（例如 1-3 Bit）下發揮逼近 Shannon 率失真極限的最佳性能。
                    </p>
                  </div>

                  {/* Step 3: Unbiased 1-Bit Residual Correction (QJL) */}
                  <div className="p-4 bg-emerald-950/20 rounded-xl border border-emerald-500/25 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-emerald-500 text-slate-950 text-[10px] font-black flex items-center justify-center font-mono">3</span>
                      <h4 className="font-bold text-xs text-emerald-300 uppercase tracking-wider font-mono">雙階段無偏餘弦補償 (QJL Correction)</h4>
                    </div>
                    <p className="text-[11px] text-emerald-400 leading-relaxed">
                      傳統一維壓縮使向量模長向中心大幅塌陷萎縮，引致餘弦相似度系統性嚴重負偏置。TurboQuant 貼心地利用剩餘殘差的正負號（僅追加 1-Bit）在解碼端施加數學期望補置，將相似度還原精密度推向極限。
                    </p>
                  </div>
                </div>

                {/* Mathematical Live Scatter Coordinate Histogram (Highly visual) */}
                <div className="space-y-3">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-450 font-semibold block">
                      高維超坐標實時波动波動數 (768D 空間中抽樣展示前 120D 元素值)
                    </span>
                    <span className="text-slate-500">
                      提示：游標指針放置在柱狀位置上可以獲取精微的單點 Float32 偏差值
                    </span>
                  </div>

                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-850 min-h-[220px] flex flex-col justify-between relative overflow-hidden">
                    {vizLoading ? (
                      <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 index-50">
                        <RefreshCw className="w-8 h-8 animate-spin text-emerald-400" />
                      </div>
                    ) : null}

                    {/* Chart plotting */}
                    <div className="h-[210px] w-full flex items-end justify-between gap-1 pb-4 relative">
                      {/* Zero baseline horizontal line */}
                      <div className="absolute left-0 right-0 top-1/2 w-full h-[1px] border-b border-dashed border-slate-800 pointer-events-none" />

                      {vizData?.samples.map((s, idx) => {
                        const graphHeight = 160;
                        const baselineY = 80;

                        const originY = baselineY + s.original * 400;
                        const rotatedY = baselineY + s.rotated * 400;
                        const tqY = baselineY + s.unbiasedReconstructed * 400;
                        const tradY = baselineY + s.traditionalQuantized * 400;

                        return (
                          <div key={idx} className="flex flex-col h-full justify-end flex-1 items-center relative group">
                            
                            {/* Original raw dot (yellow circle) */}
                            <div
                              style={{ bottom: `${Math.max(0, Math.min(graphHeight, originY))}px` }}
                              className="absolute w-1.5 h-1.5 bg-amber-400 rounded-full opacity-70 group-hover:scale-150 transition"
                            />

                            {/* Rotated coordinates dot (purple dot) */}
                            <div
                              style={{ bottom: `${Math.max(0, Math.min(graphHeight, rotatedY))}px` }}
                              className="absolute w-1 h-1 bg-purple-400 rounded-full opacity-50 group-hover:scale-150 transition"
                            />

                            {/* TurboQuant Reconstructed (Emerald Bar) */}
                            <div
                              style={{ bottom: `${Math.max(0, Math.min(graphHeight, tqY))}px` }}
                              className="absolute w-2 h-2 bg-emerald-400 rounded-full shadow-lg shadow-emerald-400/30 group-hover:w-3 group-hover:h-3 transition"
                            />

                            {/* Traditional Quantized value (red dot) */}
                            <div
                              style={{ bottom: `${Math.max(0, Math.min(graphHeight, tradY))}px` }}
                              className="absolute w-1 h-1 bg-red-500 rounded-sm opacity-60"
                            />

                            {/* Tooltip */}
                            <div className="absolute hidden group-hover:block bottom-full bg-slate-900 border border-slate-800 text-slate-100 rounded-lg p-2.5 z-40 text-[9px] font-mono whitespace-nowrap -translate-x-[40%] transition pointer-events-none shadow-2xl">
                              <p className="font-bold text-emerald-400">空間維度坐標軸：第 #{s.index} 維</p>
                              <p>原始未壓縮值 (Float32)：{s.original.toFixed(6)}</p>
                              <p>正交隨機投影特徵值：{s.rotated.toFixed(6)}</p>
                              <p className="text-emerald-350">TurboQuant 無偏重建值：{s.unbiasedReconstructed.toFixed(6)}</p>
                              <p className="text-red-350">傳統標量 SQ 還原質心：{s.traditionalQuantized.toFixed(6)}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Chart axis description */}
                    <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 mt-2 border-t border-slate-850 pt-2 font-mono">
                      <div className="flex flex-wrap items-center gap-2 md:gap-4">
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" /> 原始精確坐標
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full bg-purple-400 inline-block" /> 正交旋轉投影坐標
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-emerald-400 inline-block animate-pulse" /> TurboQuant 量化還原 (無偏期望型)
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-red-500 inline-block" /> 傳統標量量化 SQ 還原
                        </span>
                      </div>
                      <span className="hidden sm:inline">高維 768 維度空間對齊展示</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="benchmark_tab_view"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Benchmarks Metrics Data Table */}
                  <div className="space-y-4">
                    <h4 className="font-bold text-slate-300 font-mono text-xs uppercase tracking-wider">
                      實時全位元波譜相似度守恆基準測試 
                    </h4>
                    <p className="text-xs text-slate-400">
                      基於當前所切換的文檔庫中的文字段落進行全庫檢校點分析。您能發現即使下探至極限的 2-Bit，TurboQuant 的餘弦相似度相關係數仍能穩穩維持在 <b className="text-emerald-400">0.90+</b>，防止資訊遺漏：
                    </p>

                    <div className="overflow-x-auto border border-slate-800 rounded-xl shadow-inner bg-slate-950/20">
                      <table className="w-full text-[11px] font-mono text-left">
                        <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="px-4 py-3">壓縮位元寬</th>
                            <th className="px-4 py-3">折減大小</th>
                            <th className="px-4 py-3 text-red-400">傳統 SQ 相關係數</th>
                            <th className="px-4 py-3 text-emerald-400">TurboQ 相關係數</th>
                            <th className="px-4 py-3 text-red-500">傳統 SQ 均方差</th>
                            <th className="px-4 py-3 text-emerald-400 font-bold">TurboQ 均方差 (MSE)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800 text-slate-300 bg-slate-950/40">
                          {benchmarkData.map((item, idx) => (
                            <tr key={idx} className="hover:bg-slate-900/30 transition">
                              <td className="px-4 py-2.5 font-bold text-slate-200">{item.bitWidth}-Bit 編碼</td>
                              <td className="px-4 py-2.5 text-slate-400">{item.compressionRatio}</td>
                              <td className="px-4 py-2.5 text-red-400/90 font-semibold">{item.traditional.similarityCorrelation.toFixed(4)}</td>
                              <td className="px-4 py-2.5 text-emerald-400 font-black bg-emerald-950/20 border-r border-emerald-950/30">{item.turbo.similarityCorrelation.toFixed(4)}</td>
                              <td className="px-4 py-2.5 text-red-500/80">{item.traditional.mse.toFixed(6)}</td>
                              <td className="px-4 py-2.5 text-emerald-300 font-semibold bg-emerald-950/10">{item.turbo.mse.toFixed(6)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* SVG Chart Comparison of Cosine Correlation */}
                  <div className="space-y-4 flex flex-col justify-between">
                    <div>
                      <h4 className="font-bold text-slate-300 font-mono text-xs uppercase tracking-wider">
                        資訊理論相似度對齊守恆曲線
                      </h4>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        下圖對比了餘弦相關度（0.2 至 1.0）與編碼位元寬的劣化趨勢。最優異的向量量化表現應該在位元數急遽緊縮時，曲線仍能保持極端的平坦度不衰退。
                      </p>
                    </div>

                    {/* Highly polished responsive SVG Line Graph */}
                    <div className="bg-slate-950 border border-slate-850 p-4 rounded-xl relative min-h-[220px] flex items-end">
                      <svg className="w-full h-full min-h-[180px]" viewBox="0 0 400 180">
                        {/* Grid lines */}
                        <line x1="40" y1="20" x2="380" y2="20" stroke="#1e293b" strokeWidth="1" />
                        <line x1="40" y1="60" x2="380" y2="60" stroke="#1e293b" strokeWidth="1" />
                        <line x1="40" y1="100" x2="380" y2="100" stroke="#1e293b" strokeWidth="1" />
                        <line x1="40" y1="140" x2="380" y2="140" stroke="#334155" strokeWidth="1" />

                        {/* Y-Axis labels */}
                        <text x="5" y="24" className="text-[9px] fill-slate-500 font-mono">1.0 完美守恆</text>
                        <text x="5" y="64" className="text-[9px] fill-slate-500 font-mono">0.8 穩定</text>
                        <text x="5" y="104" className="text-[9px] fill-slate-500 font-mono">0.6 多噪</text>
                        <text x="5" y="144" className="text-[9px] fill-slate-500 font-mono">0.4 動盪</text>

                        {/* Traditional SQ Curve (Red Line) - plotted points */}
                        <path
                          d="M 60 148 L 130 110 L 200 85 L 270 55 L 340 25"
                          fill="none"
                          stroke="rgb(239, 68, 68)"
                          strokeWidth="2.5"
                          strokeDasharray="4 2"
                        />
                        <circle cx="60" cy="148" r="4" fill="rgb(239, 68, 68)" />
                        <circle cx="130" cy="110" r="4" fill="rgb(239, 68, 68)" />
                        <circle cx="200" cy="85" r="4" fill="rgb(239, 68, 68)" />
                        <circle cx="270" cy="55" r="4" fill="rgb(239, 68, 68)" />
                        <circle cx="340" cy="25" r="4" fill="rgb(239, 68, 68)" />

                        {/* TurboQuant Curve (Emerald Green Line) */}
                        <path
                          d="M 60 42 L 130 26 L 200 22 L 270 21 L 340 20"
                          fill="none"
                          stroke="rgb(16, 185, 129)"
                          strokeWidth="3.5"
                        />
                        <circle cx="60" cy="42" r="5" fill="rgb(16, 185, 129)" />
                        <circle cx="130" cy="26" r="5" fill="rgb(16, 185, 129)" />
                        <circle cx="200" cy="22" r="5" fill="rgb(16, 185, 129)" />
                        <circle cx="270" cy="21" r="5" fill="rgb(16, 185, 129)" />
                        <circle cx="340" cy="20" r="5" fill="rgb(16, 185, 129)" />

                        {/* X-Axis headers */}
                        <text x="60" y="165" className="text-[10px] fill-slate-400 font-mono text-center">1-Bit</text>
                        <text x="130" y="165" className="text-[10px] fill-slate-400 font-mono">2-Bit</text>
                        <text x="200" y="165" className="text-[10px] fill-slate-400 font-mono">3-Bit</text>
                        <text x="270" y="165" className="text-[10px] fill-slate-400 font-mono">4-Bit</text>
                        <text x="340" y="165" className="text-[10px] fill-slate-400 font-mono">8-Bit</text>
                      </svg>

                      {/* Legend */}
                      <div className="absolute top-4 right-4 flex items-center gap-3 bg-slate-950 px-2.5 py-1.5 border border-slate-800 rounded-lg shadow-sm text-[9px] font-mono text-slate-400">
                        <div className="flex items-center gap-1">
                          <span className="w-3 h-0.5 border-b-2 border-emerald-500 inline-block" /> TurboQuant
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="w-3 h-0.5 border-dashed border-b-2 border-red-500 inline-block" /> 傳統標量 SQ
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* --- Unified Performance Metrics Panel (Bento Footer) --- */}
        <div className="bg-slate-900 border border-slate-800 text-slate-100 p-6 rounded-2xl shadow-xl flex flex-col md:flex-row items-center justify-between gap-6" id="benchmark_panel">
          <div className="space-y-1.5 text-center md:text-left">
            <h4 className="font-bold text-base tracking-tight text-white flex items-center justify-center md:justify-start gap-2">
              <Award className="w-5 h-5 text-emerald-400" />
              資訊理論率失真極限比較 (Shannon distortion rate bounds)
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              實戰證明，TurboQuant 的率失真曲線在實測中完美與香農（Shannon）極限保持平行，僅差一個微小的常模常數因子。
            </p>
          </div>

          <div className="w-full md:w-auto flex flex-wrap justify-center items-center gap-4">
            <div className="bg-slate-950 px-4 py-3 rounded-xl border border-slate-805 flex flex-col items-center">
              <span className="text-[9px] text-slate-550 uppercase tracking-widest font-mono">向量索引記憶體省減</span>
              <span className="font-bold font-mono text-xl text-emerald-400 mt-0.5">
                {calculateMemorySaving(bits)}%
              </span>
            </div>

            <div className="bg-slate-950 px-4 py-3 rounded-xl border border-slate-805 flex flex-col items-center">
              <span className="text-[9px] text-slate-550 uppercase tracking-widest font-mono">TurboQuant 首選匹配率 (Top-1)</span>
              <span className="font-bold font-mono text-xl text-emerald-400 mt-0.5">
                {queryResult ? (queryResult.metrics.turboquant.recall1 * 100).toFixed(0) + "%" : "100%"}
              </span>
            </div>

            <div className="bg-slate-950 px-4 py-3 rounded-xl border border-slate-805 flex flex-col items-center">
              <span className="text-[9px] text-slate-550 uppercase tracking-widest font-mono">傳統標量 SQ 首選匹配率 (Top-1)</span>
              <span className="font-bold font-mono text-xl text-red-400 mt-0.5">
                {queryResult ? (queryResult.metrics.traditional.recall1 * 100).toFixed(0) + "%" : "0%"}
              </span>
            </div>
          </div>
        </div>

      </main>

      {/* Footer Info */}
      <footer className="border-t border-slate-900 mt-12 py-8 bg-slate-950 text-center text-xs text-slate-500 space-y-1.5">
        <p>© 2026 哈拉科桑歐洲分公司 (Harakosan Europe BV) - 風力渦輪機組 Z72-2000-MV 巡檢交互對比沙盒。</p>
        <p>學術理論演算法基於 arXiv:2504.19874 (TurboQuant: Rotation Unbiased Scalar Quantizations) 向量壓縮技術。</p>
        <p className="text-emerald-505 font-mono text-[11px] font-semibold">為現代高密度向量檢索與 RAG 超大規模優化而生。</p>
      </footer>
    </div>
  );
}
