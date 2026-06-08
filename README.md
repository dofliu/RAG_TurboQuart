# Harakosan Z72 Turbine RAG Evaluator (極限向量量化與 RAG 語義檢索評測儀)

這是一個基於先進**向量量化壓縮技術**的實時 RAG (檢索增強生成，Retrieval-Augmented Generation) 語義匹配與診斷沙盒。本專案旨在評估、對比並可視化傳統標量量化 (Scalar Quantization, SQ) 與近乎最優的 **TurboQuant (arXiv:2504.19874)** 技術在極限低位元 (1-4 Bits) 壓縮下的向量檢索精準度、誤差分佈、率失真 (Rate-Distortion) 曲線及最終的大型語言模型 (LLM) 問答答覆品質。

---

## 🌟 核心特色功能

1. **雙重文件庫源支援 (Dual-Source Knowledge Library)**
   - **預設風力發電機手冊**：內置 Harakosan Z72-2000-MV 2MW 變速直驅式風機手冊與常見的現場運轉、排障、安全性動作 code 等技術細節段落。
   - **自訂文件上傳與分析 (.txt, .md)**：**新增！** 支援拖曳或上載自備的任何純文字繁體中文或英文說明書。系統將自動進行重疊語義塊切分 (Chunking)，並實時建構 TurboQuant 旋轉無偏量化向量索引。

2. **精準高維向量量化編解碼模擬 (Vector Quantization Simulation)**
   - 支援將原始高維嵌入向量 (768D) 壓縮至 **1-Bit (極限二值化)**、**2-Bit (4級階層)**、**3-Bit (8級階層)** 及 **4-Bit (16級階層)**。
   - **傳統 SQ (標量量化)**：使用均勻的標量量化編碼，展示高維坐標在量化時發生的嚴重的語義扭曲、收縮與噪聲偏量。
   - **實驗組 TurboQuant**：應用確定性隨機正交旋轉投影，消除量化的系統誤差，即使在 1~3 位元的極限壓縮下，仍能取得無偏的相似度估計以及極高的檢索召回率 (Recall)。

3. **雙引擎 RAG 答覆合成對比 (RAG Dual-Engine Duel)**
   - **在線實時 API**：當配置 `GEMINI_API_KEY` 時，使用 **Google Gemini SDK (`@google/genai`)** 中的 `text-embedding-004` 提取高保真嵌入語義，並調用 `gemini-3.5-flash` 進行實時 RAG 答覆合成。
   - **離線高保真回退**：若無 API 金鑰，系統將自動回退至局部 768 維幾何超空間投影（用於相似度精準模擬）和封裝的雙引擎離線本地手冊解算器，保證 100% 的離線可用性與演示安全。

4. **度量指標與率失真評測 (Analytics & Metrics)**
   - **檢索排名與偏差跟蹤**：對比 traditional RAG 與 TurboQuant RAG 的 Top-3 召回率、均方誤差 (MSE) 以及平均絕對誤差 (MAE)。
   - **幾何坐標波動可視化**：可視化 768 維幾何超空間分佈在經過旋轉、量化解壓後的微觀軌跡波動。
   - **率失真基準測試 (Rate-Distortion Benchmark)**：展示 1~8 位元壓縮下，兩種量化方案在語義相似度相關性、MSE 上的整體退化衰減，從數學上證明 TurboQuant 收斂與衰減的緩慢優勢。

---

## 🛠️ 技術架構與模組設計

本專案採用**全端 (Full-Stack) 架構**：

* **前端 (Client-Side)**：
  - **React 18** & **TypeScript** & **Vite** 構建高性能單頁應用。
  - **Tailwind CSS** 提供洗練深邃的 slate 暗色系界面。
  - **motion/react** (Framer Motion) 驅動流程卡片的過渡與無縫動畫。
  - **Lucide React** 提供具視覺指針意義的專業工程圖標。

* **後端 (Server-Side)**：
  - 加密安全的 **Node.js Express** 伺服器作為代理守護。
  - 封裝高錐體數學量化演算法、Lloyd-Max 最優質心生成及 Householder 反射正交矩陣計算。
  - 優雅安全的 **Gemini API** Server-Side 整合，API key 僅存於伺服器端，不暴露給瀏覽器，防止安全漏洞。

---

## 📦 安裝與本地開發步驟

請確保您本地已安裝 [Node.js (版本 18 或以上)](https://nodejs.org/)。

### 1. 複製本專案倉庫
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. 安裝所有依賴
```bash
npm run install-deps
# 或者直接使用標準的 npm 安裝
npm install
```

### 3. 配置環境變量
請於根目錄創建 `.env` 文件。您可以參考 `.env.example`。

```env
# .env 內容
GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
```
*(如果不安置金鑰，系統將運行於全功能高仿真「離線/本地回退」模式下，供您演示)*

### 4. 啟動開發伺服器
```bash
npm run dev
```
此命令會使 Vite 在開發模式下以 Express 中間件運行在 `http://localhost:3000`。您可享受極佳的全端開發體驗。

### 5. 建構與打包部署
```bash
npm run build
```
後端 TypeScript 伺服器會通過 **esbuild** 自動 bundle 並編譯為一個單一、自包含的 CommonJS 格式文件：`dist/server.cjs`（完全消除 ESM 相對路徑執行時的錯誤，適合極速部署），前端資源則輸出至隨附的 `dist/` 目錄。

### 6. 啟動生產環境服務
```bash
npm run start
```
運行在生產環境的高性能伺服器上。

---

## 📐 核心算法原理

* **傳統標量量化 (Standard SQ)**:
  $$\tilde{x}_i = Q(x_i)$$
  在低位元下，大量的坐標值被粗暴地歸併至極少數的分段中心。在大維度 $D=768$ 下，量化誤差對餘弦相似度產生有偏破壞：
  $$E[\cos(e_1, e_2)] \neq \cos(\tilde{e}_1, \tilde{e}_2)$$

* **TurboQuant 正交投影無偏校正 (arXiv:2504.19874)**:
  TurboQuant 利用由確定性隨機數種子生成的 Householder 正交旋轉矩陣 $Q$，對全體點向量實施等距高維旋轉變換：
  $$y = Q \cdot x \quad (\text{其中 } Q^T Q = I)$$
  旋轉後的坐標 $y_i$ 服從近乎完美的均勻高斯分佈。在此空間中再進行量化偏置修正（Lloyd-Max 近似），能夠保持內積與餘弦相似度的數學無偏性。最終在解碼階段將矩陣反向旋轉恢復：
  $$\hat{x} = Q^T \cdot Q_{Lloyd}(Q \cdot x)$$
  這極大地保留了低位元存儲體積，同時保證了 RAG 相似度召回的精準對齊。

---

## 📜 許可證說明
本專案依據 MIT 協議授權開源。
如有相關學術引用，請參考 arXiv:2504.19874 論文。
