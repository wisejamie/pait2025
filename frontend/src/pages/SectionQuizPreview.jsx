import React, { useEffect, useState, useRef } from "react";
import ReactDOM from "react-dom/client";
import { useParams, useNavigate, Link } from "react-router-dom";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function SectionQuizPreview() {
  const { docId, sectionId } = useParams();
  const navigate = useNavigate();

  // Section & TOC state
  const [section, setSection] = useState(null);
  const [allSections, setAllSections] = useState([]);
  const [loading, setLoading] = useState(true);

  // Quiz generation state
  const [generating, setGenerating] = useState(false);

  // Summarization state
  const [summaryLevel, setSummaryLevel] = useState("tldr");
  const [summary, setSummary] = useState("");
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [summaryError, setSummaryError] = useState("");

  // Transformation state (view modes)
  const [viewMode, setViewMode] = useState("original");
  const [transformedTexts, setTransformedTexts] = useState({
    simplify: "",
    elaborate: "",
    distill: "",
  });
  const [loadingTransform, setLoadingTransform] = useState(false);
  const [transformError, setTransformError] = useState("");

  // Highlight & explain state
  const containerRef = useRef(null);
  const toolbarRef = useRef(null);
  const [selText, setSelText] = useState("");
  const [toolbarPos, setToolbarPos] = useState(null);
  const [loadingExplain, setLoadingExplain] = useState(false);

  // Mode descriptions
  const modeDescriptions = {
    original:
      "Original: full scholarly prose, contains complete details and academic language.",
    simplify:
      "Simplified: easier-to-read language while preserving meaning and structure.",
    elaborate:
      "Elaborated: adds brief explanations or examples for technical terms and complex ideas.",
    distill:
      "Distilled: only the core ideas and arguments, with supporting details removed for conciseness.",
  };

  // Load section & TOC
  useEffect(() => {
    async function loadSection() {
      try {
        const res = await axios.get(
          `${API_BASE}/documents/${docId}/sections/${sectionId}`
        );
        setSection(res.data);

        const treeRes = await axios.get(
          `${API_BASE}/documents/${docId}/sections`
        );
        const treeData = Array.isArray(treeRes.data)
          ? treeRes.data
          : [treeRes.data];

        const ordered = [];
        const walk = (secs) => {
          secs.forEach((s) => {
            ordered.push({ id: s.id, title: s.title });
            if (s.sub_sections?.length) walk(s.sub_sections);
          });
        };
        walk(treeData[0].sections || treeData);
        setAllSections(ordered);
      } catch (err) {
        console.error("Failed to load section:", err);
      } finally {
        setLoading(false);
      }
    }
    loadSection();
  }, [docId, sectionId]);

  // Show “Explain” button on text selection
  useEffect(() => {
    const onSelection = () => {
      const sel = window.getSelection();
      const txt = sel?.toString().trim() || "";
      if (
        txt &&
        containerRef.current.contains(sel.anchorNode) &&
        sel.rangeCount
      ) {
        const range = sel.getRangeAt(0);
        // getClientRects gives you one rect per line;
        // the last rect is the bottom line of your highlight
        const rects = Array.from(range.getClientRects());
        const lastRect = rects[rects.length - 1];
        const containerRect = containerRef.current.getBoundingClientRect();

        // compute relative offsets:
        // desired offsets
        const topRaw = lastRect.bottom - containerRect.top;
        let rightRaw = containerRect.right - lastRect.right;

        // clamp so the button never goes off left or right
        const btnWidth = toolbarRef.current?.offsetWidth || 0;
        // max distance from container's right edge
        const maxRight = Math.max(0, containerRect.width - btnWidth);
        let rightClamped = Math.min(rightRaw, maxRight);
        rightClamped = Math.max(0, rightClamped);
        setSelText(txt);
        setToolbarPos({ top: topRaw, right: rightClamped });
      } else {
        setSelText("");
        setToolbarPos(null);
      }
    };
    document.addEventListener("mouseup", onSelection);
    document.addEventListener("keyup", onSelection);
    return () => {
      document.removeEventListener("mouseup", onSelection);
      document.removeEventListener("keyup", onSelection);
    };
  }, []);

  // Close tooltips & button on Esc or outside click
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === "Escape") {
        document.activeElement?.blur();
        setSelText("");
        setToolbarPos(null);
      }
    };
    const onClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setSelText("");
        setToolbarPos(null);
        document.activeElement?.blur();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("mousedown", onClick);
    };
  }, []);

  // Early returns
  if (loading) return <div className="p-6">Loading section...</div>;
  if (!section) return <div className="p-6">Section not found.</div>;

  // API actions
  const generateQuestions = async () => {
    setGenerating(true);
    try {
      await axios.post(`${API_BASE}/sections/${sectionId}/questions/generate`);
      navigate(`/quiz/session-mock-${sectionId}`);
    } catch (err) {
      console.error("Failed to generate questions:", err);
      alert("Failed to generate questions. Try again.");
    } finally {
      setGenerating(false);
    }
  };

  const fetchSummary = async () => {
    setLoadingSummary(true);
    setSummaryError("");
    try {
      const res = await axios.post(
        `${API_BASE}/documents/${docId}/sections/${sectionId}/summarize`,
        { level: summaryLevel }
      );
      setSummary(res.data.summary);
    } catch (err) {
      console.error("Error fetching summary:", err);
      setSummaryError("Failed to fetch summary. Please try again.");
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleTransform = async (mode) => {
    if (!transformedTexts[mode]) {
      setLoadingTransform(true);
      setTransformError("");
      try {
        const res = await axios.post(
          `${API_BASE}/documents/${docId}/sections/${sectionId}/transform`,
          { mode }
        );
        setTransformedTexts((prev) => ({
          ...prev,
          [mode]: res.data.transformedText,
        }));
      } catch (err) {
        console.error(`Error transforming text (${mode}):`, err);
        setTransformError("Failed to transform text. Please try again.");
      } finally {
        setLoadingTransform(false);
      }
    }
    setViewMode(mode);
  };

  // Handle Explain: call API then wrap selection
  const handleExplain = async () => {
    if (!selText) return;
    setLoadingExplain(true);
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) {
      setLoadingExplain(false);
      return;
    }
    const range = sel.getRangeAt(0).cloneRange();
    try {
      const res = await axios.post(
        `${API_BASE}/documents/${docId}/sections/${sectionId}/explain`,
        { snippet: selText, context: section.text }
      );
      const exp = res.data.explanation;

      // create highlight span
      const span = document.createElement("span");
      span.className =
        "relative bg-yellow-200 outline-dotted outline-yellow-400 group inline";
      span.setAttribute("tabIndex", "0");

      // extract selected content
      const content = range.extractContents();
      span.appendChild(content);

      // create tooltip container
      const tip = document.createElement("div");
      tip.className =
        "absolute z-10 -top-8 transform mb-1 p-4 bg-gray-800 text-white text-sm rounded shadow-lg hidden group-hover:block group-focus:block overflow-auto";
      span.appendChild(tip);

      // render markdown inside tooltip
      const root = ReactDOM.createRoot(tip);
      root.render(
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{exp}</ReactMarkdown>
      );

      // insert
      range.insertNode(span);
    } catch (err) {
      console.error("Explain failed:", err);
      alert("Failed to fetch explanation. Please try again.");
    } finally {
      setLoadingExplain(false);
      window.getSelection()?.removeAllRanges();
      setSelText("");
      setToolbarPos(null);
    }
  };

  // Rendered text selection
  const displayText =
    viewMode === "original" ? section.text : transformedTexts[viewMode];

  // Navigation
  const idx = allSections.findIndex((s) => s.id === sectionId);
  const prev = idx > 0 ? allSections[idx - 1] : null;
  const next = idx < allSections.length - 1 ? allSections[idx + 1] : null;

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">{section.title}</h2>

      {/* View As */}
      <div className="mb-4">
        <label htmlFor="viewMode" className="font-medium mr-2">
          View As:
        </label>
        <select
          id="viewMode"
          value={viewMode}
          onChange={async (e) => {
            const mode = e.target.value;
            if (mode === "original") setViewMode(mode);
            else await handleTransform(mode);
          }}
          className="px-2 py-1 border rounded"
        >
          <option value="original">Original</option>
          <option value="simplify">Simplified</option>
          <option value="elaborate">Elaborated</option>
          <option value="distill">Distilled</option>
        </select>
        <div className="mt-2 text-gray-600 text-sm">
          {modeDescriptions[viewMode]}
        </div>
        {transformError && (
          <div className="text-red-600 mt-1">{transformError}</div>
        )}
      </div>

      {/* Summarization */}
      <div className="mb-4 flex items-center space-x-2">
        <select
          value={summaryLevel}
          onChange={(e) => setSummaryLevel(e.target.value)}
          className="p-1 border rounded"
        >
          <option value="tldr">TL;DR</option>
          <option value="short">Short</option>
          <option value="bullets">Bullets</option>
          <option value="simple">Simple</option>
        </select>
        <button
          onClick={fetchSummary}
          disabled={loadingSummary}
          className="bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700 disabled:opacity-50"
        >
          {loadingSummary ? "Loading..." : "Get Summary"}
        </button>
      </div>
      {summary && (
        <div className="bg-green-50 border-l-4 border-green-400 p-4 mb-4 whitespace-pre-wrap">
          {summary}
        </div>
      )}
      {summaryError && <div className="text-red-600 mb-4">{summaryError}</div>}

      {/* Content + Highlight container */}
      <div className="prose max-w-none mb-6 relative" ref={containerRef}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayText}</ReactMarkdown>
        {selText && toolbarPos && (
          <div
            ref={toolbarRef}
            className="absolute z-50 bg-white bg-opacity-90 rounded-full shadow-lg px-3 py-1 flex items-center transition-opacity duration-200"
            style={{
              top: toolbarPos.top,
              right: toolbarPos.right,
            }}
          >
            <button
              onClick={handleExplain}
              disabled={loadingExplain}
              className="flex items-center text-gray-700 hover:text-gray-900 focus:outline-none"
              aria-label="Explain selection"
            >
              {loadingExplain ? (
                <span className="animate-pulse">⏳</span>
              ) : (
                <span className="flex items-center space-x-1 text-sm">
                  <span>📝</span>
                  <span>Explain</span>
                </span>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Quiz button */}
      <button
        onClick={generateQuestions}
        disabled={generating}
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {generating ? "Generating..." : "Start Quiz"}
      </button>

      {/* Nav links */}
      <div className="mt-8 flex justify-between">
        {prev ? (
          <Link
            to={`/documents/${docId}/sections/${prev.id}`}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded"
          >
            ← {prev.title}
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link
            to={`/documents/${docId}/sections/${next.id}`}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded"
          >
            {next.title} →
          </Link>
        ) : (
          <span />
        )}
      </div>
    </div>
  );
}
