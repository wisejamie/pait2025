import React, { useEffect, useState, useRef } from "react";
import ReactDOM from "react-dom/client";
import { useParams, useNavigate, Link } from "react-router-dom";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function SectionDetail() {
  const { docId, sectionId } = useParams();
  const navigate = useNavigate();

  // Section & TOC state
  const [section, setSection] = useState(null);
  const [allSections, setAllSections] = useState([]);
  const [loading, setLoading] = useState(true);

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

  const [askOpen, setAskOpen] = useState(false);
  const [askLoading, setAskLoading] = useState(false);
  const [askQ, setAskQ] = useState("");
  const [askThread, setAskThread] = useState([]);

  async function askTutorInSection() {
    if (!askQ.trim()) return;
    setAskLoading(true);
    try {
      setAskThread((t) => [...t, { role: "user", text: askQ }]);
      setAskQ("");
      const { data } = await axios.post(`${API_BASE}/documents/${docId}/ask`, {
        question: askQ,
        context: "section",
        section_id: sectionId,
      });
      setAskThread((t) => [
        ...t,
        { role: "assistant", text: data?.answer || "No answer returned." },
      ]);
    } catch (e) {
      console.error(e);
      setAskThread((t) => [
        ...t,
        { role: "assistant", text: "_Sorry—failed to fetch an answer._" },
      ]);
    } finally {
      setAskLoading(false);
    }
  }

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

      // 1. Create highlight span
      const span = document.createElement("span");
      span.className = [
        "relative",
        "inline-block",
        "bg-yellow-200",
        "outline-dotted outline-yellow-400",
        "group",
      ].join(" ");
      span.setAttribute("tabIndex", "0");

      // 2. Move selected content inside the span
      const content = range.extractContents();
      span.appendChild(content);

      // 3. Create popover container
      const tip = document.createElement("div");
      tip.className = [
        "absolute",
        "mb-2",
        "w-max",
        "max-w-[90vw]",
        "max-h-96",
        "overflow-auto",
        "p-3",
        "bg-white",
        "text-gray-800",
        "border",
        "border-gray-200",
        "rounded-lg",
        "shadow-lg shadow-gray-400/50 backdrop-blur-sm",
        "ring-2 ring-indigo-400",
        "z-50",
        "hidden", // default hidden
      ].join(" ");
      tip.style.top = "auto";
      tip.style.bottom = "100%";
      tip.style.left = "0"; // initial, will adjust with JS
      tip.style.right = "auto";

      // 4. Mount explanation content using React
      const root = ReactDOM.createRoot(tip);
      root.render(
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{exp}</ReactMarkdown>
      );

      // 5. Append popover to span
      span.appendChild(tip);

      // 6. Show + position popover on hover/focus
      span.addEventListener("mouseenter", () => {
        tip.classList.remove("hidden");
        requestAnimationFrame(() => {
          const tipRect = tip.getBoundingClientRect();
          const spanRect = span.getBoundingClientRect();
          const overflowRight = tipRect.right - window.innerWidth;
          const overflowLeft = tipRect.left;

          if (overflowRight > 0) {
            tip.style.left = "auto";
            tip.style.right = "0";
          } else if (overflowLeft < 0) {
            tip.style.left = "0";
            tip.style.right = "auto";
          } else {
            tip.style.left = "0";
            tip.style.right = "auto";
          }
        });
      });
      span.addEventListener("mouseleave", () => tip.classList.add("hidden"));
      span.addEventListener("focus", () => tip.classList.remove("hidden"));
      span.addEventListener("blur", () => tip.classList.add("hidden"));

      // 7. Insert span into the document
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
          <option value="tldr">1-Sentence</option>
          <option value="short">Short</option>
          <option value="bullets">Bullets</option>
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

      <button
        onClick={() => setAskOpen(true)}
        title="Ask the Tutor"
        className="fixed bottom-5 right-5 z-50 rounded-full shadow-lg px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white"
      >
        💬 Ask
      </button>

      {/* Ask modal */}
      {askOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-end sm:items-center justify-center z-50">
          <div className="bg-white w-full sm:max-w-2xl sm:rounded-lg sm:shadow-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold">
                Ask the Tutor — {section.title}
              </h3>
              <button
                onClick={() => setAskOpen(false)}
                className="text-gray-600 hover:text-gray-800"
              >
                ✕
              </button>
            </div>

            <div className="mb-3">
              <textarea
                value={askQ}
                onChange={(e) => setAskQ(e.target.value)}
                placeholder="Ask a question about this section…"
                rows={3}
                className="w-full border rounded p-3"
              />
            </div>
            <div className="flex justify-end">
              <button
                onClick={askTutorInSection}
                disabled={askLoading}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded disabled:opacity-50"
              >
                {askLoading ? "Thinking…" : "Ask"}
              </button>
            </div>

            {/* Thread */}
            {askThread.length > 0 && (
              <div className="mt-4 max-h-80 overflow-auto space-y-3">
                {askThread.map((msg, i) => (
                  <div
                    key={i}
                    className={
                      msg.role === "user"
                        ? "bg-blue-50 border border-blue-200 p-3 rounded"
                        : "bg-gray-50 border border-gray-200 p-3 rounded"
                    }
                  >
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">
                      {msg.role === "user" ? "You" : "Tutor"}
                    </div>
                    <div className="prose max-w-none text-gray-900">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.text}
                      </ReactMarkdown>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

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
