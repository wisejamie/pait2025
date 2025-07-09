import React, { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function SectionQuizPreview() {
  const { docId, sectionId } = useParams();
  const navigate = useNavigate();

  const [section, setSection] = useState(null);
  const [allSections, setAllSections] = useState([]);
  const [loading, setLoading] = useState(true);
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

  // Descriptions for each mode
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

  const generateQuestions = async () => {
    try {
      setGenerating(true);
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
    setTransformError("");
    if (!transformedTexts[mode]) {
      setLoadingTransform(true);
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
        return;
      } finally {
        setLoadingTransform(false);
      }
    }
    setViewMode(mode);
  };

  if (loading) return <div className="p-6">Loading section...</div>;
  if (!section) return <div className="p-6">Section not found.</div>;

  const idx = allSections.findIndex((s) => s.id === sectionId);
  const prev = idx > 0 ? allSections[idx - 1] : null;
  const next = idx < allSections.length - 1 ? allSections[idx + 1] : null;

  // Determine display text
  const displayText =
    viewMode === "original" ? section.text : transformedTexts[viewMode];

  // Normalize paragraph breaks for markdown rendering
  // Only normalize for original mode to preserve tables in transformed modes
  const markdownText =
    viewMode === "original"
      ? displayText
          .split(/\n\s*\n+/)
          .filter((para) => para.trim() !== "")
          .join("\n\n")
      : displayText;

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">{section.title}</h2>

      {/* View As control */}
      <div className="mb-4">
        <label htmlFor="viewMode" className="font-medium mr-2">
          View As:
        </label>
        <select
          id="viewMode"
          value={viewMode}
          onChange={async (e) => {
            const mode = e.target.value;
            if (mode === "original") {
              setViewMode("original");
            } else {
              await handleTransform(mode);
            }
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

      {/* Summarization controls */}
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

      {/* Display summary if available */}
      {summary && (
        <div className="bg-green-50 border-l-4 border-green-400 p-4 mb-4 whitespace-pre-wrap">
          {summary}
        </div>
      )}
      {summaryError && <div className="text-red-600 mb-4">{summaryError}</div>}

      {/* Section text rendered as markdown */}
      <div className="prose max-w-none mb-6">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {markdownText}
        </ReactMarkdown>
      </div>

      {/* Quiz button */}
      <button
        onClick={generateQuestions}
        disabled={generating}
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {generating ? "Generating..." : "Start Quiz"}
      </button>

      {/* Navigation links */}
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
