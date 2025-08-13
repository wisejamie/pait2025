// src/pages/AskTutor.jsx
import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function AskTutor() {
  const { id } = useParams();
  const [title, setTitle] = useState("");
  const [sections, setSections] = useState([]);
  const [contextType, setContextType] = useState("document"); // "document" | "section"
  const [sectionId, setSectionId] = useState("");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [thread, setThread] = useState([]); // [{role:'user'|'assistant', text: string}, ...]

  useEffect(() => {
    (async () => {
      try {
        const docRes = await axios.get(`${API_BASE}/documents/${id}`);
        setTitle(docRes.data.title || "Untitled");

        const secRes = await axios.get(`${API_BASE}/documents/${id}/sections`);
        const data = Array.isArray(secRes.data) ? secRes.data[0] : secRes.data;
        setSections(data?.sections || []);
      } catch (e) {
        console.error("Failed to load doc/sections", e);
      }
    })();
  }, [id]);

  const flatSections = [];
  const flatten = (arr) =>
    arr?.forEach((s) => {
      flatSections.push({ id: s.id, title: s.title });
      if (s.sub_sections?.length) flatten(s.sub_sections);
    });
  flatten(sections);

  async function ask() {
    if (!question.trim()) return;
    setLoading(true);

    const payload = {
      question,
      context: contextType, // "document" | "section"
      section_id: contextType === "section" ? sectionId || null : null,
    };

    try {
      setThread((t) => [...t, { role: "user", text: question }]);
      setQuestion("");
      const { data } = await axios.post(
        `${API_BASE}/documents/${id}/ask`,
        payload
      );
      const answer = data?.answer || "No answer returned.";
      setThread((t) => [...t, { role: "assistant", text: answer }]);
    } catch (e) {
      console.error(e);
      setThread((t) => [
        ...t,
        { role: "assistant", text: "_Sorry—failed to fetch an answer._" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">💬 Ask the Tutor</h1>
        <Link to={`/documents/${id}`} className="text-blue-600 hover:underline">
          ← Back to Document
        </Link>
      </div>
      <p className="text-gray-600 mb-4">
        Document: <span className="font-medium">{title}</span>
      </p>

      {/* Context Picker */}
      <div className="mb-4 flex flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <input
            id="ctx-doc"
            type="radio"
            name="context"
            value="document"
            checked={contextType === "document"}
            onChange={() => setContextType("document")}
          />
          <label htmlFor="ctx-doc">Whole document</label>
        </div>
        <div className="flex items-center gap-2">
          <input
            id="ctx-sec"
            type="radio"
            name="context"
            value="section"
            checked={contextType === "section"}
            onChange={() => setContextType("section")}
          />
          <label htmlFor="ctx-sec">Specific section</label>
        </div>

        {contextType === "section" && (
          <select
            value={sectionId}
            onChange={(e) => setSectionId(e.target.value)}
            className="border rounded px-2 py-1"
          >
            <option value="">Select a section…</option>
            {flatSections.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Ask box */}
      <div className="mb-3">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about this document…"
          rows={3}
          className="w-full border rounded p-3"
        />
      </div>
      <button
        onClick={ask}
        disabled={loading || (contextType === "section" && !sectionId)}
        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded disabled:opacity-50"
      >
        {loading ? "Thinking…" : "Ask"}
      </button>

      {/* Thread */}
      <div className="mt-6 space-y-4">
        {thread.map((msg, i) => (
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
    </div>
  );
}
