import { useParams, useNavigate, Link } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function SectionQuizPreview() {
  const { docId, sectionId } = useParams();
  const navigate = useNavigate();

  const [section, setSection] = useState(null);
  const [allSections, setAllSections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    async function loadSection() {
      try {
        const res = await axios.get(
          `${API_BASE}/documents/${docId}/sections/${sectionId}`
        );
        setSection(res.data);

        const tree = await axios
          .get(`${API_BASE}/documents/${docId}/sections`)
          .then((r) => (Array.isArray(r.data) ? r.data : [r.data]));

        // flatten in-order: parent then its subsections
        const ordered = [];
        const walk = (secs) => {
          secs.forEach((s) => {
            ordered.push({ id: s.id, title: s.title });
            if (s.sub_sections?.length) walk(s.sub_sections);
          });
        };
        walk(tree[0].sections || tree);
        setAllSections(ordered);
      } catch (err) {
        console.error("Failed to load section:", err);
      } finally {
        setLoading(false);
      }
    }

    loadSection();
  }, [sectionId]);

  const generateQuestions = async () => {
    try {
      setGenerating(true);
      await axios.post(`${API_BASE}/sections/${sectionId}/questions/generate`);
      navigate(`/quiz/session-mock-${sectionId}`); // Replace with real session route when ready
    } catch (err) {
      console.error("Failed to generate questions:", err);
      alert("Failed to generate questions. Try again.");
    } finally {
      setGenerating(false);
    }
  };

  if (loading) return <div className="p-6">Loading section...</div>;
  if (!section) return <div className="p-6">Section not found.</div>;

  const idx = allSections.findIndex((s) => s.id === sectionId);
  const prev = idx > 0 ? allSections[idx - 1] : null;
  const next = idx < allSections.length - 1 ? allSections[idx + 1] : null;

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">{section.title}</h2>
      <div className="space-y-4">
        {section.text &&
          section.text
            .split("\n")
            .map((para, index) => <p key={index}>{para}</p>)}
      </div>

      <button
        onClick={generateQuestions}
        disabled={generating}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        {generating ? "Generating..." : "Start Quiz"}
      </button>
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

        {next && (
          <Link
            to={`/documents/${docId}/sections/${next.id}`}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded"
          >
            {next.title} →
          </Link>
        )}
      </div>
    </div>
  );
}
