import { useParams, Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function DocumentDetail() {
  const { id } = useParams();
  const [sections, setSections] = useState([]);
  const [title, setTitle] = useState("");
  const [objectives, setObjectives] = useState({});
  const [questions, setQuestions] = useState(null);
  const [building, setBuilding] = useState(false);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    async function load() {
      try {
        const docRes = await axios.get(`${API_BASE}/documents/${id}`);
        setTitle(docRes.data.title || "Untitled");

        const sectionRes = await axios.get(
          `${API_BASE}/documents/${id}/sections`
        );
        const response = Array.isArray(sectionRes.data)
          ? sectionRes.data[0]
          : sectionRes.data;
        setSections(response.sections || []);
        setObjectives(response.learning_objectives || {});

        console.log("Fetched sections response:", sectionRes.data);
        // ➤ fetch existing questions for this document
        const qRes = await axios.get(`${API_BASE}/documents/${id}/questions`);

        console.log("Fetched questions response:", qRes.data);
        setQuestions(Array.isArray(qRes.data) ? qRes.data : []);
      } catch (err) {
        console.error("Failed to load document or sections:", err);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [id]);

  const renderSectionTree = (sections, depth = 0, parentKey = "") =>
    sections.map((section, index) => {
      const sectionKey = section.id;
      const linkToQuiz = section.sub_sections?.length === 0;

      return (
        <li
          key={sectionKey}
          className={`pl-${depth * 4} border-l-2 border-blue-400 ml-2`}
        >
          {linkToQuiz ? (
            <Link
              to={`/documents/${id}/sections/${sectionKey}`}
              className="text-blue-600 hover:underline"
            >
              {section.title}
            </Link>
          ) : (
            <span className="font-semibold">{section.title}</span>
          )}

          <ul className="ml-4 mt-1">
            {section.sub_sections?.length > 0 &&
              renderSectionTree(section.sub_sections, depth + 1, sectionKey)}
          </ul>
        </li>
      );
    });

  const startQuiz = async () => {
    try {
      const res = await axios.post(`${API_BASE}/quiz-sessions/`, {
        document_id: id,
        num_questions: 5,
      });
      navigate(`/quiz/${res.data.session_id}`);
    } catch (err) {
      console.error("Failed to start quiz:", err);
    }
  };

  if (loading) return <div className="p-6">Loading document...</div>;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">📘 {title}</h1>

      {Object.keys(objectives).length > 0 && (
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-2">
            Global Learning Objectives
          </h2>
          <ul className="list-disc list-inside text-gray-700">
            {Object.entries(objectives).map(([num, goal]) => (
              <li key={num}>{goal}</li>
            ))}
          </ul>
        </div>
      )}

      <h2 className="text-lg font-semibold mb-2">Sections</h2>
      <ul className="space-y-2">{renderSectionTree(sections)}</ul>
      {questions !== null && questions.length === 0 ? (
        <button
          onClick={async () => {
            setBuilding(true);
            await axios.post(
              `${API_BASE}/documents/${id}/questions/generate-all`
            );
            const fresh = await axios.get(
              `${API_BASE}/documents/${id}/questions`
            );
            setQuestions(Array.isArray(fresh.data) ? fresh.data : []);
            setBuilding(false);
          }}
          disabled={building}
          className="mt-6 px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded disabled:opacity-50"
        >
          {building ? "Building quiz…" : "Build Quiz"}
        </button>
      ) : (
        <button
          onClick={startQuiz}
          className="mt-6 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded"
        >
          Start Full Quiz
        </button>
      )}
    </div>
  );
}
