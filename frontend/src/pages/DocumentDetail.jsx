import { useParams, Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function DocumentDetail() {
  const { id } = useParams();
  const [selectedSections, setSelectedSections] = useState([]);
  const [questionCount, setQuestionCount] = useState(5);
  const [allQuestionsCount, setAllQuestionsCount] = useState(0);
  const [isCustomizeOpen, setCustomizeOpen] = useState(false);
  const [sections, setSections] = useState([]);
  const [title, setTitle] = useState("");
  const [objectives, setObjectives] = useState({});
  const [questions, setQuestions] = useState(null);
  const [building, setBuilding] = useState(false);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const getDescendantIds = (section) => {
    let ids = [section.id];
    if (section.sub_sections) {
      section.sub_sections.forEach((sub) => {
        ids = ids.concat(getDescendantIds(sub));
      });
    }
    return ids;
  };

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
        const qRes = await axios.get(`${API_BASE}/documents/${id}/questions`);
        setAllQuestionsCount(qRes.data.length);
        console.log("Fetched questions response:", qRes.data);
        setQuestions(Array.isArray(qRes.data) ? qRes.data : []);

        const allIds = [];
        const collect = (secs) => {
          secs.forEach((s) => {
            allIds.push(s.id);
            if (s.sub_sections) collect(s.sub_sections);
          });
        };
        collect(response.sections || []);
        setSelectedSections(allIds);
      } catch (err) {
        console.error("Failed to load document or sections:", err);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [id]);

  // Render all sections as clickable links
  const renderSectionTree = (sections, depth = 0) =>
    sections.map((section) => {
      const sectionKey = section.id;
      return (
        <li
          key={sectionKey}
          className={`pl-${depth * 4} border-l-2 border-blue-400 ml-2`}
        >
          <Link
            to={`/documents/${id}/sections/${sectionKey}`}
            className="text-blue-600 hover:underline font-semibold"
          >
            {section.title}
          </Link>

          {section.sub_sections?.length > 0 && (
            <ul className="ml-4 mt-1">
              {renderSectionTree(section.sub_sections, depth + 1)}
            </ul>
          )}
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

  const renderCheckboxTree = (section, depth = 0) => {
    const ids = getDescendantIds(section);
    const allSelected = ids.every((id) => selectedSections.includes(id));
    return (
      <div
        key={section.id}
        style={{ paddingLeft: depth * 16 }}
        className="mb-1"
      >
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={allSelected}
            onChange={(e) => {
              if (e.target.checked) {
                setSelectedSections((prev) =>
                  Array.from(new Set([...prev, ...ids]))
                );
              } else {
                setSelectedSections((prev) =>
                  prev.filter((id) => !ids.includes(id))
                );
              }
            }}
            className="mr-2"
          />
          {section.title}
        </label>
        {section.sub_sections?.map((sub) => renderCheckboxTree(sub, depth + 1))}
      </div>
    );
  };

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
      <button
        onClick={() => navigate(`/documents/${id}/ask`)}
        className="mt-6 ml-0 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded"
      >
        Ask the Tutor
      </button>
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
      {questions !== null && questions.length > 0 && !isCustomizeOpen && (
        <button
          onClick={() => setCustomizeOpen(true)}
          className="mt-6 px-4 py-2 bg-blue-600 text-white rounded"
        >
          Customize Quiz
        </button>
      )}

      {isCustomizeOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center">
          <div className="bg-white p-6 rounded shadow-lg w-full max-w-md">
            <h2 className="text-xl font-semibold mb-4">Customize Quiz</h2>

            <div className="mb-4 max-h-64 overflow-y-auto border p-2 rounded">
              {sections.map((sec) => renderCheckboxTree(sec))}
            </div>

            <div className="mb-4">
              <label className="block mb-1">
                Number of questions (1–{allQuestionsCount}):
              </label>
              <input
                type="number"
                min={1}
                max={allQuestionsCount}
                value={questionCount}
                onChange={(e) => setQuestionCount(+e.target.value)}
                className="border p-1 w-20"
              />
            </div>

            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setCustomizeOpen(false)}
                className="px-4 py-2 bg-gray-200 rounded"
              >
                Cancel
              </button>
              <button
                disabled={
                  questionCount < 1 ||
                  questionCount > allQuestionsCount ||
                  selectedSections.length === 0
                }
                onClick={async () => {
                  const { data } = await axios.post(
                    `${API_BASE}/quiz-sessions/`,
                    {
                      document_id: id,
                      num_questions: questionCount,
                      sections: selectedSections,
                    }
                  );
                  navigate(`/quiz/${data.session_id}`);
                }}
                className="px-4 py-2 bg-green-600 text-white rounded disabled:opacity-50"
              >
                Start Quiz
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
