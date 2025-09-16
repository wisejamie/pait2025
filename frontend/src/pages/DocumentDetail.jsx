import { useParams, Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import {
  askTutor,
  createTutorSession,
  nextTutorTurn,
  fetchSummary,
  streamSummary,
} from "../services/api";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function DocumentDetail() {
  const { id } = useParams();
  const [mode, setMode] = useState("advanced");
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
  const [currentSectionId, setCurrentSectionId] = useState(null);
  const [chatHistory, setChatHistory] = useState([]); // [{role:"user"|"assistant", text:"..."}]
  const [chatInput, setChatInput] = useState("");
  const [sending, setSending] = useState(false);
  const navigate = useNavigate();

  // Tutor test session state (Tom’s 3-question flow)
  const [tutorSessionId, setTutorSessionId] = useState(null);
  const [tutorPhase, setTutorPhase] = useState(null); // "offer_test"|"ask_q"|"eval_q"|"wrap_up"|"done"|...
  const [tutorProgress, setTutorProgress] = useState(null); // {q_index,total,...}
  const [inTestMode, setInTestMode] = useState(false);

  // Summary state (Basic)
  const [summary, setSummary] = useState("");
  const [summaryCached, setSummaryCached] = useState(false);
  const [summaryStreaming, setSummaryStreaming] = useState(false);
  const [summaryError, setSummaryError] = useState(null);
  const [summaryES, setSummaryES] = useState(null);

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
    // Cleanup any open EventSource on id change/unmount
    return () => {
      if (summaryES) {
        summaryES.close?.();
      }
    };
  }, [summaryES]);

  useEffect(() => {
    async function load() {
      try {
        const docRes = await axios.get(`${API_BASE}/documents/${id}`);
        setTitle(docRes.data.title || "Untitled");
        setMode(docRes.data.mode || "advanced");

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
        // choose a sensible default section for chat
        if ((response.sections || []).length > 0) {
          setCurrentSectionId(response.sections[0].id);
        }

        // --- BASIC: fetch or stream the summary ---
        if ((docRes.data.mode || "advanced") === "basic") {
          setSummary("");
          setSummaryCached(false);
          setSummaryStreaming(false);
          setSummaryError(null);
          try {
            const s = await fetchSummary(id);
            if (s.cached && s.summary) {
              setSummary(s.summary);
              setSummaryCached(true);
            } else {
              // Start SSE stream
              setSummaryStreaming(true);
              const es = streamSummary(id, {
                onChunk: (chunk) => setSummary((prev) => prev + chunk),
                onDone: () => {
                  setSummaryStreaming(false);
                  setSummaryCached(true); // it’s now stored server-side
                  setSummaryES(null);
                },
                onError: (err) => {
                  setSummaryStreaming(false);
                  setSummaryError(err || "Stream error");
                  setSummaryES(null);
                },
              });
              setSummaryES(es);
            }
          } catch (e) {
            setSummaryError("Failed to load summary.");
          }
        }
      } catch (err) {
        console.error("Failed to load document or sections:", err);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [id]);

  // ADVANCED: links to section pages
  const renderSectionTreeAdvanced = (sections, depth = 0) =>
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
              {renderSectionTreeAdvanced(section.sub_sections, depth + 1)}
            </ul>
          )}
        </li>
      );
    });

  const startThreeQTest = async () => {
    if (!currentSectionId) return;
    try {
      const ses = await createTutorSession(id, currentSectionId);
      setTutorSessionId(ses.session_id);
      setTutorPhase(ses.phase);
      setTutorProgress(ses.progress || null);
      setInTestMode(true);
      // Show the offer message in the chat stream
      setChatHistory((h) => [
        ...h,
        { role: "assistant", text: ses.assistant_msg },
      ]);
    } catch (e) {
      setChatHistory((h) => [
        ...h,
        { role: "assistant", text: "Couldn’t start the test right now." },
      ]);
      console.error(e);
    }
  };

  // Send a turn to the tutor-session state machine (yes/no or an answer)
  const sendTutorTurn = async (user_turn) => {
    if (!tutorSessionId) return;
    try {
      const res = await nextTutorTurn(tutorSessionId, user_turn);
      setTutorPhase(res.phase);
      setTutorProgress(res.progress || null);
      // If we just answered a question, we already appended the user message in chat; now add assistant reply:
      setChatHistory((h) => [
        ...h,
        { role: "assistant", text: res.assistant_msg },
      ]);
      if (res.done) {
        setInTestMode(false); // session concluded
      }
    } catch (e) {
      setChatHistory((h) => [
        ...h,
        { role: "assistant", text: "Ran into an error continuing the test." },
      ]);
      console.error(e);
    }
  };

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

  // BASIC: buttons select current section for the tutor
  const renderSectionTreeBasic = (sections, depth = 0) =>
    sections.map((section) => {
      const sectionKey = section.id;
      const isActive = currentSectionId === sectionKey;
      return (
        <li key={sectionKey} className="ml-2">
          <button
            onClick={() => setCurrentSectionId(sectionKey)}
            className={`text-left w-full px-2 py-1 rounded ${
              isActive
                ? "bg-blue-50 border-l-4 border-blue-500 font-semibold"
                : "hover:bg-gray-50"
            }`}
            style={{ paddingLeft: depth * 12 + 8 }}
          >
            {section.title}
          </button>
          {section.sub_sections?.length > 0 && (
            <ul className="mt-1">
              {renderSectionTreeBasic(section.sub_sections, depth + 1)}
            </ul>
          )}
        </li>
      );
    });

  // BASIC: send a chat turn scoped to current section
  const sendChat = async () => {
    if (!currentSectionId || !chatInput.trim() || sending) return;
    const question = chatInput.trim();
    setChatHistory((h) => [...h, { role: "user", text: question }]);
    setChatInput("");
    setSending(true);
    try {
      if (
        inTestMode &&
        tutorSessionId &&
        (tutorPhase === "ask_q" || tutorPhase === "offer_test")
      ) {
        // Treat this user message as the test answer (or yes/no in offer phase)
        await sendTutorTurn(question);
      } else {
        // Regular tutoring turn
        const resp = await askTutor(id, {
          question,
          context: "section",
          section_id: currentSectionId,
          history: chatHistory,
        });
        setChatHistory((h) => [...h, { role: "assistant", text: resp.answer }]);
      }
    } catch (e) {
      setChatHistory((h) => [
        ...h,
        {
          role: "assistant",
          text: "Sorry — I ran into an error answering that.",
        },
      ]);
      console.error(e);
    } finally {
      setSending(false);
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

  // BASIC LAYOUT
  // BASIC LAYOUT with Summary panel at the top
  if (mode === "basic") {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-4">📘 {title}</h1>

        {/* Overall Summary */}
        <div className="mb-4 bg-white rounded shadow p-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold">Overall Summary</h2>
            {summaryStreaming && (
              <span className="text-xs text-gray-500">Streaming…</span>
            )}
          </div>
          {summaryError ? (
            <div className="text-sm text-red-600">
              {summaryError}{" "}
              <span className="text-gray-500">(try reloading)</span>
            </div>
          ) : summary ? (
            <div className="prose max-w-none whitespace-pre-wrap">
              {summary}
            </div>
          ) : (
            <div className="animate-pulse text-gray-500 text-sm">
              Preparing a concise summary of the document…
            </div>
          )}
        </div>

        {/* Optional: Global learning objectives */}
        {objectives?.objectives?.length ? (
          <div className="mb-4 bg-white rounded shadow p-4">
            <h2 className="text-lg font-semibold mb-2">
              Global Learning Objectives
            </h2>
            <ul className="list-disc list-inside text-gray-700">
              {objectives.objectives.map((goal, i) => (
                <li key={i}>{goal}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* TOC + Quiz controls */}
          <div className="md:col-span-1 bg-white rounded shadow p-3">
            <h3 className="font-semibold mb-2">Sections</h3>
            <ul className="space-y-1">{renderSectionTreeBasic(sections)}</ul>
            <div className="mt-4 space-y-2">
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
                  className="w-full px-3 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded disabled:opacity-50"
                >
                  {building ? "Building quiz…" : "Build Quiz"}
                </button>
              ) : (
                <button
                  onClick={async () => {
                    const res = await axios.post(`${API_BASE}/quiz-sessions/`, {
                      document_id: id,
                      num_questions: 5,
                    });
                    navigate(`/quiz/${res.data.session_id}`);
                  }}
                  className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded"
                >
                  Start Full Quiz
                </button>
              )}
              <button
                onClick={startThreeQTest}
                disabled={!currentSectionId || inTestMode}
                className="w-full px-3 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded disabled:opacity-50"
              >
                Test me on this section (3 Qs)
              </button>
            </div>
          </div>

          {/* Tutor Chat */}
          <div className="md:col-span-2 bg-white rounded shadow p-3 flex flex-col">
            <div className="mb-2 text-sm text-gray-600">
              {currentSectionId
                ? "Chatting about the selected section."
                : "Select a section to begin."}
            </div>
            <div className="flex-1 overflow-y-auto border rounded p-3 space-y-3">
              {chatHistory.length === 0 ? (
                <div className="text-gray-500 text-sm">
                  Tip: Ask things like “Explain this section”, “Give me an
                  example”, or “Why does this matter?”
                </div>
              ) : (
                chatHistory.map((m, i) => (
                  <div
                    key={i}
                    className={m.role === "user" ? "text-right" : "text-left"}
                  >
                    <div
                      className={`inline-block px-3 py-2 rounded ${
                        m.role === "user"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 text-gray-900"
                      }`}
                    >
                      {m.text}
                    </div>
                  </div>
                ))
              )}
            </div>
            {/* Quick replies for test offer/ask phases */}
            {inTestMode && (
              <div className="mt-2 flex flex-wrap gap-2 text-sm">
                {tutorPhase === "offer_test" ? (
                  <>
                    <button
                      onClick={() => {
                        setChatHistory((h) => [
                          ...h,
                          { role: "user", text: "Yes" },
                        ]);
                        sendTutorTurn("yes");
                      }}
                      className="px-2 py-1 border rounded hover:bg-gray-50"
                    >
                      Yes
                    </button>
                    <button
                      onClick={() => {
                        setChatHistory((h) => [
                          ...h,
                          { role: "user", text: "No" },
                        ]);
                        sendTutorTurn("no");
                      }}
                      className="px-2 py-1 border rounded hover:bg-gray-50"
                    >
                      Not now
                    </button>
                  </>
                ) : tutorPhase === "ask_q" ? (
                  <>
                    <span className="text-gray-500 self-center">
                      Answer in the box below, or
                    </span>
                    <button
                      onClick={() => {
                        setChatHistory((h) => [
                          ...h,
                          { role: "user", text: "Skip" },
                        ]);
                        sendTutorTurn("Skip");
                      }}
                      className="px-2 py-1 border rounded hover:bg-gray-50"
                    >
                      Skip
                    </button>
                  </>
                ) : null}
              </div>
            )}
            <div className="mt-3 flex gap-2">
              <input
                className="flex-1 border rounded px-3 py-2"
                placeholder={
                  inTestMode
                    ? "Type your answer…"
                    : "Ask a question about this section…"
                }
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") sendChat();
                }}
                disabled={!currentSectionId || sending}
              />
              <button
                onClick={sendChat}
                disabled={!currentSectionId || sending || !chatInput.trim()}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded disabled:opacity-50"
              >
                {sending ? "Sending…" : inTestMode ? "Submit answer" : "Send"}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
      <ul className="space-y-2">{renderSectionTreeAdvanced(sections)}</ul>
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
