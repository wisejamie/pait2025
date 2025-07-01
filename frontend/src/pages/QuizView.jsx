import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getNextQuestion, submitAnswer } from "../services/api";

export default function QuizView() {
  const { sessionId } = useParams();
  const navigate = useNavigate();

  const [questionData, setQuestionData] = useState(null);
  const [selected, setSelected] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [feedback, setFeedback] = useState(null);

  useEffect(() => {
    async function fetchNext() {
      try {
        const data = await getNextQuestion(sessionId);
        if (data.finished) {
          navigate(`/quiz/${sessionId}/summary`);
        } else {
          setQuestionData(data);
          setSelected(null);
          setSubmitted(false);
          setFeedback(null);
        }
      } catch (err) {
        console.error("Error fetching question:", err);
      }
    }
    fetchNext();
  }, [sessionId]);

  async function handleSubmit() {
    if (selected === null || submitted) return;

    const response = await submitAnswer(sessionId, {
      question_id: questionData.question_id,
      selected_index: selected,
    });

    setFeedback(response);
    setSubmitted(true);

    if (response.completed) {
      setTimeout(() => navigate(`/quiz/${sessionId}/summary`), 2000);
    } else {
      setTimeout(() => window.location.reload(), 1500); // simplistic
    }
  }

  if (!questionData) return <p>Loading question...</p>;

  return (
    <div className="max-w-xl mx-auto mt-8 p-4 bg-white rounded shadow">
      <h2 className="text-xl font-bold mb-4">
        Question {questionData.index + 1} of {questionData.total}
      </h2>
      <p className="mb-4">{questionData.question.question_text}</p>
      <ul className="space-y-2">
        {questionData.question.options.map((opt, i) => (
          <li
            key={i}
            className={`p-2 border rounded cursor-pointer ${
              selected === i ? "bg-blue-100" : ""
            }`}
            onClick={() => setSelected(i)}
          >
            {opt}
          </li>
        ))}
      </ul>
      <button
        onClick={handleSubmit}
        disabled={submitted}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
      >
        Submit
      </button>

      {feedback && (
        <div className="mt-4 p-4 rounded bg-gray-100">
          <p>
            {feedback.correct ? "✅ Correct!" : "❌ Incorrect."} Explanation:{" "}
            {feedback.explanation}
          </p>
        </div>
      )}
    </div>
  );
}
