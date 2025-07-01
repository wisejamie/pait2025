import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getQuizSummary } from "../services/api";

export default function QuizSummary() {
  const { sessionId } = useParams();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadSummary = async () => {
      try {
        const res = await getQuizSummary(sessionId);
        setSummary(res);
      } catch (err) {
        console.error("Failed to load quiz summary:", err);
      } finally {
        setLoading(false);
      }
    };
    loadSummary();
  }, [sessionId]);

  if (loading) return <div>Loading summary...</div>;
  if (!summary || !summary.finished)
    return <div>Summary not available yet.</div>;

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-2">Quiz Summary</h1>
      <p>Correct: {summary.correct}</p>
      <p>Incorrect: {summary.incorrect}</p>
      <p>Score: {summary.score_percent}%</p>

      <h2 className="text-xl mt-4 font-semibold">Missed Questions</h2>
      {summary.missed_questions.length === 0 ? (
        <p className="text-green-700">Nice! You got everything right.</p>
      ) : (
        <ul className="mt-2 space-y-4">
          {summary.missed_questions.map((q, idx) => (
            <li key={q.question_id} className="border p-3 rounded bg-red-50">
              <p className="font-semibold">{q.question_text}</p>
              <ul className="list-disc ml-5">
                {q.options.map((opt, i) => (
                  <li
                    key={i}
                    className={
                      i === q.correct_index
                        ? "text-green-700"
                        : i === q.selected_index
                        ? "text-red-700"
                        : ""
                    }
                  >
                    {opt}
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-sm italic text-gray-700">
                {q.explanation}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
