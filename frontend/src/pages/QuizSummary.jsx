import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function QuizSummary() {
  const { sessionId } = useParams();
  const [summary, setSummary] = useState(null);
  const [sectionsTree, setSectionsTree] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openSections, setOpenSections] = useState(new Set());

  useEffect(() => {
    async function loadAll() {
      try {
        // 1) load summary (now contains document_id)
        const sum = await axios
          .get(`${API_BASE}/quiz-sessions/${sessionId}/summary`)
          .then((r) => r.data);
        setSummary(sum);

        if (!sum.finished) {
          setLoading(false);
          return;
        }

        // 2) fetch that document’s section tree
        const secRes = await axios.get(
          `${API_BASE}/documents/${sum.document_id}/sections`
        );
        const top = Array.isArray(secRes.data) ? secRes.data[0] : secRes.data;
        setSectionsTree(top.sections || []);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadAll();
  }, [sessionId]);

  if (loading) return <div>Loading summary…</div>;
  if (!summary || !summary.finished) return <div>Quiz not complete yet.</div>;

  // ── Build aggregated stats per node ──
  const scoreMap = {};
  summary.section_scores.forEach((s) => {
    scoreMap[s.section_id] = { correct: s.correct, total: s.total };
  });

  // Recursively build a tree annotating each node with aggCorrect, aggTotal, percent
  const buildTree = (secs) =>
    secs.map((sec) => {
      const children = buildTree(sec.sub_sections || []);
      const self = scoreMap[sec.id] || { correct: 0, total: 0 };
      const childTotals = children.reduce(
        (acc, c) => {
          acc.correct += c.aggCorrect;
          acc.total += c.aggTotal;
          return acc;
        },
        { correct: 0, total: 0 }
      );

      const aggCorrect = self.correct + childTotals.correct;
      const aggTotal = self.total + childTotals.total;
      const percent = aggTotal
        ? Math.round((aggCorrect / aggTotal) * 100 * 100) / 100
        : null;

      return {
        ...sec,
        sub_sections: children,
        aggCorrect,
        aggTotal,
        percent,
      };
    });

  const scoredTree = buildTree(sectionsTree);

  // toggle expand/collapse
  const toggle = (id) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  // recursive renderer
  const renderTree = (secs, depth = 0) => (
    <ul className="ml-4">
      {secs.map((sec) => (
        <li key={sec.id} className="mb-1">
          <div
            className={`flex items-center p-1 rounded 
            ${sec.score ? "bg-blue-50" : "bg-gray-50"}`}
          >
            {sec.sub_sections.length > 0 && (
              <button onClick={() => toggle(sec.id)} className="mr-2 text-sm">
                {openSections.has(sec.id) ? "▼" : "►"}
              </button>
            )}
            <span className="flex-1 font-medium">{sec.title}</span>
            {sec.aggTotal > 0 ? (
              <span className="text-sm">
                {sec.aggCorrect}/{sec.aggTotal} ({sec.percent}%)
              </span>
            ) : (
              <span className="text-sm text-gray-500">No questions</span>
            )}
          </div>

          {sec.sub_sections.length > 0 &&
            openSections.has(sec.id) &&
            renderTree(sec.sub_sections, depth + 1)}
        </li>
      ))}
    </ul>
  );

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-2">Quiz Summary</h1>

      <p>Correct: {summary.correct}</p>
      <p>Incorrect: {summary.incorrect}</p>
      <p>Score: {summary.score_percent}%</p>

      <h2 className="text-xl mt-4 font-semibold">Section-by-Section Scores</h2>
      {renderTree(scoredTree)}

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
