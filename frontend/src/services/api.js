import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const fetchDocuments = async () => {
  const response = await axios.get(`${API_BASE}/documents`);
  return response.data;
};

export async function getNextQuestion(sessionId) {
  const res = await axios.get(`${API_BASE}/quiz-sessions/${sessionId}/next`);
  return res.data;
}

export async function submitAnswer(sessionId, payload) {
  const res = await axios.post(
    `${API_BASE}/quiz-sessions/${sessionId}/answer`,
    payload
  );
  return res.data;
}

export async function getQuizSummary(sessionId) {
  const response = await axios.get(
    `${API_BASE}/quiz-sessions/${sessionId}/summary`
  );
  return response.data;
}
