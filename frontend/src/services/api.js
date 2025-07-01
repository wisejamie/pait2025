import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const fetchDocuments = async () => {
  const response = await axios.get(`${API_BASE}/documents`);
  return response.data;
};

export const createDocument = async ({ text, title }) => {
  const payload = { raw_text: text };
  if (title) payload.title = title;
  const { data } = await axios.post(`${API_BASE}/documents/`, payload);
  return data;
};

export const uploadDocumentFile = async (file, title) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("title", title);

  const resp = await fetch(`${API_BASE}/documents/upload-file`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(
      `Upload failed (${resp.status}): ${
        err.detail?.[0]?.msg || resp.statusText
      }`
    );
  }

  return await resp.json();
};

export const detectSections = async (docId) => {
  await axios.post(`${API_BASE}/documents/${docId}/sections/detect`);
};

export const fetchQuestions = async (docId) => {
  const { data } = await axios.get(`${API_BASE}/documents/${docId}/questions`);
  return data;
};

export const generateAllQuestions = async (docId) => {
  const { data } = await axios.post(
    `${API_BASE}/documents/${docId}/questions/generate-all`
  );
  return data;
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
