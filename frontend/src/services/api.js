import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const fetchDocuments = async () => {
  const response = await axios.get(`${API_BASE}/documents`);
  return response.data;
};

export const createDocument = async ({ text, title, mode = "basic" }) => {
  const payload = { raw_text: text, mode };
  if (title) payload.title = title;
  const { data } = await axios.post(`${API_BASE}/documents/`, payload);
  return data;
};

export const uploadDocumentFile = async (file, title, mode = "basic") => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("title", title);
  formData.append("mode", mode);

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

export async function askTutor(
  documentId,
  { question, context = "document", section_id = null }
) {
  const { data } = await axios.post(`${API_BASE}/documents/${documentId}/ask`, {
    question,
    context,
    section_id,
  });
  return data; // { answer: string }
}

export async function createTutorSession(document_id, section_id = null) {
  const res = await axios.post(`${API_BASE}/tutor-sessions/`, {
    document_id,
    section_id,
  });
  return res.data; // {session_id, phase, assistant_msg, ...}
}

export async function nextTutorTurn(session_id, user_turn = null) {
  const res = await axios.post(
    `${API_BASE}/tutor-sessions/${session_id}/next`,
    {
      user_turn,
    }
  );
  return res.data;
}

// --- SUMMARY (Basic) ---
export async function fetchSummary(docId) {
  const { data } = await axios.get(`${API_BASE}/documents/${docId}/summary`);
  return data; // { summary: string|null, cached: boolean }
}

// Server-Sent Events stream; returns the EventSource so caller can close it
export function streamSummary(docId, { onChunk, onDone, onError } = {}) {
  const es = new EventSource(`${API_BASE}/documents/${docId}/summary/stream`);
  es.onmessage = (evt) => {
    // default SSE "message" → deliver text chunks
    if (onChunk) onChunk(evt.data || "");
  };
  es.addEventListener("done", () => {
    if (onDone) onDone();
    es.close();
  });
  es.addEventListener("error", (evt) => {
    try {
      const payload = JSON.parse(evt.data || "{}");
      if (onError) onError(payload?.error || "stream error");
    } catch {
      if (onError) onError("stream error");
    }
    es.close();
  });
  return es;
}
