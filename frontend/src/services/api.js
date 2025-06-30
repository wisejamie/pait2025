import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const fetchDocuments = async () => {
  const response = await axios.get(`${API_BASE}/documents`);
  return response.data;
};
