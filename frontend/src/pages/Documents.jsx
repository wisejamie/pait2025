import { useEffect, useState } from "react";
import {
  fetchDocuments,
  createDocument,
  uploadDocumentFile,
  detectSections,
  deleteDocument,
} from "../services/api";
import { Link } from "react-router-dom";

export default function Documents() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setModalOpen] = useState(false);
  const [uploadTitle, setUploadTitle] = useState("");
  const [uploadText, setUploadText] = useState("");
  const [uploadFile, setUploadFile] = useState(null);
  const [isUploading, setUploading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);

  const openModal = () => setModalOpen(true);
  const closeModal = () => {
    setModalOpen(false);
    setUploadText("");
    setUploadFile(null);
  };

  const loadDocuments = async () => {
    setLoading(true);
    try {
      const data = await fetchDocuments();
      setDocuments(data);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    setUploading(true);
    try {
      // 1) create or upload
      const res = uploadFile
        ? await uploadDocumentFile(uploadFile, uploadTitle)
        : await createDocument({ text: uploadText, title: uploadTitle });
      const docId = res.document_id; // <— use the correct field
      // 2) auto-detect sections
      await detectSections(docId);
      // 3) refresh list & close
      await loadDocuments();
      closeModal();
    } catch (err) {
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId) => {
    if (!window.confirm("Are you sure you want to delete this document?")) {
      return;
    }
    setDeletingId(docId);
    try {
      await deleteDocument(docId);
      await loadDocuments();
    } catch (err) {
      console.error("Delete failed:", err);
      alert("Failed to delete document");
    } finally {
      setDeletingId(null);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  if (loading) return <div className="p-6">Loading documents...</div>;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">📄 Documents</h1>
      {documents.length === 0 ? (
        <p>No documents found.</p>
      ) : (
        <ul className="space-y-4">
          <button
            onClick={openModal}
            className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded shadow"
          >
            Upload New Document
          </button>
          {documents.map((doc) => (
            <li
              key={doc.document_id}
              className="p-4 bg-white shadow rounded flex items-center justify-between"
            >
              <div>
                <Link
                  to={`/documents/${doc.document_id}`}
                  className="text-blue-600 hover:underline"
                >
                  {doc.title || "Untitled Document"}
                </Link>
                <p className="text-sm text-gray-500">
                  Uploaded on: {new Date(doc.upload_time).toLocaleString()}
                </p>
              </div>
              <button
                onClick={() => handleDelete(doc.document_id)}
                disabled={deletingId === doc.document_id}
                className="bg-red-500 hover:bg-red-600 text-white font-semibold py-1 px-3 rounded ml-4 disabled:opacity-50"
              >
                {deletingId === doc.document_id ? "Deleting…" : "Delete"}
              </button>
            </li>
          ))}
        </ul>
      )}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-lg">
            <h2 className="text-xl font-semibold mb-4">Upload Document</h2>
            <label
              className="block text-sm font-medium mb-1"
              htmlFor="doc-title"
            >
              Title <span className="text-red-500">*</span>
            </label>
            <input
              id="doc-title"
              type="text"
              value={uploadTitle}
              onChange={(e) => setUploadTitle(e.target.value)}
              disabled={isUploading}
              className="w-full border border-gray-300 rounded p-2 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-400"
              placeholder="Enter document title"
            />
            <textarea
              placeholder="Paste text here or upload file below"
              value={uploadText}
              onChange={(e) => setUploadText(e.target.value)}
              disabled={isUploading}
              className="w-full h-32 border border-gray-300 rounded p-2 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={(e) => setUploadFile(e.target.files[0])}
              disabled={isUploading}
              className="block w-full text-gray-700 mb-4"
            />
            <button
              onClick={handleUpload}
              disabled={
                isUploading ||
                !uploadTitle.trim() ||
                (!uploadText && !uploadFile)
              }
              className="bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded mr-2 disabled:opacity-50"
            >
              {isUploading ? "Processing…" : "Upload & Process"}
            </button>
            <button
              onClick={closeModal}
              disabled={isUploading}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
