import { useEffect, useState } from "react";
import {
  fetchDocuments,
  createDocument,
  uploadDocumentFile,
  detectSections,
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
  const [uploadMode, setUploadMode] = useState("basic"); // "basic" | "advanced"

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
      // Advanced requires a PDF file; if user picked Advanced but pasted text, fall back to Basic.
      let effectiveMode = uploadMode;
      if (!uploadFile && uploadMode === "advanced") {
        effectiveMode = "basic";
        console.warn(
          "Advanced mode needs a PDF; falling back to Basic for pasted text."
        );
      }

      const res = uploadFile
        ? await uploadDocumentFile(uploadFile, uploadTitle, effectiveMode)
        : await createDocument({ text: uploadText, title: uploadTitle }); // treated as Basic on backend
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
            <li key={doc.document_id} className="p-4 bg-white shadow rounded">
              <Link
                to={`/documents/${doc.document_id}`}
                className="text-blue-600 hover:underline"
              >
                {doc.title || "Untitled Document"}
              </Link>
              <p className="text-sm text-gray-500">
                Uploaded on: {new Date(doc.upload_time).toLocaleString()}
              </p>
            </li>
          ))}
        </ul>
      )}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-lg">
            <h2 className="text-xl font-semibold mb-4">Upload Document</h2>

            {/* Mode selector */}
            <div className="mb-4 border rounded p-3">
              <p className="text-sm font-medium mb-2">Processing Mode</p>
              <div className="space-y-2">
                <label className="flex items-start gap-2">
                  <input
                    type="radio"
                    name="mode"
                    value="basic"
                    checked={uploadMode === "basic"}
                    onChange={() => setUploadMode("basic")}
                    disabled={isUploading}
                  />
                  <div>
                    <div className="font-medium">Basic (recommended)</div>
                    <div className="text-xs text-gray-600">
                      Fast & reliable on most PDFs. Uses text-only extraction
                      and GPT for understanding. Doesn’t render the original PDF
                      formatting in-app.
                    </div>
                  </div>
                </label>
                <label className="flex items-start gap-2">
                  <input
                    type="radio"
                    name="mode"
                    value="advanced"
                    checked={uploadMode === "advanced"}
                    onChange={() => setUploadMode("advanced")}
                    disabled={isUploading}
                  />
                  <div>
                    <div className="font-medium">Advanced</div>
                    <div className="text-xs text-gray-600">
                      Extracts cleaned Markdown (tables/figures) for in-app
                      reading. Slower and works best on simple, single-column
                      PDFs.
                    </div>
                  </div>
                </label>
              </div>
            </div>
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
              disabled={isUploading || uploadMode === "advanced"}
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
                // If Advanced is selected, require a file; if Basic, allow either.
                (uploadMode === "advanced"
                  ? !uploadFile
                  : !uploadText && !uploadFile)
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
