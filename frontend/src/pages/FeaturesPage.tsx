// src/pages/FeaturesPage.tsx
import { Header } from '../components/shared/Header';

export const FeaturesPage = () => (
  <div className="min-h-screen bg-background">
    <Header />
    <main className="max-w-4xl mx-auto py-16 px-4">
      <h2 className="text-3xl font-bold mb-6">Features</h2>
      <ul className="space-y-6">
        <li className="bg-card p-6 rounded-xl shadow">
          <h3 className="text-xl font-semibold mb-2">AI-Powered Document Analysis</h3>
          <p>Instantly analyze legal documents and extract key clauses, obligations, and risks.</p>
        </li>
        <li className="bg-card p-6 rounded-xl shadow">
          <h3 className="text-xl font-semibold mb-2">Clause Comparison</h3>
          <p>Compare clauses across multiple documents for consistency and compliance.</p>
        </li>
        <li className="bg-card p-6 rounded-xl shadow">
          <h3 className="text-xl font-semibold mb-2">Multilingual Support</h3>
          <p>Translate legal sections and get audio explanations in your preferred language.</p>
        </li>
        <li className="bg-card p-6 rounded-xl shadow">
          <h3 className="text-xl font-semibold mb-2">AI Legal Assistant Chatbot</h3>
          <p>Ask questions about your document and get instant, AI-generated answers.</p>
        </li>
      </ul>
    </main>
  </div>
);

export default FeaturesPage;
