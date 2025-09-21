// src/pages/AboutPage.tsx
import { Header } from '../components/shared/Header';

export const AboutPage = () => (
  <div className="min-h-screen bg-background">
    <Header />
    <main className="max-w-2xl mx-auto py-16 px-4">
      <h2 className="text-3xl font-bold mb-6">About LawSimplify</h2>
      <p className="mb-6 text-lg">LawSimplify is an AI-powered platform designed to make legal documents easy to understand, analyze, and manage. Our mission is to empower individuals and businesses to navigate legal complexities with confidence and clarity.</p>
      <h3 className="text-xl font-semibold mb-4">Why LawSimplify?</h3>
      <ul className="space-y-4 mb-8">
        <li>✔️ Instant document analysis using advanced AI</li>
        <li>✔️ Multilingual support for diverse users</li>
        <li>✔️ Secure and private document handling</li>
        <li>✔️ User-friendly interface for all experience levels</li>
      </ul>
      <h3 className="text-xl font-semibold mb-4">Contact Us</h3>
      <p>Email: support@lawsimplify.com</p>
      <p>Phone: +91-12345-67890</p>
    </main>
  </div>
);

export default AboutPage;
