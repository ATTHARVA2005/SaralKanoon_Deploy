// src/pages/PricingPage.tsx
import { Header } from '../components/shared/Header';

export const PricingPage = () => (
  <div className="min-h-screen bg-background">
    <Header />
    <main className="max-w-3xl mx-auto py-16 px-4">
      <h2 className="text-3xl font-bold mb-6">Pricing</h2>
      <div className="grid gap-8 md:grid-cols-3">
        <div className="bg-card p-6 rounded-xl shadow flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-2">Free</h3>
          <p className="mb-4">Basic document analysis and chatbot access.</p>
          <span className="text-2xl font-bold mb-2">₹0</span>
          <ul className="text-sm mb-4 list-disc pl-4 text-left">
            <li>Single document upload</li>
            <li>Basic clause extraction</li>
            <li>AI chatbot (limited)</li>
          </ul>
          <button className="bg-gradient-to-r from-gradient-start to-gradient-end text-white px-4 py-2 rounded-xl mt-auto">Get Started</button>
        </div>
        <div className="bg-card p-6 rounded-xl shadow flex flex-col items-center border-2 border-accent">
          <h3 className="text-xl font-semibold mb-2">Pro</h3>
          <p className="mb-4">Advanced features for professionals and teams.</p>
          <span className="text-2xl font-bold mb-2">₹499/mo</span>
          <ul className="text-sm mb-4 list-disc pl-4 text-left">
            <li>Multiple document uploads</li>
            <li>Clause comparison</li>
            <li>Multilingual support</li>
            <li>Unlimited chatbot queries</li>
          </ul>
          <button className="bg-gradient-to-r from-gradient-start to-gradient-end text-white px-4 py-2 rounded-xl mt-auto">Start Pro Trial</button>
        </div>
        <div className="bg-card p-6 rounded-xl shadow flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-2">Enterprise</h3>
          <p className="mb-4">Custom solutions for large organizations.</p>
          <span className="text-2xl font-bold mb-2">Contact Us</span>
          <ul className="text-sm mb-4 list-disc pl-4 text-left">
            <li>Custom integrations</li>
            <li>Dedicated support</li>
            <li>API access</li>
          </ul>
          <button className="bg-gradient-to-r from-gradient-start to-gradient-end text-white px-4 py-2 rounded-xl mt-auto">Contact Sales</button>
        </div>
      </div>
    </main>
  </div>
);

export default PricingPage;
