// src/pages/GetStartedPage.tsx
import { Header } from '../components/shared/Header';
import { Button } from '../components/ui/Button';

export const GetStartedPage = () => (
  <div className="min-h-screen bg-background">
    <Header />
    <main className="max-w-2xl mx-auto py-16 px-4 text-center">
      <h2 className="text-4xl font-bold mb-6">Get Started with LawSimplify</h2>
      <p className="mb-8 text-lg">Upload your legal document and let our AI analyze it for key clauses, risks, and obligations. Start simplifying your legal work today!</p>
      <Button size="lg" className="bg-gradient-to-r from-gradient-start to-gradient-end text-white px-8 py-4 rounded-xl text-xl">Upload Document</Button>
      <div className="mt-12 text-text-secondary">
        <p>Need help? Visit our <a href="#" className="text-accent underline">Features</a> or <a href="#" className="text-accent underline">About</a> page.</p>
      </div>
    </main>
  </div>
);

export default GetStartedPage;
