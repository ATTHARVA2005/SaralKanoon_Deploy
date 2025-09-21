// src/pages/HomePage.tsx
import { useState } from 'react';
import { Header } from '../components/shared/Header';
import { FileUploader } from '../components/shared/FileUploader';
import { AnalysisDisplay } from '../components/shared/AnalysisDisplay';
import { Chatbot } from '../components/shared/Chatbot';
import { Loader } from '../components/ui/Loader';
import { analyzeDocument } from '../lib/api';
import type { AnalysisResult } from '../types';
import { Button } from '../components/ui/Button';
import { RotateCcw, Sparkles, Zap, Shield } from 'lucide-react';

export const HomePage = () => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await analyzeDocument(file);
      setAnalysisResult(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze document. Please try again.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-background relative overflow-x-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-background/80 pointer-events-none" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-gradient-start/10 to-gradient-end/10 rounded-full blur-3xl pointer-events-none animate-pulse-slow" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-l from-destructive/10 to-accent/10 rounded-full blur-3xl pointer-events-none animate-pulse-slow" />
      
      <Header />
      
      <main className="relative z-10">
        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center min-h-[80vh]">
            <div className="glass-effect rounded-2xl p-12 text-center glow-effect">
              <Loader size={64} text="Analyzing your legal document with AI..." />
            </div>
          </div>
        )}

        {/* Initial Upload State */}
        {!isLoading && !analysisResult && (
          <div className="px-6 py-16 lg:py-24">
            <div className="max-w-6xl mx-auto">
              <div className="text-center space-y-8 mb-16">
                <div className="space-y-6">
                  <h1 className="text-5xl lg:text-7xl font-bold font-display leading-tight">
                    <span className="gradient-text">AI-Powered</span><br />
                    <span className="text-text-primary">Legal Document</span><br />
                    <span className="text-text-primary">Analysis</span>
                  </h1>
                  <p className="text-xl lg:text-2xl text-text-secondary max-w-3xl mx-auto leading-relaxed font-medium">
                    Upload your legal document and get instant AI-powered insights with 
                    <span className="text-accent font-semibold"> key clause identification</span>, 
                    <span className="text-destructive font-semibold"> red flag detection</span>, and 
                    <span className="text-gradient-start font-semibold"> plain-English summaries</span>.
                  </p>
                </div>
                
                {/* Feature Cards */}
                <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto mt-16 mb-16">
                  <div className="glass-effect rounded-xl p-6 text-center group hover:glow-effect transition-all duration-300">
                    <div className="w-12 h-12 bg-gradient-to-r from-accent to-gradient-end rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                      <Sparkles className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-lg font-semibold text-text-primary mb-2">Smart Analysis</h3>
                    <p className="text-text-secondary text-sm">Advanced AI understands legal language and context</p>
                  </div>
                  
                  <div className="glass-effect rounded-xl p-6 text-center group hover:glow-effect transition-all duration-300">
                    <div className="w-12 h-12 bg-gradient-to-r from-destructive to-red-500 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                      <Shield className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-lg font-semibold text-text-primary mb-2">Risk Detection</h3>
                    <p className="text-text-secondary text-sm">Automatically identify potential legal red flags</p>
                  </div>
                  
                  <div className="glass-effect rounded-xl p-6 text-center group hover:glow-effect transition-all duration-300">
                    <div className="w-12 h-12 bg-gradient-to-r from-gradient-start to-purple-500 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                      <Zap className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-lg font-semibold text-text-primary mb-2">Instant Results</h3>
                    <p className="text-text-secondary text-sm">Get comprehensive analysis in seconds</p>
                  </div>
                </div>
              </div>
              
              <FileUploader
                onFileUpload={handleFileUpload}
                isLoading={isLoading}
                error={error}
              />
            </div>
          </div>
        )}

        {/* Analysis Results State */}
        {!isLoading && analysisResult && (
          <div className="px-6 py-12">
            <div className="max-w-7xl mx-auto space-y-12">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
                <div>
                  <h1 className="text-4xl font-bold font-display gradient-text mb-2">
                    Analysis Complete
                  </h1>
                  <p className="text-text-secondary text-lg">
                    Your document has been thoroughly analyzed by our AI
                  </p>
                </div>
                <Button
                  onClick={handleReset}
                  variant="secondary"
                  size="lg"
                  className="flex items-center gap-3 glass-effect hover:glow-effect transition-all"
                >
                  <RotateCcw className="w-5 h-5" />
                  Analyze Another Document
                </Button>
              </div>
              
              <AnalysisDisplay analysisResult={analysisResult} />
              
              <div className="max-w-5xl mx-auto">
                <Chatbot />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};