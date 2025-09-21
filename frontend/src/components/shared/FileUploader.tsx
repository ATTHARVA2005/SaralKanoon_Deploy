// src/components/shared/FileUploader.tsx
import { useState, useRef, useCallback } from 'react';
import { Upload, FileText, AlertCircle, Cloud } from 'lucide-react';
import { Button } from '../ui/Button';
import { Loader } from '../ui/Loader';
import { cn, isValidPDF } from '../../lib/utils';

interface FileUploaderProps {
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  error?: string | null;
}

export const FileUploader = ({ onFileUpload, isLoading, error }: FileUploaderProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const file = files[0];
    
    if (!file) return;
    
    if (!isValidPDF(file)) {
      return;
    }
    
    onFileUpload(file);
  }, [onFileUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    if (!isValidPDF(file)) {
      return;
    }
    
    onFileUpload(file);
  }, [onFileUpload]);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[500px]">
        <div className="glass-effect rounded-2xl p-12 text-center glow-effect max-w-md">
          <Loader size={64} />
          <div className="mt-8 space-y-3">
            <h3 className="text-xl font-semibold text-text-primary">Analyzing Document</h3>
            <p className="text-text-secondary">Our AI is processing your legal document...</p>
            <div className="flex items-center justify-center space-x-1 mt-4">
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce delay-0"></div>
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce delay-150"></div>
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce delay-300"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div
        className={cn(
          'relative glass-effect border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-300 cursor-pointer group',
          isDragOver 
            ? 'border-accent bg-accent/5 glow-effect scale-105' 
            : 'border-border hover:border-accent/50 hover:bg-accent/5',
          'hover:glow-effect hover:scale-105'
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleButtonClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
          title="Upload PDF document"
        />
        
        {/* Background Animation */}
        <div className="absolute inset-0 bg-gradient-to-r from-gradient-start/5 to-gradient-end/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        
        <div className="relative z-10 space-y-8">
          <div className="relative mx-auto w-24 h-24">
            <div className={cn(
              "w-24 h-24 rounded-2xl flex items-center justify-center transition-all duration-500",
              isDragOver 
                ? "bg-gradient-to-r from-accent to-gradient-end scale-110" 
                : "bg-gradient-to-r from-gradient-start/20 to-gradient-end/20 group-hover:from-accent/30 group-hover:to-gradient-end/30"
            )}>
              {isDragOver ? (
                <Cloud className="w-12 h-12 text-white animate-bounce-gentle" />
              ) : (
                <Upload className="w-12 h-12 text-accent group-hover:text-white transition-colors" />
              )}
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-gradient-end/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 -z-10" />
          </div>
          
          <div className="space-y-4">
            <h3 className="text-2xl font-bold font-display text-text-primary">
              {isDragOver ? 'Drop your document here' : 'Upload Legal Document'}
            </h3>
            <p className="text-text-secondary text-lg max-w-lg mx-auto leading-relaxed">
              {isDragOver 
                ? 'Release to start analyzing your legal document'
                : 'Drag and drop your PDF document or click to browse files'
              }
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <Button 
              size="lg" 
              className="min-w-[180px] h-12 text-base font-semibold bg-gradient-to-r from-gradient-start to-gradient-end hover:from-gradient-start/90 hover:to-gradient-end/90 glow-effect"
            >
              Browse Files
            </Button>
            <div className="flex items-center gap-2 text-text-secondary">
              <FileText className="w-4 h-4" />
              <span className="text-sm font-mono">PDF files only • Max 25MB</span>
            </div>
          </div>
        </div>
      </div>
      
      {error && (
        <div className="mt-6 glass-effect border border-destructive/30 rounded-xl p-4 bg-destructive/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-destructive font-semibold mb-1">Upload Error</h4>
              <p className="text-destructive/80 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};