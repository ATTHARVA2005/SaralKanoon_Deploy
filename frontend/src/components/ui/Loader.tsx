// src/components/ui/Loader.tsx
import { Loader2, Sparkles } from 'lucide-react';
import { cn } from '../../lib/utils';

interface LoaderProps {
  size?: number;
  className?: string;
  text?: string;
  variant?: 'default' | 'gradient';
}

export const Loader = ({ size = 32, className, text, variant = 'default' }: LoaderProps) => {
  if (variant === 'gradient') {
    return (
      <div className={cn('flex flex-col items-center justify-center gap-4', className)}>
        <div className="relative">
          <div className="w-16 h-16 rounded-full bg-gradient-to-r from-gradient-start to-gradient-end animate-spin">
            <div className="absolute inset-2 bg-background rounded-full"></div>
          </div>
          <div className="absolute inset-0 bg-gradient-to-r from-gradient-start to-gradient-end rounded-full blur-lg opacity-50 animate-pulse"></div>
        </div>
        {text && (
          <div className="text-center space-y-2">
            <p className="text-text-primary font-semibold">{text}</p>
            <div className="flex items-center justify-center gap-1">
              <Sparkles className="w-3 h-3 text-accent animate-pulse" />
              <span className="text-text-secondary text-sm font-mono">Processing...</span>
              <Sparkles className="w-3 h-3 text-accent animate-pulse" />
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col items-center justify-center gap-3', className)}>
      <Loader2 className="animate-spin text-accent" size={size} />
      {text && (
        <p className="text-text-secondary text-sm font-medium">{text}</p>
      )}
    </div>
  );
};