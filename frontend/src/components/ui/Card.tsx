// src/components/ui/Card.tsx
import React from 'react';
import { cn } from '../../lib/utils';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title: string;
}

export const Card = ({ children, className, title }: CardProps) => {
  return (
    <div className={cn(
      'bg-foreground border border-border rounded-xl shadow-sm p-6 transition-all duration-200 hover:shadow-md',
      className
    )}>
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        {title}
      </h3>
      <div className="text-text-secondary space-y-4">
        {children}
      </div>
    </div>
  );
};