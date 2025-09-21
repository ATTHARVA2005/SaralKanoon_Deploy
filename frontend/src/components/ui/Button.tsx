// src/components/ui/Button.tsx
import React from 'react';
import { cn } from '../../lib/utils';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
}

export const Button = ({ 
  children, 
  className, 
  variant = 'primary', 
  size = 'md',
  disabled,
  ...props 
}: ButtonProps) => {
  const baseStyles = 'inline-flex items-center justify-center rounded-xl font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-accent/50 focus:ring-offset-2 focus:ring-offset-background active:scale-95';
  
  const variants = {
    primary: 'bg-gradient-to-r from-gradient-start to-gradient-end text-white hover:from-gradient-start/90 hover:to-gradient-end/90 shadow-lg hover:shadow-xl disabled:from-gray-400 disabled:to-gray-500',
    secondary: 'glass-effect text-text-primary border border-border/50 hover:border-accent/50 hover:bg-accent/5 disabled:opacity-50',
    ghost: 'text-text-primary hover:bg-muted/50 disabled:text-text-secondary',
    destructive: 'bg-gradient-to-r from-destructive to-red-500 text-white hover:from-destructive/90 hover:to-red-500/90 shadow-lg hover:shadow-xl',
  };

  const sizes = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-3 text-sm',
    lg: 'px-6 py-4 text-base',
  };

  return (
    <button
      className={cn(
        baseStyles,
        variants[variant],
        sizes[size],
        disabled && 'cursor-not-allowed opacity-60',
        className
      )}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
};