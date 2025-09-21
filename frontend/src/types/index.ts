// src/types/index.ts
export interface Clause {
  title: string;
  detail: string;
}

export interface AnalysisResult {
  summary: string;
  keyClauses: Clause[];
  redFlags: Clause[];
}

export interface ApiError {
  error: string;
}

export interface ChatMessage {
  user: string;
  ai: string;
  timestamp: number;
}

export type Theme = 'light' | 'dark';