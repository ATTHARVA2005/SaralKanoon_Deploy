// src/lib/api.ts
import axios from 'axios';
import type { AnalysisResult } from '../types';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    const errorMessage = error.response?.data?.error || error.message || 'An unexpected error occurred';
    console.error('API Error:', errorMessage);
    return Promise.reject(new Error(errorMessage));
  }
);

export const analyzeDocument = async (file: File): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('document', file);

  const response = await apiClient.post('/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data as AnalysisResult;
};

export const askQuestion = async (question: string): Promise<{ answer: string }> => {
  const response = await apiClient.post<{ answer: string }>('/ask', { question });
  return response.data;
};

// Translate a section of the document
export const translateSection = async (section: string, text: string, targetLang: string): Promise<string> => {
  const response = await apiClient.post<{ translated: string }>('/translate', {
    section,
    text,
    target_lang: targetLang,
  });
  return response.data.translated;
};

// Get audio for a section of text
export const getAudio = async (text: string, lang: string): Promise<Blob> => {
  const response = await apiClient.post<Blob>('/audio', {
    text,
    lang,
  }, { responseType: 'blob' });
  return response.data;
};