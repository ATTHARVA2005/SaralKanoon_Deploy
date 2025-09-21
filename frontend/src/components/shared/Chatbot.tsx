// src/components/shared/Chatbot.tsx
import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, MessageCircle } from 'lucide-react';
import { Button } from '../ui/Button';
import { askQuestion } from '../../lib/api';
import type { ChatMessage } from '../../types';
import { cn } from '../../lib/utils';

interface ChatbotProps {
  className?: string;
}

export const Chatbot = ({ className }: ChatbotProps) => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestedQuestions = [
    "What are the key payment terms?",
    "Are there any termination clauses?",
    "What are my obligations under this contract?",
    "Are there any penalties mentioned?",
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSendMessage = async (questionText = inputValue) => {
    if (!questionText.trim()) return;
    setIsLoading(true);
    const userMessage = questionText;
    setMessages((prev) => [...prev, { user: userMessage, ai: '', timestamp: Date.now() }]);
    if (inputValue) setInputValue('');

    try {
      const aiResponse = await askQuestion(userMessage);
      setMessages((prev) => {
        const updated = [...prev];
        const lastMessage = updated[updated.length - 1];
        if (lastMessage) lastMessage.ai = aiResponse.answer;
        return updated;
      });
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        const lastMessage = updated[updated.length - 1];
        if (lastMessage) lastMessage.ai = 'Sorry, I encountered an error. Please try again.';
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {/* Floating button */}
      {!open && (
        <button
          className="fixed bottom-6 right-6 z-50 w-16 h-16 rounded-full bg-gradient-to-r from-gradient-start to-gradient-end flex items-center justify-center shadow-2xl hover:scale-105 transition-all animate-fade-in"
          onClick={() => setOpen(true)}
          aria-label="Open Chatbot"
        >
          <MessageCircle className="w-8 h-8 text-white" />
        </button>
      )}
      {/* Chatbot panel */}
      {open && (
        <div
          className={cn(
            'glass-effect rounded-2xl overflow-hidden glow-effect fixed bottom-6 right-6 z-50 max-w-md w-full md:w-[400px] shadow-2xl flex flex-col', // <-- ADDED flex flex-col
            'animate-fade-in-up',
            className
          )}
          style={{ height: '80vh', maxHeight: '700px' }}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-gradient-start/10 to-gradient-end/10 p-4 border-b border-border/50 flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-r from-gradient-start to-gradient-end rounded-xl flex items-center justify-center">
                  <Bot className="w-7 h-7 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-bold font-display text-text-primary flex items-center gap-2">
                  AI Legal Assistant
                </h3>
                <p className="text-text-secondary text-sm">Ask about your document</p>
              </div>
            </div>
            <button
              className="p-2 rounded-full text-text-secondary hover:bg-muted/50 hover:text-destructive"
              onClick={() => setOpen(false)}
              aria-label="Close Chatbot"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
          {/* Messages Area */}
          <div className="p-6 space-y-6 flex-grow overflow-y-auto"> {/* <-- CHANGED: flex-grow and overflow-y-auto */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                <div className="w-16 h-16 bg-gradient-to-r from-gradient-start/20 to-gradient-end/20 rounded-2xl flex items-center justify-center">
                  <Bot className="w-10 h-10 text-accent" />
                </div>
                <div>
                  <h4 className="text-text-primary font-bold text-lg font-display">Ready to Help</h4>
                  <p className="text-text-secondary text-sm">I've analyzed your document. Try one of the suggestions below.</p>
                </div>
                <div className="w-full pt-2">
                  {suggestedQuestions.map((q) => (
                    <button key={q} onClick={() => handleSendMessage(q)} className="w-full text-left p-3 bg-muted/30 hover:bg-accent/10 rounded-lg text-sm text-text-secondary hover:text-accent transition-all duration-200 border border-border/30 hover:border-accent/30 mb-2">
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((message, index) => (
              <div key={index} className="space-y-4">
                <div className="flex items-start justify-end gap-2">
                  <div className="max-w-[85%] bg-gradient-to-r from-gradient-start to-gradient-end text-white rounded-2xl rounded-tr-md p-3 shadow-md">
                    <p className="text-sm leading-relaxed">{message.user}</p>
                  </div>
                   <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-text-secondary" />
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-accent" />
                  </div>
                  <div className="max-w-[85%] bg-card/50 border border-border/50 rounded-2xl rounded-tl-md p-3 shadow-md">
                    <p className="text-text-primary text-sm leading-relaxed whitespace-pre-wrap">{message.ai}</p>
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex items-start gap-2">
                <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-accent" />
                </div>
                <div className="bg-card/50 border border-border/50 rounded-2xl rounded-tl-md p-3">
                    <div className="flex items-center justify-center space-x-1">
                        <div className="w-2 h-2 bg-accent rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                        <div className="w-2 h-2 bg-accent rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                        <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>
                    </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          {/* Input Area */}
          <div className="p-4 border-t border-border/50 bg-muted/20 flex-shrink-0">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask a follow-up question..."
                className="w-full bg-card/50 border border-border/50 rounded-lg px-4 py-2 outline-none text-text-primary placeholder-text-secondary text-sm focus:border-accent/50 focus:bg-card/80 transition-all"
                disabled={isLoading}
              />
              <Button onClick={() => handleSendMessage()} disabled={!inputValue.trim() || isLoading} className="p-2 h-10 w-10 rounded-lg">
                <Send className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
