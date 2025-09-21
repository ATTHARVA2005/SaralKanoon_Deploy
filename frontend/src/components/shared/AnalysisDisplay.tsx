// src/components/shared/AnalysisDisplay.tsx
import { Flag, FileText, AlertTriangle, CheckCircle, Shield, Eye, TrendingUp, Play, Pause } from 'lucide-react';
import type { AnalysisResult } from '../../types';
import { cn } from '../../lib/utils';
import { useState, useRef, useEffect } from 'react';
import { translateSection, getAudio } from '../../lib/api';

interface AnalysisDisplayProps {
  analysisResult: AnalysisResult;
}

export const AnalysisDisplay = ({ analysisResult }: AnalysisDisplayProps) => {
  const { summary, keyClauses, redFlags } = analysisResult;

  // --- State for Audio ---
  const [audioUrls, setAudioUrls] = useState<{ [key: string]: string | null }>({});
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(new Audio());

  // --- State for Translation ---
  const [translatedSummary, setTranslatedSummary] = useState<string | null>(null);
  const [summaryLang, setSummaryLang] = useState('en');
  const [translatedClauses, setTranslatedClauses] = useState<(string | null)[]>(keyClauses.map(() => null));
  const [clausesLangs, setClausesLangs] = useState<string[]>(keyClauses.map(() => 'en'));
  const [translatedFlags, setTranslatedFlags] = useState<(string | null)[]>(redFlags.map(() => null));
  const [flagsLangs, setFlagsLangs] = useState<string[]>(redFlags.map(() => 'en'));
  const [translatingKey, setTranslatingKey] = useState<string | null>(null); // NEW: State for translation loading

  // --- Proactive Audio Fetching ---
  useEffect(() => {
    const prefetchAllAudio = async () => {
      console.log("Pre-fetching all audio files...");
      const urlsToFetch: { key: string; text: string; lang: string }[] = [];
      urlsToFetch.push({ key: 'summary', text: summary, lang: 'en' });
      keyClauses.forEach((c, i) => urlsToFetch.push({ key: `clause-${i}`, text: `${c.title}. ${c.detail}`, lang: 'en' }));
      redFlags.forEach((f, i) => urlsToFetch.push({ key: `flag-${i}`, text: `${f.title}. ${f.detail}`, lang: 'en' }));

      const audioPromises = urlsToFetch.map(item => getAudio(item.text, item.lang).then(blob => ({
        key: item.key,
        url: URL.createObjectURL(blob)
      })));

      try {
        const fetchedAudios = await Promise.all(audioPromises);
        const newAudioUrls = fetchedAudios.reduce((acc, current) => {
          acc[current.key] = current.url;
          return acc;
        }, {} as { [key: string]: string });
        setAudioUrls(newAudioUrls);
        console.log("All audio files pre-fetched successfully.");
      } catch (error) {
        console.error("Failed to pre-fetch audio files:", error);
      }
    };

    if (analysisResult && Object.keys(audioUrls).length === 0) {
      prefetchAllAudio();
    }

    return () => {
      Object.values(audioUrls).forEach(url => { if (url) URL.revokeObjectURL(url); });
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
      }
    };
  }, [analysisResult]);

  // --- Handlers ---
  const handleAudioToggle = (key: string) => {
    const url = audioUrls[key];
    if (!url) return;
    const audio = audioRef.current;
    if (playingKey === key) {
      audio.pause();
      setPlayingKey(null);
    } else {
      if (!audio.paused) audio.pause();
      audio.src = url;
      audio.currentTime = 0;
      audio.play();
      setPlayingKey(key);
    }
    audio.onended = () => setPlayingKey(null);
  };

  const handleTranslate = async (
    type: 'summary' | 'clause' | 'flag',
    index: number = 0
  ) => {
    let textToTranslate = '';
    let currentLang = 'en';
    let key = '';

    if (type === 'summary') {
      textToTranslate = summary;
      currentLang = summaryLang;
      key = 'summary';
    } else if (type === 'clause') {
      textToTranslate = `${keyClauses[index].title}. ${keyClauses[index].detail}`;
      currentLang = clausesLangs[index];
      key = `clause-${index}`;
    } else {
      textToTranslate = `${redFlags[index].title}. ${redFlags[index].detail}`;
      currentLang = flagsLangs[index];
      key = `flag-${index}`;
    }

    const targetLang = currentLang === 'en' ? 'hi' : 'en';
    
    setTranslatingKey(key); // Set loading state

    try {
      const translatedText = await translateSection('section', textToTranslate, targetLang);
      const audioBlob = await getAudio(translatedText, targetLang);
      const audioUrl = URL.createObjectURL(audioBlob);

      if (type === 'summary') {
        setTranslatedSummary(targetLang === 'hi' ? translatedText : null);
        setSummaryLang(targetLang);
      } else if (type === 'clause') {
        setTranslatedClauses(p => p.map((t, i) => i === index ? (targetLang === 'hi' ? translatedText : null) : t));
        setClausesLangs(p => p.map((l, i) => i === index ? targetLang : l));
      } else {
        setTranslatedFlags(p => p.map((t, i) => i === index ? (targetLang === 'hi' ? translatedText : null) : t));
        setFlagsLangs(p => p.map((l, i) => i === index ? targetLang : l));
      }
      
      // Update the audio URL for the new language
      setAudioUrls(prev => ({...prev, [key]: audioUrl}));

    } catch (error) {
      console.error(`Failed to translate ${type}:`, error);
    } finally {
      setTranslatingKey(null); // Clear loading state
    }
  };


  return (
    <div className="space-y-16">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-effect rounded-xl p-6 text-center group hover:glow-effect transition-all duration-300">
            <div className="w-12 h-12 bg-gradient-to-r from-gradient-start to-gradient-end rounded-lg flex items-center justify-center mx-auto mb-4">
                <Eye className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold font-mono text-primary mb-1">1</h3>
            <p className="text-secondary text-sm font-medium">Document Analyzed</p>
        </div>
        <div className="glass-effect rounded-xl p-6 text-center group hover:glow-effect transition-all duration-300">
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold font-mono text-green-500 mb-1">{keyClauses.length}</h3>
            <p className="text-secondary text-sm font-medium">Key Clauses Found</p>
        </div>
        <div className="glass-effect rounded-xl p-6 text-center group hover:glow-effect transition-all duration-300">
            <div className="w-12 h-12 bg-gradient-to-r from-destructive to-red-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Shield className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold font-mono text-destructive mb-1">{redFlags.length}</h3>
            <p className="text-secondary text-sm font-medium">Red Flags Detected</p>
        </div>
      </div>

      {/* Main Analysis Grid */}
      <div className="grid gap-8 xl:grid-cols-3">
        {/* Summary */}
        <div className="xl:col-span-3">
          <div className="glass-effect rounded-2xl p-8 glow-effect">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-gradient-to-r from-gradient-start to-gradient-end rounded-xl flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold font-display text-primary">Document Summary</h2>
                <p className="text-secondary">AI-generated overview of your legal document</p>
              </div>
            </div>
            <div className="bg-muted/30 rounded-xl p-6 border border-border/50">
              <p className="text-primary text-lg leading-relaxed font-medium">
                {translatedSummary || summary}
              </p>
              <div className="flex gap-2 mt-4">
                <button
                  className="text-xs px-3 py-2 rounded bg-muted text-primary border border-border/30 hover:bg-accent/10 disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={() => handleTranslate('summary')}
                  disabled={translatingKey !== null}
                >
                  {translatingKey === 'summary' ? 'Translating...' : `Translate to ${summaryLang === 'en' ? 'Hindi' : 'English'}`}
                </button>
                <button
                  className="text-xs px-2 py-1 rounded bg-gradient-to-r from-gradient-start to-gradient-end text-white active:scale-95 transition-all duration-150 cursor-pointer disabled:opacity-50 flex items-center gap-1"
                  onClick={() => handleAudioToggle('summary')}
                  disabled={!audioUrls['summary']}
                >
                  {playingKey === 'summary' ? <Pause size={12} /> : <Play size={12} />}
                  {playingKey === 'summary' ? 'Pause' : 'Play'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Key Clauses */}
        <div className="xl:col-span-2">
          <div className="glass-effect rounded-2xl p-8 h-full">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold font-display text-primary">Key Clauses</h2>
                <p className="text-secondary">Important terms and conditions identified</p>
              </div>
            </div>
            <div className="space-y-6">
              {keyClauses.map((clause, index) => (
                <div key={index} className="group">
                  <div className="bg-card/60 border-2 border-green-400/30 rounded-xl p-8 transition-all duration-300 hover:border-accent/50 hover:bg-card/80 shadow-md">
                    <div className="flex items-start gap-6">
                       <div className="w-10 h-10 bg-gradient-to-r from-gradient-start/30 to-gradient-end/30 rounded-lg flex items-center justify-center flex-shrink-0 mt-1 group-hover:from-gradient-start/40 group-hover:to-gradient-end/40 transition-all">
                        <span className="text-accent font-bold text-base">{index + 1}</span>
                      </div>
                      <div className="flex-1">
                          <h4 className="font-bold text-green-500 text-xl mb-3 font-display tracking-wide">{clause.title}</h4>
                          <p className="text-secondary leading-relaxed text-base mb-4">{translatedClauses[index] || clause.detail}</p>
                          <div className="flex gap-4 mt-4">
                              <button
                                  className="text-xs px-3 py-2 rounded bg-muted text-primary border border-border/30 hover:bg-accent/10 disabled:opacity-50 disabled:cursor-not-allowed"
                                  onClick={() => handleTranslate('clause', index)}
                                  disabled={translatingKey !== null}
                              >
                                {translatingKey === `clause-${index}` ? 'Translating...' : `Translate to ${clausesLangs[index] === 'en' ? 'Hindi' : 'English'}`}
                              </button>
                              <button
                                  className="text-xs px-3 py-2 rounded bg-gradient-to-r from-gradient-start to-gradient-end text-white active:scale-95 transition-all duration-150 cursor-pointer disabled:opacity-50 flex items-center gap-1"
                                  onClick={() => handleAudioToggle(`clause-${index}`)}
                                  disabled={!audioUrls[`clause-${index}`]}
                              >
                                  {playingKey === `clause-${index}` ? <Pause size={12} /> : <Play size={12} />}
                                  {playingKey === `clause-${index}` ? 'Pause' : 'Listen'}
                              </button>
                          </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Red Flags */}
        <div className="xl:col-span-1">
          <div className={cn(
            "glass-effect rounded-2xl p-8 h-full border-2 transition-all duration-300",
            redFlags.length > 0 ? "border-destructive/30 bg-destructive/5 glow-effect" : "border-green-500/30 bg-green-500/5"
          )}>
            <div className="flex items-center gap-4 mb-6">
               <div className={cn(
                "w-12 h-12 rounded-xl flex items-center justify-center",
                redFlags.length > 0
                  ? "bg-gradient-to-r from-destructive to-red-500"
                  : "bg-gradient-to-r from-green-500 to-emerald-500"
              )}>
                {redFlags.length > 0 ? (
                  <Flag className="w-6 h-6 text-white" />
                ) : (
                  <Shield className="w-6 h-6 text-white" />
                )}
              </div>
              <div>
                <h2 className="text-2xl font-bold font-display text-primary flex items-center gap-2">
                  Red Flags
                  {redFlags.length > 0 && (
                    <span className="bg-destructive text-white text-sm px-2 py-1 rounded-full font-mono">
                      {redFlags.length}
                    </span>
                  )}
                </h2>
                <p className="text-secondary">
                  {redFlags.length > 0 ? "Potential risks detected" : "No risks found"}
                </p>
              </div>
            </div>
            {redFlags.length > 0 ? (
              <div className="space-y-4">
                {redFlags.map((flag, index) => (
                  <div key={index} className="group">
                    <div className="bg-destructive/10 border-2 border-destructive/20 rounded-xl p-6 transition-all duration-300">
                      <div className="flex items-start gap-4">
                        <div className="w-8 h-8 bg-destructive/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1 group-hover:bg-destructive/30 transition-all">
                          <AlertTriangle className="w-4 h-4 text-destructive" />
                        </div>
                        <div className="flex-1">
                            <h4 className="font-bold text-red-500 text-xl mb-3 font-display tracking-wide">{flag.title}</h4>
                            <p className="text-secondary leading-relaxed text-base mb-4">{translatedFlags[index] || flag.detail}</p>
                            <div className="flex gap-4 mt-4">
                              <button
                                  className="text-xs px-3 py-2 rounded bg-muted text-primary border border-border/30 hover:bg-accent/10 disabled:opacity-50 disabled:cursor-not-allowed"
                                  onClick={() => handleTranslate('flag', index)}
                                  disabled={translatingKey !== null}
                              >
                                {translatingKey === `flag-${index}` ? 'Translating...' : `Translate to ${flagsLangs[index] === 'en' ? 'Hindi' : 'English'}`}
                              </button>
                              <button
                                  className="text-xs px-3 py-2 rounded bg-gradient-to-r from-gradient-start to-gradient-end text-white active:scale-95 transition-all duration-150 cursor-pointer disabled:opacity-50 flex items-center gap-1"
                                  onClick={() => handleAudioToggle(`flag-${index}`)}
                                  disabled={!audioUrls[`flag-${index}`]}
                              >
                                  {playingKey === `flag-${index}` ? <Pause size={12} /> : <Play size={12} />}
                                  {playingKey === `flag-${index}` ? 'Pause' : 'Listen'}
                              </button>
                            </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-green-500/10 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-8 h-8 text-green-500" />
                </div>
                <h4 className="font-bold text-green-500 text-lg mb-2 font-display">All Clear!</h4>
                <p className="text-secondary text-sm">
                  No significant red flags detected in this document. The terms appear to be standard and reasonable.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

