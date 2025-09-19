
import React, { useState } from 'react';

interface SentenceInputProps {
  onAnalyze: (sentence: string) => void;
  isLoading: boolean;
  disabled: boolean;
}

export const SentenceInput: React.FC<SentenceInputProps> = ({ onAnalyze, isLoading, disabled }) => {
  const [sentence, setSentence] = useState<string>('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!disabled) {
      onAnalyze(sentence);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mb-6 md:mb-8">
      <div className="flex flex-col sm:flex-row gap-3 items-stretch">
        <textarea
          value={sentence}
          onChange={(e) => setSentence(e.target.value)}
          placeholder="Escribe aquí la oración que deseas analizar..."
          rows={3}
          className="flex-grow p-3 border border-slate-600 rounded-lg shadow-sm focus:ring-2 focus:ring-sky-500 focus:border-sky-500 bg-slate-700 text-slate-100 placeholder-slate-400 disabled:opacity-50 disabled:cursor-not-allowed resize-none transition-opacity"
          disabled={isLoading || disabled}
          aria-label="Campo de entrada para la oración"
        />
        <button
          type="submit"
          disabled={isLoading || disabled || !sentence.trim()}
          className="w-full sm:w-auto px-6 py-3 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50 transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          style={{minHeight: 'calc(3 * 1.5rem + 2 * 0.75rem + 2px)'}} // Match textarea height approx.
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analizando...
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-1 hidden sm:inline">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L1.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.25 12L17.437 9.154a4.5 4.5 0 00-3.09-3.09L11.5 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L18.25 12zm0 0l.813 2.846a4.5 4.5 0 003.09 3.09L24.75 12l-2.846-.813a4.5 4.5 0 00-3.09-3.09L18.25 12z" />
              </svg>
              <span>Analizar</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};
