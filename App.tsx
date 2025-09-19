
import React, { useState, useCallback } from 'react';
import { SentenceInput } from './components/SentenceInput';
import { AnalysisDisplay } from './components/AnalysisDisplay';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import { analyzeSentence, isApiKeyConfigured } from './services/geminiService';
import type { SentenceAnalysis } from './types';

const apiKeyAvailable = isApiKeyConfigured();

const App: React.FC = () => {
  const [sentence, setSentence] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<SentenceAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = useCallback(async (text: string) => {
    setSentence(text);
    if (!text.trim()) {
      setError("Por favor, ingresa una oración para analizar.");
      setAnalysisResult(null);
      return;
    }
    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);
    try {
      const result = await analyzeSentence(text);
      if (result) {
        setAnalysisResult(result);
      } else {
        setError("No se pudo obtener un análisis válido. La respuesta del modelo podría estar vacía o malformada. Revisa la consola para más detalles.");
      }
    } catch (e: any) {
      console.error("Analysis error:", e);
      setError(e.message || "Ocurrió un error al analizar la oración.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <div className="container mx-auto p-4 md:p-8 max-w-screen-xl w-full">
      <header className="relative text-center mb-8 md:mb-12">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-sky-400 tracking-tight">
          Analizador Sintáctico (NGLE)
        </h1>
        <p className="text-slate-300 mt-3 text-md sm:text-lg">
          Aplicación desarrollada por Francisco David Sánchez Valencia, profesor de Lengua y Literatura.
        </p>
        <p className="text-slate-400 mt-2 text-sm sm:text-base">
          Introduce una oración en español y obtén su análisis sintáctico detallado según la NGLE.
        </p>
      </header>

      {!apiKeyAvailable && (
        <div className="mb-6">
            <ErrorMessage message="CONFIGURACIÓN REQUERIDA: La clave API de Gemini no está configurada en esta aplicación. El análisis no funcionará." />
        </div>
      )}

      <main className="bg-slate-800 shadow-2xl rounded-xl p-4 sm:p-6 md:p-8">
        <SentenceInput onAnalyze={handleAnalyze} isLoading={isLoading} disabled={!apiKeyAvailable} />

        {isLoading && <LoadingSpinner />}
        {error && !isLoading && <ErrorMessage message={error} />}
        
        {analysisResult && !isLoading && !error && (
          <AnalysisDisplay result={analysisResult} />
        )}
        
        {!isLoading && !error && !analysisResult && apiKeyAvailable && (
          <div className="mt-8 p-6 bg-slate-700/50 rounded-lg text-center text-slate-400">
            <p className="text-lg">Esperando una oración para analizar...</p>
            <p className="text-sm mt-2">Ejemplo: El libro que me prestaste ayer es muy interesante y lo leeré pronto.</p>
          </div>
        )}
      </main>
      <footer className="text-center mt-10 md:mt-16 text-slate-500 text-xs sm:text-sm">
        <p>Potenciado por Gemini API. Análisis basado en la Nueva Gramática de la Lengua Española.</p>
        <p>&copy; {new Date().getFullYear()} Francisco David Sánchez Valencia</p>
      </footer>
    </div>
  );
};

export default App;
