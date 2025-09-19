
import React, { useState } from 'react';

interface ApiKeyModalProps {
  onSave: (apiKey: string) => void;
  onClose: () => void;
  currentApiKey: string | null;
}

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ onSave, onClose, currentApiKey }) => {
  const [apiKey, setApiKey] = useState(currentApiKey || '');

  const handleSave = () => {
    onSave(apiKey);
  };

  // Prevent modal from closing when clicking inside the panel
  const handlePanelClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };
  
  return (
    <div 
        className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 backdrop-blur-sm"
        onClick={onClose}
        role="dialog"
        aria-modal="true"
        aria-labelledby="apiKeyModalTitle"
    >
      <div 
        className="bg-slate-800 rounded-xl shadow-2xl p-6 md:p-8 m-4 w-full max-w-lg border border-slate-700"
        onClick={handlePanelClick}
      >
        <div className="flex justify-between items-center mb-4">
          <h2 id="apiKeyModalTitle" className="text-xl font-bold text-sky-400">
            Configurar Clave API de Gemini
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white" aria-label="Cerrar modal">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <p className="text-slate-300 mb-4 text-sm">
          Para utilizar el analizador, necesitas una clave API de Google Gemini. Es gratuita para un uso moderado y puedes obtenerla en Google AI Studio.
        </p>

        <a 
          href="https://aistudio.google.com/app/apikey" 
          target="_blank" 
          rel="noopener noreferrer" 
          className="inline-block text-sky-400 hover:text-sky-300 mb-6 text-sm font-medium"
        >
          Obtener una clave API aquí &rarr;
        </a>

        <div>
          <label htmlFor="apiKeyInput" className="block text-sm font-medium text-slate-200 mb-2">
            Tu Clave API
          </label>
          <input
            id="apiKeyInput"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Pega tu clave API aquí..."
            className="w-full p-2 border border-slate-600 rounded-md shadow-sm focus:ring-2 focus:ring-sky-500 focus:border-sky-500 bg-slate-700 text-slate-100 placeholder-slate-400"
          />
        </div>

        <div className="mt-8 flex justify-end gap-4">
          <button 
            onClick={onClose} 
            className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white font-semibold rounded-md shadow-md focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-opacity-50 transition duration-150"
          >
            Cancelar
          </button>
          <button 
            onClick={handleSave}
            disabled={!apiKey.trim()}
            className="px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-md shadow-md focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Guardar Clave
          </button>
        </div>
      </div>
    </div>
  );
};
