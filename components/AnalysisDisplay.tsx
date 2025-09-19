
import React from 'react';
import type { SentenceAnalysis, SyntacticElement } from '../types';
import { getColorClass } from '../constants';
import { TreeNode } from './TreeNode'; 

interface AnalysisDisplayProps {
  result: SentenceAnalysis;
}

const countNodes = (elements: SyntacticElement[]): number => {
  let count = 0;
  for (const el of elements) {
    count++;
    if (el.children) {
      count += countNodes(el.children);
    }
  }
  return count;
};

export const AnalysisDisplay: React.FC<AnalysisDisplayProps> = ({ result }) => {
  const totalNodes = countNodes(result.structure);
  let density: 'normal' | 'compact' | 'super-compact' = 'normal';
  let rootNodeGapClass = 'gap-1'; 

  if (totalNodes >= 30) {
    density = 'super-compact';
    rootNodeGapClass = 'gap-px'; 
  } else if (totalNodes >= 15) { 
    density = 'compact';
    rootNodeGapClass = 'gap-0.5'; 
  }

  return (
    <div className="mt-6 p-4 md:p-6 bg-slate-700/50 rounded-lg shadow-inner">
      <div 
        className="relative mb-6 pb-4 border-b border-slate-600 rounded-md p-3 -m-3"
      >
        <div className="flex items-center mb-1">
          <h2 className="text-lg sm:text-xl font-semibold text-sky-300">Oración Analizada:</h2>
        </div>
        <p className="text-md sm:text-lg text-slate-200">{result.fullSentence}</p>
      </div>

      <div className="mb-6 pb-4 border-b border-slate-600">
        <h2 className="text-lg sm:text-xl font-semibold text-sky-300 mb-1">Clasificación:</h2>
        <p className="text-sm sm:text-md text-slate-300">{result.classification}</p>
      </div>
      
      <div className="mb-4">
        <h2 className="text-lg sm:text-xl font-semibold text-sky-300 mb-3">Estructura Sintáctica:</h2>
        <div className="overflow-x-auto pb-4 -mx-1 px-1">
            <div className={`flex flex-row flex-nowrap justify-start items-start p-1 min-w-max ${rootNodeGapClass}`}>
            {result.structure.map((element, index) => (
                <TreeNode key={`root-${index}-${element.label}-${element.text.slice(0,5)}`} element={element} level={0} density={density} />
            ))}
            </div>
        </div>
      </div>
    </div>
  );
};