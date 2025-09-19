
import React from 'react';
import type { SyntacticElement } from '../types';
import { getColorClass } from '../constants';

interface TreeNodeProps {
  element: SyntacticElement;
  level: number;
  density: 'normal' | 'compact' | 'super-compact';
}

const getDensityConfig = (density: TreeNodeProps['density']) => {
  // Baseline configuration - corresponds to "normal" density
  let config = {
    // Overall node container
    nodePadding: 'p-1', 
    nodeMargin: 'm-0.5', 
    minNodeWidth: 'min-w-[75px]', 
    // Text element (<p>) styling
    textFontSize: 'text-sm', 
    textLineHeight: 'leading-normal',
    textMarginBottom: 'mb-1', 
    textMaxWidth: 'max-w-[100px]', 
    // Wrapper around text and label
    textLabelWrapperMarginBottom: 'mb-1', // Adjusted from marginTop
    // Label element (<p>) styling
    labelFontSize: 'text-[11px]', 
    labelLineHeight: 'leading-tight',
    // Line at the very bottom for Sintagmas (moved from top)
    sintagmaBottomLineMarginTop: 'mt-1', // Adjusted from topLineMarginBottom
    // Children area
    childrenContainerPaddingBottom: 'pb-1.5', // Adjusted from paddingTop
    childrenContainerGap: 'gap-1', 
  };

  if (density === 'compact') {
    config = {
      nodePadding: 'px-1 py-0.5', 
      nodeMargin: 'm-0.5',
      minNodeWidth: 'min-w-[60px]',
      textFontSize: 'text-xs', 
      textLineHeight: 'leading-snug',
      textMarginBottom: 'mb-0.5',
      textMaxWidth: 'max-w-[90px]',
      textLabelWrapperMarginBottom: 'mb-0.5', // Adjusted
      labelFontSize: 'text-[10px]', 
      labelLineHeight: 'leading-snug',
      sintagmaBottomLineMarginTop: 'mt-0.5', // Adjusted
      childrenContainerPaddingBottom: 'pb-1', // Adjusted
      childrenContainerGap: 'gap-0.5', 
    };
  } else if (density === 'super-compact') {
    config = {
      nodePadding: 'p-px', 
      nodeMargin: 'm-px', 
      minNodeWidth: 'min-w-[30px]', 
      textFontSize: 'text-[7px]', 
      textLineHeight: 'leading-none', 
      textMarginBottom: 'mb-px',
      textMaxWidth: 'max-w-[50px]', 
      textLabelWrapperMarginBottom: 'mb-px', // Adjusted
      labelFontSize: 'text-[6px]', 
      labelLineHeight: 'leading-none', 
      sintagmaBottomLineMarginTop: 'mt-px', // Adjusted
      childrenContainerPaddingBottom: 'pb-px', // Adjusted
      childrenContainerGap: 'gap-px', 
    };
  }
  return config;
};


export const TreeNode: React.FC<TreeNodeProps> = ({ element, level, density }) => {
  const colorClasses = getColorClass(element.label);
  const [textColorClass, borderColorString] = 
    typeof colorClasses === 'string' && colorClasses.includes(' ') 
    ? colorClasses.split(' ') 
    : ['text-slate-300', 'border-slate-500'];
  
  const hasChildren = element.children && element.children.length > 0;
  const isSintagma = element.label.toUpperCase().startsWith('S'); 
  const bgLineColorClass = borderColorString ? borderColorString.replace(/^border-/, 'bg-') : 'bg-slate-500';

  const config = getDensityConfig(density);

  const showElementTextForThisNode = !isSintagma || (isSintagma && !hasChildren);

  return (
    <div 
      className={`flex flex-col-reverse items-center ${config.nodePadding} ${config.nodeMargin} ${config.minNodeWidth} relative ${hasChildren ? 'bg-slate-700/40 rounded-lg shadow-md' : ''}`}
      style={{flexBasis: 'auto', flexShrink: 0}}
    >
      {/* Scope line for Sintagmas - MOVED TO BE FIRST (visually at bottom due to flex-col-reverse) */}
      {isSintagma && (
        <div className={`w-full h-[1px] ${bgLineColorClass} ${config.sintagmaBottomLineMarginTop}`}></div>
      )}

      {/* Element's own text and label - Rendered visually above the bottom line */}
      <div className={`w-full flex flex-col items-center ${config.textLabelWrapperMarginBottom}`}> 
        {showElementTextForThisNode && (
          <p className={`font-medium text-center text-slate-100 break-words ${config.textFontSize} ${config.textLineHeight} ${config.textMarginBottom} ${config.textMaxWidth}`}>
            {element.text}
          </p>
        )}
        <div className="flex flex-col items-center w-full">
          {/* Small line above label REMOVED */}
          <p className={`font-semibold ${textColorClass} text-center uppercase tracking-wider ${config.labelFontSize} ${config.labelLineHeight}`}>{element.label}</p>
        </div>
      </div>

      {/* Children nodes - Rendered LAST in JSX (visually at top due to flex-col-reverse) */}
      {hasChildren && (
        <div 
          className={`flex flex-row flex-wrap justify-center items-start border-b ${borderColorString} border-opacity-30 w-full ${config.childrenContainerPaddingBottom} ${config.childrenContainerGap}`} /* Changed to border-b and padding-bottom */
        >
          {element.children?.map((child, index) => (
            <TreeNode key={`${level}-${index}-${child.label}-${child.text.slice(0,5)}`} element={child} level={level + 1} density={density} />
          ))}
        </div>
      )}
    </div>
  );
};
