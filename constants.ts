
// Tailwind CSS color classes for different syntactic labels
// Format: 'text-{color}-{shade} border-{color}-{shade-100 or same shade}'
// Ensure good contrast with background (bg-slate-800/700) and text readability.

export const LABEL_COLORS: { [key: string]: string } = {
  // Sintagmas Principales y Sujeto/Predicado
  'SN Sujeto': 'text-yellow-300 border-yellow-400',
  'SV - Predicado verbal': 'text-green-400 border-green-500',
  'SV - Predicado nominal': 'text-lime-400 border-lime-500',
  'ST': 'text-red-400 border-red-500', // Sujeto Tácito (antes SO)
  
  // Tipos de Sintagmas (como constituyentes o con función específica)
  'SN': 'text-sky-300 border-sky-400', // Sintagma Nominal genérico
  'SAdj': 'text-violet-400 border-violet-500', // Sintagma Adjetival
  'SAdv': 'text-amber-400 border-amber-500', // Sintagma Adverbial
  'SPrep': 'text-rose-400 border-rose-500', // Sintagma Preposicional genérico

  // Funciones dentro de Sintagmas o a nivel oracional
  'SN - CD': 'text-sky-400 border-sky-500', // Complemento Directo
  'SN - CI': 'text-blue-400 border-blue-500', // Complemento Indirecto
  'SN - Atrib': 'text-indigo-400 border-indigo-500', // Atributo (si es SN)
  'SAdj - Atrib': 'text-indigo-400 border-indigo-500', // Atributo (si es SAdj)
  'SN - Voc': 'text-purple-400 border-purple-500', // Vocativo
  'SN - Término': 'text-orange-500 border-orange-600', // Término de SPrep (si es SN)
  'SAdj - Término': 'text-orange-500 border-orange-600',
  'SAdv - Término': 'text-orange-500 border-orange-600',
  'SN - CPred': 'text-pink-400 border-pink-500',      // Complemento Predicativo (SN)
  'SAdj - CPred': 'text-pink-400 border-pink-500',    // Complemento Predicativo (SAdj)

  // Sintagmas Preposicionales con función
  'SPrep - CRég': 'text-rose-500 border-rose-600',    // C. de Régimen
  'SPrep - CC': 'text-yellow-500 border-yellow-600',  // C. Circunstancial (genérico)
  'SPrep - CC de Lugar': 'text-yellow-500 border-yellow-600',
  'SPrep - CC de Tiempo': 'text-yellow-500 border-yellow-600',
  'SPrep - CC de Modo': 'text-yellow-500 border-yellow-600',
  'SPrep - CC de Causa': 'text-yellow-500 border-yellow-600',
  'SPrep - CC de Finalidad': 'text-yellow-500 border-yellow-600',
  'SPrep - CC de Compañía': 'text-yellow-500 border-yellow-600',
  'SPrep - CC de Instrumento': 'text-yellow-500 border-yellow-600',
  'SPrep - CI': 'text-blue-500 border-blue-600',      // C. Indirecto (si es SPrep)
  'SPrep - CAg': 'text-teal-400 border-teal-500',      // C. Agente
  'SPrep - CN': 'text-cyan-400 border-cyan-500',       // C. del Nombre
  'SPrep - CAdj': 'text-fuchsia-400 border-fuchsia-500', // C. del Adjetivo
  'SPrep - CAdv': 'text-purple-500 border-purple-600', // C. del Adverbio
  'SPrep - Atrib': 'text-indigo-500 border-indigo-600', // Atributo (si es SPrep)
  'SAdv - CC de Negación': 'text-amber-400 border-amber-500', // Para "no" como CC

  // Proposiciones Coordinadas
  'Prop': 'text-teal-300 border-teal-400', // Proposición genérica (usada para coordinadas)
  'Prop - Coordinada': 'text-teal-400 border-teal-500',
  'Prop - Coordinada Copulativa': 'text-teal-400 border-teal-500',
  'Prop - Coordinada Disyuntiva': 'text-teal-400 border-teal-500',
  'Prop - Coordinada Adversativa': 'text-teal-400 border-teal-500',
  'Prop - Coordinada Distributiva': 'text-teal-400 border-teal-500',
  'Prop - Coordinada Explicativa': 'text-teal-400 border-teal-500',
  'Prop - Yuxtapuesta': 'text-teal-400 border-teal-500',
  
  // Oraciones Subordinadas Sustantivas
  'Oración - Subordinada Sustantiva': 'text-cyan-400 border-cyan-500',
  'Oración - Subordinada Sustantiva de Sujeto': 'text-cyan-400 border-cyan-500',
  'Oración - Subordinada Sustantiva de CD': 'text-cyan-400 border-cyan-500',
  'Oración - Subordinada Sustantiva de Término': 'text-cyan-400 border-cyan-500', // para CI, CRég, CN, CAdj, CAdv
  'Oración - Subordinada Sustantiva de Atributo': 'text-cyan-400 border-cyan-500',

  // Oraciones Subordinadas Relativas (antes Adjetivas)
  'Oración - Subordinada Relativa': 'text-sky-400 border-sky-500',
  'Oración - Subordinada Relativa Especificativa': 'text-sky-400 border-sky-500',
  'Oración - Subordinada Relativa Explicativa': 'text-sky-400 border-sky-500',
  'Oración - Subordinada Relativa Libre': 'text-sky-400 border-sky-500', 
  'Oración - Subordinada Relativa Semilibre': 'text-sky-400 border-sky-500',

  // Oraciones Subordinadas Construcciones (antes Adverbiales)
  'Oración - Subordinada Construcción': 'text-lime-500 border-lime-600', // Main color for all construcciones
  'Oración - Subordinada Construcción de Tiempo': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción de Lugar': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción de Modo': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción Condicional': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción Concesiva': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción Causal': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción Final': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción Consecutiva': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción Comparativa': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Construcción Ilativa': 'text-lime-500 border-lime-600',
  'Oración - Subordinada Superlativa': 'text-lime-500 border-lime-600', // Grouped with construccion-like for color
  
  // Núcleos y Partículas
  'N': 'text-pink-300 border-pink-400',             // Nombre (si no es núcleo explícito N (N))
  'N (N)': 'text-pink-300 border-pink-400',         // Nombre (Núcleo de SN)
  'V (N)': 'text-emerald-300 border-emerald-400',   // Verbo (Núcleo de SV)
  'Adj': 'text-violet-300 border-violet-400',       // Adjetivo (si no es núcleo explícito Adj (N))
  'Adj (N)': 'text-violet-300 border-violet-400',   // Adjetivo (Núcleo de SAdj)
  'Adv': 'text-amber-300 border-amber-400',         // Adverbio (si no es núcleo explícito Adv (N))
  'Adv (N)': 'text-amber-300 border-amber-400',     // Adverbio (Núcleo de SAdv)
  'Prep': 'text-orange-400 border-orange-500',       // Preposición (si no es núcleo explícito Prep (N))
  'Prep (N)': 'text-orange-400 border-orange-500',   // Preposición (Núcleo de SPrep)
  'Pron': 'text-sky-200 border-sky-300',             // Pronombre
  'Det': 'text-slate-300 border-slate-400',           // Determinante
  'nx': 'text-stone-400 border-stone-500',            // Nexo (conjunción pura)
  'PronRel': 'text-stone-400 border-stone-500',       // Pronombre Relativo (nexo + función)
  'AdvRel': 'text-stone-400 border-stone-500',        // Adverbio Relativo (nexo + función)
  'DetRel': 'text-stone-400 border-stone-500',        // Determinante Relativo (nexo + función)
  'Interj': 'text-red-400 border-red-500',            // Interjección
  
  // Fallback
  'Enl': 'text-yellow-400 border-yellow-500', // Enlace (genérico, menos usado con NGLE SPrep)
  'Térm': 'text-orange-400 border-orange-500', // Término (genérico, menos usado con NGLE SPrep)
  'DEFAULT': 'text-slate-300 border-slate-500',
};

// Helper function to get color class based on label
export const getColorClass = (label: string): string => {
  const normalizedLabel = label.trim();
  
  if (LABEL_COLORS[normalizedLabel]) {
    return LABEL_COLORS[normalizedLabel];
  }

  // Fallback for "Prop - Coordinada..."
  if (normalizedLabel.startsWith('Prop - Coordinada')) {
    if (LABEL_COLORS['Prop - Coordinada']) return LABEL_COLORS['Prop - Coordinada'];
    if (LABEL_COLORS['Prop']) return LABEL_COLORS['Prop'];
  }
  
  // Fallback for "Oración - Subordinada Sustantiva..."
  if (normalizedLabel.startsWith('Oración - Subordinada Sustantiva')) {
    if (LABEL_COLORS['Oración - Subordinada Sustantiva']) return LABEL_COLORS['Oración - Subordinada Sustantiva'];
  }
  // Fallback for "Oración - Subordinada Relativa..."
  if (normalizedLabel.startsWith('Oración - Subordinada Relativa')) {
    if (LABEL_COLORS['Oración - Subordinada Relativa']) return LABEL_COLORS['Oración - Subordinada Relativa'];
  }
  // Enhanced fallback for all construction types ("Oración - Subordinada Construcción..." or "Oración - Subordinada Superlativa")
  if (normalizedLabel.startsWith('Oración - Subordinada Construcción') || normalizedLabel === 'Oración - Subordinada Superlativa') {
    if (LABEL_COLORS['Oración - Subordinada Construcción']) return LABEL_COLORS['Oración - Subordinada Construcción'];
  }
  // General fallback for other "Oración - Subordinada..."
  if (normalizedLabel.startsWith('Oración - Subordinada')) {
    // Attempt a more generic fallback if a very specific "Oración - Subordinada Xyz" wasn't found
    const generalSubordinateType = normalizedLabel.split(' de ')[0].split(' ').slice(0, 3).join(' '); // e.g. "Oración - Subordinada Sustantiva"
    if (LABEL_COLORS[generalSubordinateType]) return LABEL_COLORS[generalSubordinateType];
  }

  // Fallback for "SPrep - CC de Specific" to "SPrep - CC" or "SPrep"
  if (normalizedLabel.startsWith('SPrep - CC')) {
    if (LABEL_COLORS['SPrep - CC']) return LABEL_COLORS['SPrep - CC'];
    if (LABEL_COLORS['SPrep']) return LABEL_COLORS['SPrep'];
  }
  if (normalizedLabel.startsWith('SPrep -')) {
    if (LABEL_COLORS['SPrep']) return LABEL_COLORS['SPrep'];
  }
  
  // Fallback for "SN - Specific Function" to "SN"
  if (normalizedLabel.startsWith('SN -')) {
    if (LABEL_COLORS['SN']) return LABEL_COLORS['SN'];
  }
  // Fallback for "SAdj - Specific Function" to "SAdj"
  if (normalizedLabel.startsWith('SAdj -')) {
    if (LABEL_COLORS['SAdj']) return LABEL_COLORS['SAdj'];
  }
  // Fallback for "SAdv - Specific Function" to "SAdv"
  if (normalizedLabel.startsWith('SAdv -')) {
    if (LABEL_COLORS['SAdv - CC de Negación'] && normalizedLabel === 'SAdv - CC de Negación') return LABEL_COLORS['SAdv - CC de Negación'];
    if (LABEL_COLORS['SAdv']) return LABEL_COLORS['SAdv'];
  }
  // Fallback for PronRel/AdvRel etc. with function
  if (normalizedLabel.includes('(') && normalizedLabel.includes(')')) {
    const baseLabel = normalizedLabel.substring(0, normalizedLabel.indexOf(' (')).trim();
    if (LABEL_COLORS[baseLabel]) return LABEL_COLORS[baseLabel];
  }

  return LABEL_COLORS['DEFAULT'];
};
