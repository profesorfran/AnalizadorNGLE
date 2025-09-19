export interface SyntacticElement {
  text: string;
  label: string;
  children?: SyntacticElement[];
}

export interface SentenceAnalysis {
  fullSentence: string;
  classification: string;
  structure: SyntacticElement[];
}
