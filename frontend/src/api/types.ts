export interface SceneCard {
  query: string;
  text: string;
  thumb?: string;
}

export interface BootstrapData {
  cards: SceneCard[];
  memory: number;
  upload_max_bytes: number;
}

export interface SessionItem {
  session_id: string;
  title: string;
  updated_at: string;
  relative?: string;
}

export interface SessionListData {
  items: SessionItem[];
  total: number;
  offset: number;
  has_more: boolean;
}

export interface Source {
  kind?: string;
  label: string;
}

export interface ChatDelta {
  type: "delta";
  html: string;
}

export interface ChatDone {
  type: "done";
  html: string;
  session_id: string;
  memory: number;
  sources: Source[];
  cancelled: boolean;
  request_id: string;
  error: string;
}

export type ChatEvent = ChatDelta | ChatDone;

export interface HistoryMessage {
  role: string;
  content?: string;
  sources?: Source[];
  doc_id?: string;
  doc_name?: string;
  kind?: string;
  report?: ArtworkAnalysisReport;
  title?: string;
  analysis?: boolean;
}

export interface SessionDetailData {
  messages: HistoryMessage[];
}

export interface FeedbackItem {
  session_id: string;
  rating: number;
  reason?: string;
  comment?: string;
}

export interface FeedbackListData {
  items: FeedbackItem[];
}

export interface MemoryItem {
  id: string;
  kind?: string;
  content?: string;
  value?: string;
  source?: string;
  entity?: string;
  updated_at?: string;
}

export interface MemoryListData {
  items: MemoryItem[];
}

export interface DocProposedSchema {
  entity_col?: string;
  group_axis_col?: string;
  description_col?: string;
  image_col?: string;
  display_name?: string;
  reasoning?: string;
}

export interface Doc {
  doc_id: string;
  doc_name?: string;
  kind?: "table" | "pdf";
  status: string;
  rows?: number;
  cols?: number;
  columns?: string[];
  sheet_name?: string;
  proposed_schema?: DocProposedSchema;
  text_chunks?: number;
  image_pages?: number;
  pages?: number;
  supports_timeline?: boolean;
  supports_recommendation?: boolean;
  route_distribution?: unknown;
  error?: string;
}

export interface UploadResult {
  ok: boolean;
  error?: string;
  code?: string;
  doc_id?: string;
  doc_name?: string;
  kind?: "table" | "pdf";
  split?: boolean;
  count?: number;
  documents?: Array<{ doc_id: string; doc_name: string }>;
  max_bytes?: number;
}

export interface ChipEntry {
  id: string;
  doc_id: string | null;
  name: string;
  size: number | null;
  kind: "pdf" | "table" | "image";
  status:
    | "uploading"
    | "processing"
    | "pending"
    | "pending_confirm"
    | "done"
    | "active"
    | "failed";
  error: string;
  progress?: number;
  thumb_url?: string;
}

export interface UserImageUploadResult {
  ok: boolean;
  error?: string;
  image_id?: string;
  thumb_url?: string;
  width?: number;
  height?: number;
}

export type AnalysisFramework =
  | "realistic"
  | "abstract"
  | "childlike"
  | "decorative";

export interface PerspectiveAnalysis {
  applies: boolean;
  kind?: string;
  vanishing_points?: Array<{ description: string; consistency: string }>;
  assessment: string;
  confidence?: number;
  evidence?: string[];
  [key: string]: unknown;
}

export interface DimensionAnalysis {
  applies?: boolean;
  assessment: string;
  confidence?: number;
  evidence?: string[];
  [key: string]: unknown;
}

export interface Layer1Technique {
  perspective?: PerspectiveAnalysis;
  composition?: DimensionAnalysis;
  color?: DimensionAnalysis;
  line_brushwork?: DimensionAnalysis;
}

export interface SuggestionItem {
  issue: string;
  principle: string;
  action: string;
  difficulty?: string;
  location_hint?: string;
}

export interface ArtworkAnalysisReport {
  framework?: string;
  overall_assessment?: string;
  layer1_technique?: Layer1Technique;
  layer2_style_mood?: {
    mood?: string;
    mood_evidence?: Record<string, string>;
    style_affinity?: string[];
    caveat?: string;
  };
  layer3_suggestions?: { priority_items?: SuggestionItem[] };
  disclaimer?: string;
  [key: string]: unknown;
}

export interface AnalysisStageEvent {
  type: "stage";
  stage: string;
  label: string;
  detail?: string;
}

export interface AnalysisMetricsEvent {
  type: "metrics";
  dominant_colors: Array<{ hex: string; ratio: number }>;
  [key: string]: unknown;
}

export interface AnalysisDoneEvent {
  type: "done";
  image_id: string;
  report: ArtworkAnalysisReport;
  gate?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  focus?: string;
  cached?: boolean;
}

export interface AnalysisRejectedEvent {
  type: "rejected";
  reason: string;
  guide?: string;
}

export interface AnalysisErrorEvent {
  type: "error";
  message: string;
}

export type AnalysisEvent =
  | AnalysisStageEvent
  | AnalysisMetricsEvent
  | AnalysisDoneEvent
  | AnalysisRejectedEvent
  | AnalysisErrorEvent;
