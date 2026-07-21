import { z } from "zod";

export const candidateSummarySchema = z.object({
  perfume_id: z.number().int(),
  label: z.string(),
  name: z.string(),
  brand: z.string(),
});

export const chatResponseSchema = z.object({
  request_id: z.string(),
  session_id: z.string(),
  answer: z.string(),
  route: z.string(),
  language: z.string(),
  recommendations: z.array(candidateSummarySchema),
  validation_passed: z.boolean(),
  generation_attempts: z.number().int(),
  total_seconds: z.number(),
  debug: z.record(z.unknown()).nullable().optional(),
});

export const chatJobAcceptedSchema = z.object({
  job_id: z.string(),
  status: z.literal("queued"),
  poll_after_ms: z.number().int().positive(),
});

export const chatJobStatusSchema = z.object({
  job_id: z.string(),
  status: z.enum(["queued", "running", "succeeded", "failed"]),
  response: chatResponseSchema.nullable().optional(),
  error: z.string().nullable().optional(),
  error_status: z.number().int().nullable().optional(),
  poll_after_ms: z.number().int().positive(),
});

export const warmupJobAcceptedSchema = z.object({
  job_id: z.string(),
  status: z.enum(["queued", "running"]),
  poll_after_ms: z.number().int().positive(),
});

export const warmupJobStatusSchema = z.object({
  job_id: z.string(),
  status: z.enum(["queued", "running", "succeeded", "failed"]),
  ready: z.boolean(),
  report: z.record(z.unknown()).nullable().optional(),
  error: z.string().nullable().optional(),
  poll_after_ms: z.number().int().positive(),
});

export type CandidateSummary = z.infer<typeof candidateSummarySchema>;
export type ChatResponse = z.infer<typeof chatResponseSchema>;
export type ChatJobStatus = z.infer<typeof chatJobStatusSchema>;
export type WarmupJobStatus = z.infer<typeof warmupJobStatusSchema>;

export interface ChatRequest {
  query: string;
  session_id?: string;
  debug?: boolean;
}

export interface ConnectionConfig {
  apiUrl: string;
  apiKey: string;
}
