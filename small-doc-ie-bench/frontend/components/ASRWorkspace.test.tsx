import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ASRWorkspace } from "@/components/ASRWorkspace";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { ASRJob, ASRJobItem, DeploymentRecord } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getDeployments: vi.fn(),
    listASRJobs: vi.fn(),
    getASRJob: vi.fn(),
    createASRJob: vi.fn(),
    cancelASRJob: vi.fn(),
    downloadASRArtifact: vi.fn(),
    getRealtimeToken: vi.fn(),
    fileToBase64: vi.fn(async (file: File) => `B64<${file.name}>`),
  };
});

function deployment(
  name: string,
  runtime: string,
  state = "ready",
  endpoint: string | null = "http://127.0.0.1:9000",
): DeploymentRecord {
  return {
    spec: { name, launch: { runtime, model: `${name}-model` } },
    state,
    endpoint,
  };
}

function item(overrides: Partial<ASRJobItem> = {}): ASRJobItem {
  return {
    position: 0,
    filename: "meeting.wav",
    input_sha256: "abc",
    input_size_bytes: 44,
    raw_available: false,
    mime_type: "audio/wav",
    reference: "hello world",
    status: "completed",
    detected_language: "en",
    duration_seconds: 2,
    processing_seconds: 0.5,
    result: {
      text: "hello world",
      language: "en",
      duration: 2,
      segments: [{ id: 0, start: 0, end: 1.25, text: "hello world" }],
      processing_seconds: 0.5,
      real_time_factor: 0.25,
      model: "whisper-small",
      backend: "faster-whisper",
    },
    metrics: {
      wer: 0,
      cer: 0,
      word_errors: 0,
      reference_words: 2,
      character_errors: 0,
      reference_characters: 11,
    },
    error: null,
    attempts: 1,
    started_at: "2026-08-26T10:00:01Z",
    completed_at: "2026-08-26T10:00:02Z",
    artifacts: [
      {
        id: "artifact-text",
        name: "0000-meeting.txt",
        kind: "text",
        sha256: "def",
        size_bytes: 12,
        media_type: "text/plain",
        uri: "/v1/audio/transcription-jobs/asr-1/artifacts/artifact-text",
      },
      {
        id: "artifact-srt",
        name: "0000-meeting.srt",
        kind: "srt",
        sha256: "ghi",
        size_bytes: 40,
        media_type: "application/x-subrip",
        uri: "/v1/audio/transcription-jobs/asr-1/artifacts/artifact-srt",
      },
    ],
    ...overrides,
  };
}

function job(overrides: Partial<ASRJob> = {}): ASRJob {
  return {
    event_id: "asr-1",
    channel: "asr:asr-1",
    deployment: "whisper-live",
    model: "whisper-small",
    status: "completed",
    total_items: 1,
    completed_items: 1,
    failed_items: 0,
    options: { temperature: 0 },
    metrics: {
      completed_items: 1,
      failed_items: 0,
      scored_items: 1,
      audio_seconds: 2,
      processing_seconds: 0.5,
      real_time_factor: 0.25,
      wer: 0,
      cer: 0,
      character_errors: 0,
    },
    error: null,
    raw_retention: "delete_after_completion",
    raw_expires_at: "2026-08-26T10:00:02Z",
    cancel_requested_at: null,
    created_at: "2026-08-26T10:00:00Z",
    started_at: "2026-08-26T10:00:01Z",
    completed_at: "2026-08-26T10:00:02Z",
    updated_at: "2026-08-26T10:00:02Z",
    artifacts: [],
    items: [item()],
    ...overrides,
  };
}

function renderWorkspace() {
  return render(
    <ToastProvider>
      <ASRWorkspace />
    </ToastProvider>,
  );
}

describe("ASRWorkspace", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.mocked(api.getDeployments).mockReset().mockResolvedValue([
      deployment("whisper-live", "asr"),
      deployment("chat-live", "llama_cpp"),
      deployment("whisper-stopped", "asr", "stopped", null),
    ]);
    vi.mocked(api.listASRJobs).mockReset().mockResolvedValue([]);
    vi.mocked(api.getASRJob).mockReset();
    vi.mocked(api.createASRJob).mockReset().mockResolvedValue({
      job_id: "asr-new",
      status: "queued",
      channel: "asr:asr-new",
      topics: ["status", "progress", "result", "error"],
      status_uri: "/v1/audio/transcription-jobs/asr-new",
      deduplicated: false,
    });
    vi.mocked(api.cancelASRJob).mockReset();
    vi.mocked(api.downloadASRArtifact).mockReset().mockResolvedValue(undefined);
    vi.mocked(api.getRealtimeToken).mockReset().mockRejectedValue(new Error("realtime offline"));
    vi.mocked(api.fileToBase64).mockClear();
  });

  it("offers only ready, reachable ASR deployments and has no free-form model field", async () => {
    renderWorkspace();
    const select = await screen.findByLabelText(/Live ASR deployment/);
    expect(within(select).getByRole("option", { name: "whisper-live" })).toBeInTheDocument();
    expect(within(select).queryByRole("option", { name: "chat-live" })).not.toBeInTheDocument();
    expect(within(select).queryByRole("option", { name: "whisper-stopped" })).not.toBeInTheDocument();
    expect(select).toHaveValue("whisper-live");
  });

  it("always submits uploads through the durable job endpoint with per-file references", async () => {
    renderWorkspace();
    await screen.findByRole("option", { name: "whisper-live" });
    const audio = new File(["RIFF0000WAVE"], "interview.wav", { type: "audio/wav" });
    await userEvent.upload(screen.getByLabelText(/Audio recordings/), audio);
    await userEvent.type(
      screen.getByLabelText("Reference transcript for interview.wav"),
      "Bonjour le monde",
    );
    await userEvent.type(screen.getByLabelText("Language"), "fr");
    await userEvent.clear(screen.getByLabelText("Temperature"));
    await userEvent.type(screen.getByLabelText("Temperature"), "0.2");
    await userEvent.click(screen.getByRole("button", { name: /Start transcription/ }));

    await waitFor(() => expect(api.createASRJob).toHaveBeenCalledTimes(1));
    expect(api.createASRJob).toHaveBeenCalledWith({
      model: "whisper-live",
      recordings: [
        {
          filename: "interview.wav",
          content_b64: "B64<interview.wav>",
          reference: "Bonjour le monde",
        },
      ],
      language: "fr",
      temperature: 0.2,
      raw_audio_retention: "delete_after_completion",
    });
    expect(window.localStorage.getItem("docie-studio-asr-last-job")).toBe("asr-new");
  });

  it("recovers the persisted durable job after refresh and renders timestamps and evaluation", async () => {
    const completed = job();
    window.localStorage.setItem("docie-studio-asr-last-job", completed.event_id);
    vi.mocked(api.listASRJobs).mockResolvedValue([completed]);
    vi.mocked(api.getASRJob).mockResolvedValue(completed);
    renderWorkspace();

    await waitFor(() => expect(api.getASRJob).toHaveBeenCalledWith("asr-1"));
    expect(await screen.findByText("Real-time factor")).toBeInTheDocument();
    expect(screen.getByText("0.250")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /meeting\.wav/ }));
    expect((await screen.findAllByText("hello world")).length).toBeGreaterThan(0);
    expect(screen.getByText(/0\.00s → 1\.25s/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /TEXT/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /SRT/ })).toBeInTheDocument();
  });

  it("keeps a bad recording visible without hiding successful batch output", async () => {
    const failed = item({
      position: 1,
      filename: "broken.mp3",
      status: "failed",
      result: null,
      metrics: null,
      error: "ASR runtime rejected corrupt audio",
      detected_language: null,
      duration_seconds: null,
      processing_seconds: null,
      artifacts: [],
    });
    const partial = job({
      status: "completed_with_errors",
      total_items: 2,
      completed_items: 1,
      failed_items: 1,
      items: [item(), failed],
      metrics: { ...job().metrics, completed_items: 1, failed_items: 1 },
    });
    window.localStorage.setItem("docie-studio-asr-last-job", partial.event_id);
    vi.mocked(api.listASRJobs).mockResolvedValue([partial]);
    vi.mocked(api.getASRJob).mockResolvedValue(partial);
    renderWorkspace();

    expect((await screen.findAllByText("completed with errors")).length).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("button", { name: /broken\.mp3/ }));
    expect(await screen.findByText("ASR runtime rejected corrupt audio")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /meeting\.wav/ }));
    expect((await screen.findAllByText("hello world")).length).toBeGreaterThan(0);
  });

  it("downloads artifacts through the authenticated API helper", async () => {
    const completed = job();
    window.localStorage.setItem("docie-studio-asr-last-job", completed.event_id);
    vi.mocked(api.listASRJobs).mockResolvedValue([completed]);
    vi.mocked(api.getASRJob).mockResolvedValue(completed);
    renderWorkspace();
    await userEvent.click(await screen.findByRole("button", { name: /meeting\.wav/ }));
    await userEvent.click(await screen.findByRole("button", { name: /TEXT/ }));
    expect(api.downloadASRArtifact).toHaveBeenCalledWith(
      expect.objectContaining({ id: "artifact-text", uri: expect.stringContaining("/artifacts/") }),
    );
  });

  it("requests realtime for a running job, retains polling fallback, and supports cancellation", async () => {
    const running = job({
      status: "running",
      completed_at: null,
      completed_items: 0,
      metrics: null,
      items: [item({ status: "running", result: null, metrics: null, artifacts: [] })],
    });
    window.localStorage.setItem("docie-studio-asr-last-job", running.event_id);
    vi.mocked(api.listASRJobs).mockResolvedValue([running]);
    vi.mocked(api.getASRJob).mockResolvedValue(running);
    vi.mocked(api.cancelASRJob).mockResolvedValue({ ...running, status: "cancelling" });
    renderWorkspace();

    await waitFor(() => expect(api.getRealtimeToken).toHaveBeenCalledWith(
      "asr:asr-1",
      ["status", "progress", "result", "error"],
    ));
    expect(await screen.findByText("polling")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Cancel/ }));
    expect(api.cancelASRJob).toHaveBeenCalledWith("asr-1");
  });
});
