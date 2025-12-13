export type DatasetStats = {
  rows: number;
  features: number;
  x_min: number[];
  x_max: number[];
  y_min: number;
  y_max: number;
};

export type UploadResponse = {
  message: string;
  stored_filename: string;
  original_filename: string;
  stats: DatasetStats;
};

export type TrainResponse = {
  message: string;
  train_loss_start?: number;
  train_loss_end?: number;
  val_loss_start?: number;
  val_loss_end?: number;
  plots: string[];
  best_val_rmse?: number;
  best_train_rmse?: number;
  best_epoch?: number;
  epochs_run?: number;
  baseline_rmse?: number;
  artifacts?: string[];
  final_train_r2?: number;
  final_val_r2?: number;
  model_type?: "numpy" | "tf";
  duration_ms?: number;
};

export type TrainJobResponse = {
  job_id: string;
  status: "queued" | "started" | "finished" | "failed";
  backend?: "numpy" | "tf";
};

export type JobStatusResponse = TrainJobResponse & {
  result?: TrainResponse;
  error?: string;
};

export type PredictResponse = { y_pred: number[][]; model_type?: "numpy" | "tf" };

class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

const handle = async <T>(res: Response): Promise<T> => {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(data?.error || `Request failed with ${res.status}`, res.status);
  }
  return data as T;
};

export async function healthCheck(baseUrl: string) {
  const res = await fetch(`${baseUrl}/health`, { cache: "no-store" });
  return handle<{ status: string; version: string }>(res);
}

export async function uploadDataset(baseUrl: string, file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${baseUrl}/upload`, {
    method: "POST",
    body: formData,
  });
  return handle<UploadResponse>(res);
}

export async function trainModel(baseUrl: string, formData: FormData): Promise<TrainResponse | TrainJobResponse> {
  const res = await fetch(`${baseUrl}/train`, {
    method: "POST",
    body: formData,
  });
  return handle<TrainResponse | TrainJobResponse>(res);
}

export async function getJobStatus(baseUrl: string, jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${baseUrl}/jobs/${jobId}`, { cache: "no-store" });
  return handle<JobStatusResponse>(res);
}

export async function predictModel(baseUrl: string, formData: FormData): Promise<PredictResponse> {
  const res = await fetch(`${baseUrl}/predict`, {
    method: "POST",
    body: formData,
  });
  return handle<PredictResponse>(res);
}

export async function resetBackend(baseUrl: string) {
  const res = await fetch(`${baseUrl}/reset`, { method: "POST" });
  return handle<{ message: string }>(res);
}
