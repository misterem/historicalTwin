export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** Face bounding box, normalized to 0–1 relative to its image. */
export type Box = { x1: number; y1: number; x2: number; y2: number };

export type Match = Box & {
  image_id: string;
  image_path: string;
  face_idx: number;
  det_score: number;
  face_px: number;
  face_id: number;
  score: number; // cosine similarity
  thumb_url: string;
  crop_url: string;
  // Not returned by the API yet: the dataset has no painting metadata.
  title?: string;
  artist?: string;
  year?: string;
};

export type MatchResponse = { selfie_face: Box; matches: Match[] };

export type MatchErrorKind = "no-face" | "bad-image" | "network" | "server";

export class MatchError extends Error {
  kind: MatchErrorKind;

  constructor(kind: MatchErrorKind, message: string) {
    super(message);
    this.kind = kind;
  }
}

/** The API returns root-relative image URLs in local dev and absolute CDN URLs in production. */
function resolveImageUrl(url: string): string {
  return url.startsWith("/") ? `${API_URL}${url}` : url;
}

export async function matchSelfie(photo: Blob, k: number, signal?: AbortSignal): Promise<MatchResponse> {
  const body = new FormData();
  body.append("file", photo, photo instanceof File ? photo.name : "selfie.jpg");

  let res: Response;
  try {
    res = await fetch(`${API_URL}/match?k=${k}`, { method: "POST", body, signal });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new MatchError("network", "Couldn't reach the server. Check your connection and try again.");
  }

  if (!res.ok) {
    if (res.status === 422) {
      throw new MatchError("no-face", "We couldn't find a face in that photo. Try facing the camera in good light.");
    }
    if (res.status === 400 || res.status === 413) {
      throw new MatchError("bad-image", "That photo couldn't be read. Try a different one.");
    }
    throw new MatchError("server", "Something went wrong on our side. Please try again.");
  }

  const data = (await res.json()) as MatchResponse;
  return {
    ...data,
    matches: data.matches.map((m) => ({
      ...m,
      thumb_url: resolveImageUrl(m.thumb_url),
      crop_url: resolveImageUrl(m.crop_url),
    })),
  };
}

/** Fire-and-forget request so a scaled-to-zero API starts loading its models early. */
export function warmUpApi(): void {
  fetch(`${API_URL}/health`).catch(() => {});
}
