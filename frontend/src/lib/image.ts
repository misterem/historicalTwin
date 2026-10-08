import type { Box } from "./api";

// Phone photos are often 3–12 MB; the API only needs ~800px to find and embed the face.
// Shrinking before upload makes matching much faster on mobile networks.
const UPLOAD_MAX_SIDE = 800;
const JPEG_QUALITY = 0.9;

function canvasToJpeg(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
}

function drawScaled(source: CanvasImageSource, width: number, height: number): HTMLCanvasElement {
  const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.getContext("2d")!.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/** Shrink a chosen photo for upload. Falls back to the original if the browser can't decode it (e.g. HEIC outside Safari). */
export async function prepareUpload(file: Blob): Promise<Blob> {
  try {
    // "from-image" applies EXIF rotation so the face isn't sideways.
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const canvas = drawScaled(bitmap, bitmap.width, bitmap.height);
    bitmap.close();
    const blob = await canvasToJpeg(canvas);
    return blob && blob.size < file.size ? blob : file;
  } catch {
    return file;
  }
}

/** Grab the current camera frame as an upload-sized JPEG (un-mirrored, i.e. as the camera sees it). */
export async function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const canvas = drawScaled(video, video.videoWidth, video.videoHeight);
  const blob = await canvasToJpeg(canvas);
  if (!blob) throw new Error("Could not capture the camera frame");
  return blob;
}

/**
 * Square crop around a face, padded by `margin` of the face size on each side.
 * Mirrors face_crop() in src/paintmatch/faces.py so both crops are framed alike.
 */
export async function cropFace(image: Blob, box: Box, size = 320, margin = 0.3): Promise<string | null> {
  try {
    const bitmap = await createImageBitmap(image, { imageOrientation: "from-image" });
    const { width: w, height: h } = bitmap;
    const [x1, y1, x2, y2] = [box.x1 * w, box.y1 * h, box.x2 * w, box.y2 * h];
    const side = Math.min(Math.max(x2 - x1, y2 - y1) * (1 + 2 * margin), w, h);
    const left = Math.min(Math.max((x1 + x2) / 2 - side / 2, 0), w - side);
    const top = Math.min(Math.max((y1 + y2) / 2 - side / 2, 0), h - side);

    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    canvas.getContext("2d")!.drawImage(bitmap, left, top, side, side, 0, 0, size, size);
    bitmap.close();
    const blob = await canvasToJpeg(canvas);
    return blob ? URL.createObjectURL(blob) : null;
  } catch {
    return null;
  }
}
