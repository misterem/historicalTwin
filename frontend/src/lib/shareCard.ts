import type { Match } from "./api";

// A square image of the user's face beside the painting's, drawn entirely in the browser so the
// selfie never leaves the device until the user picks where to share it.
const SIZE = 1080;
const FACE = 400; // diameter of each face circle
const FACE_TOP = 230;
const FACE_GAP = 60;
const TEXT_WIDTH = 900;
// The light theme, whatever the viewer's color scheme, so every card looks the same.
const COLORS = { bg: "#f6f1e7", surface: "#fffaf0", text: "#2a2420", muted: "#6f655b", accent: "#8a5a2b", frame: "#b08d57" };

export const SHARE_FILE_NAME = "historical-twin.png";
// Give up on the image after this long, so the share button falls back to a link instead of
// staying disabled.
const TIMEOUT_MS = 10_000;

/** The fonts next/font loaded, as a CSS font-family list. */
function fontFamily(variable: string, fallback: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(variable).trim() || fallback;
}

async function loadImage(url: string): Promise<ImageBitmap> {
  // "reload": the results page shows the crops with plain <img> tags, and their cached copies have
  // no CORS headers. Drawing a non-CORS image would make the canvas impossible to export.
  const res = await fetch(url, { cache: "reload" });
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  return createImageBitmap(await res.blob());
}

/** Split text into lines that fit `maxWidth`, ending with an ellipsis if it needs more than `maxLines`. */
function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, maxLines: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (!line || ctx.measureText(next).width <= maxWidth) {
      line = next;
    } else {
      lines.push(line);
      line = word;
    }
  }
  if (line) lines.push(line);
  if (lines.length <= maxLines) return lines;

  let last = lines[maxLines - 1];
  while (last && ctx.measureText(`${last}…`).width > maxWidth) last = last.slice(0, -1);
  return [...lines.slice(0, maxLines - 1), `${last.trimEnd()}…`];
}

function drawFace(ctx: CanvasRenderingContext2D, image: ImageBitmap | null, left: number, label: string, labelFont: string) {
  const r = FACE / 2;
  const cx = left + r;
  const cy = FACE_TOP + r;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.surface;
  ctx.fill();
  if (image) {
    ctx.clip();
    ctx.drawImage(image, left, FACE_TOP, FACE, FACE);
  }
  ctx.restore();

  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.lineWidth = 8;
  ctx.strokeStyle = COLORS.frame;
  ctx.stroke();

  ctx.font = labelFont;
  ctx.fillStyle = COLORS.muted;
  ctx.letterSpacing = "3px";
  ctx.fillText(label, cx, FACE_TOP + FACE + 64);
  ctx.letterSpacing = "0px";
}

function warn<T>(what: string, fallback: T) {
  return (err: unknown) => {
    console.warn(`Share image: couldn't load the ${what}`, err);
    return fallback;
  };
}

/** Render the share image for one match. `selfieFace` is the object URL of the user's face crop. */
export function makeShareCard(selfieFace: string | null, match: Match): Promise<File> {
  return Promise.race([
    renderCard(selfieFace, match),
    new Promise<never>((_, reject) => setTimeout(() => reject(new Error(`timed out after ${TIMEOUT_MS} ms`)), TIMEOUT_MS)),
  ]);
}

async function renderCard(selfieFace: string | null, match: Match): Promise<File> {
  const display = fontFamily("--font-display", "Georgia, serif");
  const body = fontFamily("--font-body", "system-ui, sans-serif");
  const fonts = {
    heading: `600 84px ${display}`,
    title: `600 56px ${display}`,
    label: `500 28px ${body}`,
    detail: `400 34px ${body}`,
    site: `500 30px ${body}`,
  };
  // Only the painting is essential: without the selfie the card shows an empty circle, and
  // without the web fonts it falls back to Georgia and system-ui.
  const [you, painting] = await Promise.all([
    selfieFace ? loadImage(selfieFace).catch(warn("selfie", null)) : null,
    loadImage(match.crop_url),
    ...Object.values(fonts).map((font) => document.fonts.load(font).catch(warn(`font ${font}`, []))),
  ]);

  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.textAlign = "center";

  ctx.font = fonts.heading;
  ctx.fillStyle = COLORS.text;
  ctx.fillText("My historical twin", SIZE / 2, 160);

  const left = (SIZE - 2 * FACE - FACE_GAP) / 2;
  drawFace(ctx, you, left, "ME", fonts.label);
  drawFace(ctx, painting, left + FACE + FACE_GAP, "THE PAINTING", fonts.label);
  you?.close();
  painting.close();

  let y = FACE_TOP + FACE + 170;
  if (match.title) {
    ctx.font = fonts.title;
    ctx.fillStyle = COLORS.text;
    for (const line of wrap(ctx, match.title, TEXT_WIDTH, 2)) {
      ctx.fillText(line, SIZE / 2, y);
      y += 64;
    }
  }
  const detail = [match.artist, match.year].filter(Boolean).join(" · ");
  if (detail) {
    ctx.font = fonts.detail;
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(wrap(ctx, detail, TEXT_WIDTH, 1)[0], SIZE / 2, y);
  }

  ctx.font = fonts.site;
  ctx.fillStyle = COLORS.accent;
  ctx.fillText(window.location.host, SIZE / 2, SIZE - 60);

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("Could not encode the share image");
  return new File([blob], SHARE_FILE_NAME, { type: "image/png" });
}

/** The message that goes with the image, for apps that accept text too. */
export function shareText(match: Match, siteUrl: string): string {
  const { title, artist } = match;
  const intro = title && artist
    ? `My historical twin is “${title}” by ${artist}.`
    : title
      ? `My historical twin is “${title}”.`
      : artist
        ? `My historical twin was painted by ${artist}.`
        : "I found my historical twin.";
  return `${intro} Find yours at ${siteUrl}`;
}

/** Fallback where the browser can't share files: save the image instead. */
export function downloadFile(file: File): void {
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = file.name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
