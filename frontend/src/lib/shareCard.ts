import type { Match } from "./api";

// A square image of the user's face beside the painting's, drawn entirely in the browser so the
// selfie never leaves the device until the user picks where to share it. It uses the site's
// "Portrait Gallery" look: oxblood damask, oval gilt frames and a brass plaque.
const SIZE = 1080;
const OVAL = { rx: 165, ry: 206, gap: 50, cy: 452 }; // outer edge of each gilt frame
const FRAME = 14; // width of the gilding
const PLAQUE_TOP = 770;
const TEXT_WIDTH = 820;
const COLORS = {
  wall: "#4d1519", wallDark: "#2e0b0e", muted: "#d2b98f", goldText: "#f0d58f", ink: "#3d2a0c",
  pattern: "rgba(232, 201, 138, 0.10)",
};
// The damask motif from globals.css, as SVG path data for a 60x80 tile.
const DAMASK = "M30 4C44 20 44 30 30 40 16 30 16 20 30 4ZM30 40C44 50 44 60 30 76 16 60 16 50 30 40Z" +
  "M0 40C14 30 14 20 0 4M60 40C46 30 46 20 60 4M0 40C14 50 14 60 0 76M60 40C46 50 46 60 60 76";

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

function drawWall(ctx: CanvasRenderingContext2D) {
  const wall = ctx.createRadialGradient(SIZE / 2, SIZE * 0.2, 0, SIZE / 2, SIZE * 0.2, SIZE * 0.95);
  wall.addColorStop(0, COLORS.wall);
  wall.addColorStop(1, COLORS.wallDark);
  ctx.fillStyle = wall;
  ctx.fillRect(0, 0, SIZE, SIZE);

  const tile = document.createElement("canvas");
  tile.width = 90;
  tile.height = 120;
  const t = tile.getContext("2d")!;
  t.scale(1.5, 1.5);
  t.strokeStyle = COLORS.pattern;
  t.lineWidth = 1.4;
  t.stroke(new Path2D(DAMASK));
  t.beginPath();
  t.arc(30, 40, 3, 0, Math.PI * 2);
  t.stroke();
  ctx.fillStyle = ctx.createPattern(tile, "repeat")!;
  ctx.fillRect(0, 0, SIZE, SIZE);
}

function gold(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number): CanvasGradient {
  const g = ctx.createLinearGradient(x0, y0, x1, y1);
  [[0, "#7a5a20"], [0.25, "#f3d58a"], [0.5, "#a8822f"], [0.75, "#f0d184"], [1, "#7a5a20"]].forEach(([at, color]) =>
    g.addColorStop(at as number, color as string),
  );
  return g;
}

function ellipse(ctx: CanvasRenderingContext2D, cx: number, cy: number, inset: number) {
  ctx.beginPath();
  ctx.ellipse(cx, cy, OVAL.rx - inset, OVAL.ry - inset, 0, 0, Math.PI * 2);
}

/** An oval gilt frame around a face, like the ones on the results page. */
function drawFace(ctx: CanvasRenderingContext2D, image: ImageBitmap | null, cx: number, label: string, labelFont: string) {
  const { cy, rx, ry } = OVAL;
  ctx.save();
  ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
  ctx.shadowBlur = 30;
  ctx.shadowOffsetY = 14;
  ellipse(ctx, cx, cy, 0);
  ctx.fillStyle = gold(ctx, cx - rx, cy - ry, cx + rx, cy + ry);
  ctx.fill();
  ctx.restore();

  // Moldings: a dark outer edge, a bright bead and a dark inner lip.
  for (const [inset, color, width] of [[1, "#4a3410", 2], [5, "#e9c97a", 3], [8, "#6b4c1a", 2]] as const) {
    ellipse(ctx, cx, cy, inset);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  ctx.save();
  ellipse(ctx, cx, cy, FRAME);
  ctx.fillStyle = COLORS.wallDark;
  ctx.fill();
  if (image) {
    ctx.clip();
    // The crops are square and the oval is taller than wide, so fill its height.
    const side = 2 * (ry - FRAME);
    ctx.drawImage(image, cx - side / 2, cy - side / 2, side, side);
  }
  ctx.restore();

  ctx.font = labelFont;
  ctx.fillStyle = COLORS.muted;
  ctx.letterSpacing = "4px";
  ctx.fillText(label, cx, cy + ry + 56);
  ctx.letterSpacing = "0px";
}

/** A thin gold line with a bead in the middle, as under the site's headings. */
function drawRule(ctx: CanvasRenderingContext2D, y: number) {
  ctx.fillStyle = "#d8b468";
  ctx.fillRect(SIZE / 2 - 100, y, 84, 2);
  ctx.fillRect(SIZE / 2 + 16, y, 84, 2);
  ctx.beginPath();
  ctx.arc(SIZE / 2, y + 1, 5, 0, Math.PI * 2);
  ctx.fill();
}

/** The brass museum label: the title over the artist and year. */
function drawPlaque(ctx: CanvasRenderingContext2D, match: Match, fonts: { title: string; byline: string }) {
  const byline = [match.artist, match.year].filter(Boolean).join(" · ");
  ctx.font = fonts.title;
  const titleLines = match.title ? wrap(ctx, match.title, TEXT_WIDTH, 2) : [];
  const widths = titleLines.map((line) => ctx.measureText(line).width);
  ctx.font = fonts.byline;
  ctx.letterSpacing = "2px";
  const bylineLine = byline ? wrap(ctx, byline, TEXT_WIDTH, 1)[0] : null;
  if (bylineLine) widths.push(ctx.measureText(bylineLine).width);
  ctx.letterSpacing = "0px";
  if (!widths.length) return;

  const width = Math.max(...widths) + 80;
  const height = 34 + titleLines.length * 50 + (bylineLine ? 40 : 0);
  const left = (SIZE - width) / 2;

  ctx.save();
  ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
  ctx.shadowBlur = 12;
  ctx.shadowOffsetY = 5;
  const brass = ctx.createLinearGradient(0, PLAQUE_TOP, 0, PLAQUE_TOP + height);
  brass.addColorStop(0, "#f6dd9a");
  brass.addColorStop(0.55, "#c9a24a");
  brass.addColorStop(1, "#a8822f");
  ctx.fillStyle = brass;
  ctx.fillRect(left, PLAQUE_TOP, width, height);
  ctx.restore();
  ctx.strokeStyle = "#6b4c1a";
  ctx.lineWidth = 2;
  ctx.strokeRect(left + 1, PLAQUE_TOP + 1, width - 2, height - 2);

  ctx.fillStyle = COLORS.ink;
  let y = PLAQUE_TOP + 54;
  ctx.font = fonts.title;
  for (const line of titleLines) {
    ctx.fillText(line, SIZE / 2, y);
    y += 50;
  }
  if (bylineLine) {
    ctx.font = fonts.byline;
    ctx.letterSpacing = "2px";
    ctx.fillText(bylineLine, SIZE / 2, titleLines.length ? y - 6 : y);
    ctx.letterSpacing = "0px";
  }
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
  const caps = fontFamily("--font-caps", "Georgia, serif");
  const fonts = {
    heading: `800 88px ${display}`,
    title: `600 40px ${display}`,
    label: `700 26px ${caps}`,
    byline: `400 28px ${caps}`,
    site: `700 30px ${caps}`,
  };
  // Only the painting is essential: without the selfie the card shows an empty frame, and
  // without the web fonts it falls back to Georgia.
  const [you, painting] = await Promise.all([
    selfieFace ? loadImage(selfieFace).catch(warn("selfie", null)) : null,
    loadImage(match.crop_url),
    ...Object.values(fonts).map((font) => document.fonts.load(font).catch(warn(`font ${font}`, []))),
  ]);

  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext("2d")!;
  drawWall(ctx);
  ctx.textAlign = "center";

  ctx.font = fonts.heading;
  const heading = ctx.createLinearGradient(0, 80, 0, 160);
  heading.addColorStop(0, "#fbe7ae");
  heading.addColorStop(0.6, "#c9a24a");
  heading.addColorStop(1, "#8d6a26");
  ctx.fillStyle = heading;
  ctx.fillText("My historical twin", SIZE / 2, 150);
  drawRule(ctx, 192);

  const offset = OVAL.rx + OVAL.gap / 2;
  drawFace(ctx, you, SIZE / 2 - offset, "ME", fonts.label);
  drawFace(ctx, painting, SIZE / 2 + offset, "THE PAINTING", fonts.label);
  you?.close();
  painting.close();

  drawPlaque(ctx, match, fonts);

  ctx.font = fonts.site;
  ctx.fillStyle = COLORS.goldText;
  ctx.letterSpacing = "3px";
  ctx.fillText(window.location.host, SIZE / 2, SIZE - 48);

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
