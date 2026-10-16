"use client";

/* eslint-disable @next/next/no-img-element -- static export has no image optimizer; the API
   already serves pre-sized thumbnails and crops (from a CDN in production). */

import { useEffect, useState } from "react";
import type { Match } from "@/lib/api";
import { downloadFile, makeShareCard, shareText } from "@/lib/shareCard";
import styles from "./MatchResults.module.css";

type Props = {
  selfieFace: string | null; // object URL of the cropped selfie face
  matches: Match[];
  onReset: () => void;
};

function caption(m: Match): string | null {
  const parts = [m.title, m.artist, m.year].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}

export default function MatchResults({ selfieFace, matches, onReset }: Props) {
  const [selected, setSelected] = useState(0);
  const match = matches[selected];
  // file is null when the image couldn't be made; sharing then falls back to text and a link.
  const [card, setCard] = useState<{ match: Match; file: File | null } | null>(null);
  const [copied, setCopied] = useState(false);
  const preparing = card?.match !== match;
  const shareFile = preparing ? null : card.file;
  const canShareFile = shareFile !== null && navigator.canShare?.({ files: [shareFile] });

  // Render the image ahead of time: browsers only open the share sheet straight from a click,
  // and drawing it after the click can take long enough to lose that.
  useEffect(() => {
    let cancelled = false;
    makeShareCard(selfieFace, match).then(
      (file) => !cancelled && setCard({ match, file }),
      (err) => {
        console.error("Couldn't make the share image", err);
        if (!cancelled) setCard({ match, file: null });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [selfieFace, match]);

  async function share() {
    const text = shareText(match, window.location.origin);
    const data = shareFile && canShareFile ? { files: [shareFile], text } : shareFile ? null : { text };
    if (data && navigator.share) {
      try {
        await navigator.share(data);
        return;
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return; // closed the share sheet
      }
    }
    if (shareFile) {
      downloadFile(shareFile);
    } else {
      await navigator.clipboard?.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  return (
    <section className={styles.results} aria-labelledby="results-heading">
      <h2 id="results-heading" className={styles.heading}>
        {selected === 0 ? "Your historical twin" : "Another resemblance"}
      </h2>

      <div className={styles.pair}>
        <figure className={styles.face}>
          {selfieFace ? <img src={selfieFace} alt="Your face" /> : <div className={styles.placeholder} />}
          <figcaption>You</figcaption>
        </figure>
        <figure className={styles.face}>
          <img src={match.crop_url} alt="The matching face in the painting" />
          <figcaption>The painting</figcaption>
        </figure>
      </div>

      <a className={styles.frame} href={match.thumb_url} target="_blank" rel="noreferrer">
        <img src={match.thumb_url} alt="The full painting" />
      </a>
      {caption(match) && <p className={styles.caption}>{caption(match)}</p>}
      {match.museum && (
        <p className={styles.credit}>
          {match.url ? (
            <a href={match.url} target="_blank" rel="noreferrer">
              {match.museum}
            </a>
          ) : (
            match.museum
          )}
        </p>
      )}

      {matches.length > 1 && (
        <>
          <h3 className={styles.subheading}>Your closest matches</h3>
          <ul className={styles.grid}>
            {matches.map((m, i) => (
              <li key={m.face_id}>
                <button
                  type="button"
                  className={styles.thumb}
                  aria-pressed={i === selected}
                  aria-label={`Show match ${i + 1}`}
                  onClick={() => setSelected(i)}
                >
                  <img src={m.crop_url} alt="" loading="lazy" />
                  <span>#{i + 1}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      <div className={styles.actions}>
        <button type="button" className="button primary" onClick={share} disabled={preparing}>
          {copied ? "Link copied" : shareFile && !canShareFile ? "Download image" : "Share"}
        </button>
        <button type="button" className="button secondary" onClick={onReset}>
          Try another photo
        </button>
      </div>
    </section>
  );
}
