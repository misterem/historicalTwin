"use client";

/* eslint-disable @next/next/no-img-element -- static export has no image optimizer; the API
   already serves pre-sized thumbnails and crops (from a CDN in production). */

import { useState } from "react";
import type { Match } from "@/lib/api";
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
      <p className={styles.score}>Similarity {match.score.toFixed(2)}</p>

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
                  aria-label={`Show match ${i + 1}, similarity ${m.score.toFixed(2)}`}
                  onClick={() => setSelected(i)}
                >
                  <img src={m.crop_url} alt="" loading="lazy" />
                  <span>{m.score.toFixed(2)}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      <button type="button" className="button primary" onClick={onReset}>
        Try another photo
      </button>
    </section>
  );
}
