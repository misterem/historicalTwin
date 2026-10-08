"use client";

/* eslint-disable @next/next/no-img-element -- local object URL preview, nothing to optimize */

import { useCallback, useEffect, useRef, useState } from "react";
import { MatchError, matchSelfie, warmUpApi, type Match } from "@/lib/api";
import { cropFace, prepareUpload } from "@/lib/image";
import CameraCapture from "./CameraCapture";
import MatchResults from "./MatchResults";
import styles from "./Matcher.module.css";

const NUM_MATCHES = 7; // the top match + 6 runners-up

type State =
  | { step: "start"; error?: string }
  | { step: "camera" }
  | { step: "matching"; preview: string }
  | { step: "results"; selfieFace: string | null; matches: Match[] };

export default function Matcher() {
  const [state, setState] = useState<State>({ step: "start" });
  const fileInput = useRef<HTMLInputElement>(null);
  const request = useRef<AbortController | null>(null);
  const objectUrls = useRef<string[]>([]);

  useEffect(warmUpApi, []);

  const releaseObjectUrls = useCallback(() => {
    objectUrls.current.forEach(URL.revokeObjectURL);
    objectUrls.current = [];
  }, []);

  useEffect(() => releaseObjectUrls, [releaseObjectUrls]);

  const showError = useCallback((error: string) => setState({ step: "start", error }), []);

  const reset = useCallback(() => {
    request.current?.abort();
    releaseObjectUrls();
    setState({ step: "start" });
  }, [releaseObjectUrls]);

  const findMatch = useCallback(
    async (photo: Blob) => {
      request.current?.abort();
      releaseObjectUrls();
      const controller = new AbortController();
      request.current = controller;

      const upload = await prepareUpload(photo);
      const preview = URL.createObjectURL(upload);
      objectUrls.current.push(preview);
      setState({ step: "matching", preview });

      try {
        const result = await matchSelfie(upload, NUM_MATCHES, controller.signal);
        const selfieFace = await cropFace(upload, result.selfie_face);
        if (selfieFace) objectUrls.current.push(selfieFace);
        if (!controller.signal.aborted) setState({ step: "results", selfieFace, matches: result.matches });
      } catch (err) {
        if (controller.signal.aborted) return;
        showError(err instanceof MatchError ? err.message : "Something went wrong. Please try again.");
      }
    },
    [releaseObjectUrls, showError],
  );

  function onFileChosen(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow choosing the same file again
    if (file) findMatch(file);
  }

  return (
    <div className={styles.matcher}>
      <input ref={fileInput} type="file" accept="image/*" hidden onChange={onFileChosen} />

      {state.step === "start" && (
        <div className={styles.start}>
          {state.error && (
            <p className={styles.error} role="alert">
              {state.error}
            </p>
          )}
          <div className={styles.actions}>
            <button type="button" className="button primary" onClick={() => setState({ step: "camera" })}>
              Take a selfie
            </button>
            <button type="button" className="button secondary" onClick={() => fileInput.current?.click()}>
              Upload a photo
            </button>
          </div>
        </div>
      )}

      {state.step === "camera" && <CameraCapture onCapture={findMatch} onCancel={reset} onError={showError} />}

      {state.step === "matching" && (
        <div className={styles.matching} aria-live="polite">
          <img className={styles.preview} src={state.preview} alt="Your photo" />
          <p>Searching the portraits for your twin…</p>
        </div>
      )}

      {state.step === "results" && (
        <MatchResults selfieFace={state.selfieFace} matches={state.matches} onReset={reset} />
      )}
    </div>
  );
}
