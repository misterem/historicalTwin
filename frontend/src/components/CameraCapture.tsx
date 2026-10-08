"use client";

import { useEffect, useRef, useState } from "react";
import { captureFrame } from "@/lib/image";
import styles from "./CameraCapture.module.css";

type Props = {
  onCapture: (photo: Blob) => void;
  onCancel: () => void;
  onError: (message: string) => void;
};

function cameraErrorMessage(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError") {
    return "Camera access was blocked. Allow it in your browser settings, or upload a photo instead.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No camera was found. Upload a photo instead.";
  }
  if (name === "NotReadableError") {
    return "Your camera is being used by another app. Close it and try again, or upload a photo.";
  }
  return "The camera couldn't start. Upload a photo instead.";
}

export default function CameraCapture({ onCapture, onCancel, onError }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [ready, setReady] = useState(false);
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      // Browsers only expose the camera on https:// (and localhost).
      onError("The camera needs a secure (https) connection. Upload a photo instead.");
      return;
    }

    let stream: MediaStream | null = null;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 1280 } },
        audio: false,
      })
      .then((s) => {
        stream = s;
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current!;
        video.srcObject = s;
        return video.play();
      })
      .catch((err) => {
        if (!cancelled) onError(cameraErrorMessage(err));
      });

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [onError]);

  async function capture() {
    if (!videoRef.current || capturing) return;
    setCapturing(true);
    try {
      onCapture(await captureFrame(videoRef.current));
    } catch {
      setCapturing(false);
      onError("Couldn't capture the photo. Please try again.");
    }
  }

  return (
    <div className={styles.camera}>
      <div className={styles.viewport}>
        <video
          ref={videoRef}
          className={styles.video}
          playsInline
          muted
          onPlaying={() => setReady(true)}
          aria-label="Camera preview"
        />
        <div className={styles.guide} aria-hidden="true" />
        {!ready && <p className={styles.starting}>Starting camera…</p>}
      </div>
      <p className={styles.hint}>Center your face in the oval, look at the camera, and use even light.</p>
      <div className={styles.actions}>
        <button type="button" className="button secondary" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" className="button primary" onClick={capture} disabled={!ready || capturing}>
          Take photo
        </button>
      </div>
    </div>
  );
}
