import { useState } from "react";

// The rollout in the stage. Until its first frame arrives the well says so, rather
// than sitting as an empty black box.
export default function RolloutVideo({ src }: { src: string }) {
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <>
      <video
        src={src}
        className={ready ? "ready" : ""}
        controls
        muted
        loop
        autoPlay
        playsInline
        preload="auto"
        onLoadedData={() => setReady(true)}
        onError={() => setFailed(true)}
      />
      {!ready && (
        <div className="well-empty video-wait" role="status">
          <strong>{failed ? "The video didn't load" : "Loading video…"}</strong>
          {failed && <span>Reload the page to try again.</span>}
        </div>
      )}
    </>
  );
}
