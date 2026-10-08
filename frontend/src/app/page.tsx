import Matcher from "@/components/Matcher";

export default function Home() {
  return (
    <main className="page">
      <header className="hero">
        <h1>Historical Twin</h1>
        <p>Take a selfie and meet the portrait painting that looks the most like you.</p>
      </header>

      <Matcher />

      <footer className="footer">
        <p>Your photo is analyzed in memory and never stored.</p>
        <p>Portraits from WikiArt.</p>
      </footer>
    </main>
  );
}
