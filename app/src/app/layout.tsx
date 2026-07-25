import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Memory-LoRA — repo-personalized model server",
  description:
    "Paste a repo URL. The hypernetwork emits a LoRA adapter, merges it into Gemma, and serves an OpenAI/Anthropic-compatible endpoint for your coding CLI.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
