import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Abalone",
  description: "Abalone playground (Rust engine via WebAssembly)",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
