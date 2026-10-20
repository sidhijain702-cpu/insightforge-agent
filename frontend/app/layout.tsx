import type { Metadata } from "next";
import { Instrument_Sans, Newsreader } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const sans = Instrument_Sans({ subsets: ["latin"], variable: "--font-sans" });
const serif = Newsreader({ subsets: ["latin"], variable: "--font-serif" });

export const metadata: Metadata = { title: "InsightForge", description: "Autonomous market intelligence" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body>
        <header className="nav">
          <Link href="/" className="brand">InsightForge</Link>
          <nav>
            <Link href="/">New brief</Link>
            <Link href="/reports">Reports</Link>
            <Link href="/watchlist">Watchlist</Link>
          </nav>
        </header>
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
