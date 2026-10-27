import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/Providers";

// The UI shipped with no webfont at all -- the stack was
// `ui-sans-serif, system-ui, Segoe UI, ...`, so it rendered as whatever the
// OS supplied, which is the single loudest "dated" signal on Windows. Inter is
// loaded as a variable font (one file, every weight) and exposed as a CSS
// variable that tailwind.config.ts puts at the front of the sans stack; the
// old stack stays behind it as the fallback.
const sans = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

// tailwind.config.ts already NAMED JetBrains Mono in the mono stack and
// nothing ever loaded it, so every port, digest and model id fell through to
// Consolas. Loading it delivers what the config always claimed.
const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "DocIE Studio",
  description:
    "Document information-extraction playground, deploy & benchmark console.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  // suppressHydrationWarning is required by next-themes (it sets the theme
  // class on <html> before hydration).
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${sans.variable} ${mono.variable}`}
    >
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
