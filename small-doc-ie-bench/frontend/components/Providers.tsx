"use client";

import { ThemeProvider } from "next-themes";
import { ToastProvider } from "./Toast";

/** Client-side app providers: theme (dark by default) + toasts. */
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
    >
      <ToastProvider>{children}</ToastProvider>
    </ThemeProvider>
  );
}
