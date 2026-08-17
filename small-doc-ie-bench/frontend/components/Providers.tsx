"use client";

import { ThemeProvider } from "next-themes";
import { ToastProvider } from "./Toast";
import { I18nProvider } from "@/lib/i18n";

/** Client-side app providers: theme (light by default) + toasts. */
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="light"
      enableSystem={false}
      disableTransitionOnChange
    >
      <I18nProvider>
        <ToastProvider>{children}</ToastProvider>
      </I18nProvider>
    </ThemeProvider>
  );
}
