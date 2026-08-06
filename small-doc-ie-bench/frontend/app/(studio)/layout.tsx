import { AppShell } from "@/components/AppShell";

/**
 * The Studio shell lives in this LAYOUT, not in the pages: layouts persist
 * across route navigation, so every section (and its in-flight extraction /
 * deploy / benchmark work) survives URL changes exactly like the old
 * setActive() model — while the URL becomes the navigation source of truth
 * (deep links, back button, refresh). The route pages below render nothing;
 * AppShell reads the pathname itself.
 */
export default function StudioLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <AppShell />
      {children}
    </>
  );
}
