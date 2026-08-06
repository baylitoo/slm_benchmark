import Link from "next/link";

export const metadata = { title: "Not found · DocIE Studio" };

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-3 bg-background px-6 text-center">
      <p className="text-5xl font-semibold text-accent">404</p>
      <h1 className="text-lg font-medium text-foreground">Page not found</h1>
      <p className="max-w-sm text-sm text-muted-foreground">
        This route doesn&apos;t exist in DocIE Studio.
      </p>
      <Link
        href="/"
        className="mt-2 rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-foreground transition hover:opacity-90"
      >
        Back to Studio
      </Link>
    </div>
  );
}
